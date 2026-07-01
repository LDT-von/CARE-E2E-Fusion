# -*- coding: utf-8 -*-
"""
WSI End-to-End Training for CARE-E2E-Fusion
==========================================
用真实 WSI 图像进行端到端训练。

支持两种模式：
1. lazy: 每次 __getitem__ 按需读取 tiles，tile coords 在首次访问时计算
2. cached: 启动时加载预计算的 tile coords 缓存（快 10 倍）
"""
from __future__ import print_function
import os, sys, random, time as _time, argparse, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from models.fusion_model import E2EViTCAREFusion
from models.utils import print_model_summary

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TILE_SIZE = 256
PATCH_SIZE = 16
BATCH_MAX_TILES = 512


# ============================================================
# WSI utilities
# ============================================================

def build_svs_index(wsi_root):
    idx = {}
    for d in os.listdir(wsi_root):
        full = os.path.join(wsi_root, d)
        if not os.path.isdir(full):
            continue
        for f in os.listdir(full):
            if f.endswith('.svs'):
                pid = f[:12]
                idx.setdefault(pid, []).append((f, full))
    return idx


def find_svs(patient_id, svs_index):
    if patient_id not in svs_index:
        return None
    return os.path.join(svs_index[patient_id][0][1], svs_index[patient_id][0][0])


def sample_tiles_quick(svs_path, tile_size, max_tiles, seed, tissue_thresh=0.3):
    """快速 tile 采样：读 level 2 的缩略图做组织检测。"""
    import openslide
    try:
        slide = openslide.OpenSlide(svs_path)
        w, h = slide.dimensions
        level = min(2, slide.level_count - 1)
        lw, lh = slide.level_dimensions[level]
        scale = slide.level_downsamples[level]
        tile_l2 = max(2, tile_size // int(scale))

        img = slide.read_region((0, 0), level, (lw, lh)).convert('RGB')
        arr = np.asarray(img, dtype=np.float32)
        gray = arr.mean(axis=-1)
        del img, arr

        mask = gray < (0.85 * 255)
        rng = random.Random(seed)

        coords = []
        for y in range(0, lh - tile_l2, tile_l2):
            for x in range(0, lw - tile_l2, tile_l2):
                if mask[y:y + tile_l2, x:x + tile_l2].mean() > tissue_thresh:
                    y_l0 = int(y * scale)
                    x_l0 = int(x * scale)
                    coords.append((y_l0, x_l0))
                    if len(coords) >= max_tiles:
                        break
            if len(coords) >= max_tiles:
                break

        if not coords:
            n = int(np.sqrt(max_tiles))
            for i in range(n):
                for j in range(n):
                    y = int(h * (i + 0.5) / n)
                    x = int(w * (j + 0.5) / n)
                    coords.append((y, x))

        rng.shuffle(coords)
        slide.close()
        return coords[:max_tiles]
    except Exception:
        return []


def load_coords_cache(csv_path, force=False):
    """尝试加载预计算的 tile coords 缓存。"""
    cache_path = os.path.join(os.path.dirname(csv_path), 'tile_coords_cache.pkl')
    if not force and os.path.exists(cache_path):
        print("Loading tile coords cache from:", cache_path)
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


# ============================================================
# Dataset
# ============================================================

class WSIDataset(Dataset):
    """懒加载 WSI 数据集。

    支持从预计算缓存加载 tile coords。
    每次 __getitem__ 读取 tiles。
    """

    def __init__(self, csv_path, wsi_root, tile_size=256, max_tiles=512,
                 tissue_thresh=0.3, seed=42, verbose=True,
                 coords_cache=None):
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        self.wsi_root = wsi_root
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        self.tissue_thresh = tissue_thresh
        self.seed = seed

        self.svs_index = build_svs_index(wsi_root)
        self.df = self.df.copy()
        self.df['patient_id'] = self.df['slide_id'].str[:12]
        self.df['svs_path'] = self.df['patient_id'].map(
            lambda p: find_svs(p, self.svs_index)
        )
        n_before = len(self.df)
        self.df = self.df[self.df['svs_path'].notna()].reset_index(drop=True)
        if verbose:
            print("[WSIDataset] %d/%d samples have WSI files" % (len(self.df), n_before))

        self._coords_cache = coords_cache if coords_cache else {}

    def _get_coords(self, idx):
        svs = self.df.iloc[idx]['svs_path']
        if svs not in self._coords_cache:
            seed = self.seed + idx
            self._coords_cache[svs] = sample_tiles_quick(
                svs, self.tile_size, self.max_tiles, seed, self.tissue_thresh
            )
        return self._coords_cache[svs]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import openslide
        row = self.df.iloc[idx]
        svs = row['svs_path']
        coords = self._get_coords(idx)

        try:
            slide = openslide.OpenSlide(svs)
            tiles = []
            for (y, x) in coords:
                tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')
                tiles.append(np.asarray(tile, dtype=np.uint8))
            slide.close()
            n = len(tiles)
            if n == 0:
                tiles = [np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)]
                n = 1
        except Exception:
            tiles = [np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)]
            n = 1

        tile_stack = np.stack(tiles, axis=0)
        x = torch.from_numpy(tile_stack).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std_t = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        x = (x - mean_t) / std_t

        n_t = x.shape[0]
        coords_norm = torch.linspace(0, 1, n_t).unsqueeze(1).expand(-1, 2).float()

        if 'label' in self.df.columns:
            label = torch.tensor([float(row['label'])], dtype=torch.float32)
        else:
            label = torch.zeros(1)

        return x, coords_norm, label, row['slide_id'], n_t


def collate_wsi(batch):
    labels = torch.stack([b[2] for b in batch])
    max_n = min(max(b[4] for b in batch), BATCH_MAX_TILES)
    n_imgs = batch[0][0].shape[1]
    h, w = batch[0][0].shape[2], batch[0][0].shape[3]

    pad_images = torch.zeros(len(batch), max_n, n_imgs, h, w)
    pad_coords = torch.zeros(len(batch), max_n, 2)
    pad_mask = torch.ones(len(batch), max_n, dtype=torch.bool)

    for i, (imgs, coords, label, sid, n_t) in enumerate(batch):
        n = min(n_t, max_n)
        pad_images[i, :n] = imgs[:n]
        pad_coords[i, :n] = coords[:n]
        pad_mask[i, :n] = False

    return pad_images, pad_coords, labels, [b[3] for b in batch], pad_mask


# ============================================================
# Trainer
# ============================================================

class E2ETrainer:
    def __init__(self, model, optimizer, device, num_tasks=1,
                 early_stopping_patience=8, label_smoothing=0.1, log_interval=5,
                 scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_tasks = num_tasks
        self.patience = early_stopping_patience
        self.label_smoothing = label_smoothing
        self.log_interval = log_interval
        self.scaler = GradScaler('cuda')
        self.best_score = None
        self.counter = 0
        self.scheduler = scheduler

    def train_epoch(self, loader, epoch):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(loader, desc='Epoch %d [Train]' % epoch, mininterval=2.0)
        for batch_idx, batch in enumerate(pbar):
            images, coords, labels, slide_ids, padding_mask = batch
            B, N, C, H, W = images.shape
            images_flat = images.view(B * N, C, H, W).to(self.device)
            coords_dev = coords.to(self.device)
            labels_dev = labels.to(self.device).float()

            self.optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = self.model.forward_wsi_direct(
                    images=images_flat,
                    tile_coords=coords_dev,
                    num_tiles=N,
                    batch_size=B,
                    labels=labels_dev,
                )
            loss = outputs['loss']
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            n_batches += 1
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': '%.4f' % loss.item()})
        return {'loss': total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_labels, all_probs_direct, all_probs_adaptive = [], [], []

        pbar = tqdm(loader, desc='[Val]', mininterval=2.0)
        for batch in pbar:
            images, coords, labels, slide_ids, padding_mask = batch
            B, N, C, H, W = images.shape
            images_flat = images.view(B * N, C, H, W).to(self.device)
            coords_dev = coords.to(self.device)
            labels_dev = labels.to(self.device).float()

            with autocast(device_type='cuda'):
                outputs = self.model.forward_wsi_direct(
                    images=images_flat,
                    tile_coords=coords_dev,
                    num_tiles=N,
                    batch_size=B,
                    labels=labels_dev,
                )
            total_loss += outputs['loss'].item()
            n_batches += 1
            all_labels.append(labels_dev.cpu().numpy())
            all_probs_direct.append(torch.sigmoid(outputs['logits_direct']).cpu().numpy())
            if 'logits_adaptive' in outputs:
                all_probs_adaptive.append(torch.sigmoid(outputs['logits_adaptive']).cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_probs_direct = np.concatenate(all_probs_direct)
        if all_probs_adaptive:
            all_probs_adaptive = np.concatenate(all_probs_adaptive)
            all_probs_ensemble = (all_probs_direct + all_probs_adaptive) / 2.0
        else:
            all_probs_ensemble = all_probs_direct

        try:
            from sklearn.metrics import roc_auc_score
            auc_direct = roc_auc_score(all_labels.ravel(), all_probs_direct.ravel())
            auc_ensemble = roc_auc_score(all_labels.ravel(), all_probs_ensemble.ravel())
            if all_probs_adaptive:
                auc_adaptive = roc_auc_score(all_labels.ravel(), all_probs_adaptive.ravel())
            else:
                auc_adaptive = 0.0
        except:
            auc_direct = auc_adaptive = auc_ensemble = 0.0

        return {
            'loss': total_loss / max(n_batches, 1),
            'auc_direct': auc_direct,
            'auc_adaptive': auc_adaptive,
            'auc_ensemble': auc_ensemble,
        }

    def save_checkpoint(self, epoch, path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
        }, path)


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
                        default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    parser.add_argument('--wsi_root', type=str,
                        default=r'E:\TCGA-WSI-data\TCGA_WSI_BLCA')
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_region_tokens', type=int, default=8)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--stop_epoch', type=int, default=5)
    parser.add_argument('--max_tiles', type=int, default=512)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--results_dir', type=str, default='./results_wsi')
    parser.add_argument('--testing', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    exp_code = 'E2E_WSI_K%d_L%d_T%d_s%d' % (
        args.num_region_tokens, args.num_layers, args.num_tasks, args.seed)
    results_dir = os.path.join(args.results_dir, exp_code)
    os.makedirs(results_dir, exist_ok=True)

    # 尝试加载预计算的 tile coords 缓存
    coords_cache = load_coords_cache(args.csv_path)

    print('\nLoading WSI dataset...')
    dataset = WSIDataset(
        csv_path=args.csv_path,
        wsi_root=args.wsi_root,
        tile_size=TILE_SIZE,
        max_tiles=args.max_tiles,
        tissue_thresh=0.3,
        seed=args.seed,
        verbose=True,
        coords_cache=coords_cache,
    )

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    indices = np.arange(len(dataset))
    all_aucs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if args.testing and fold > 0:
            break
        print('\n' + '=' * 60)
        print('  Fold %d: train=%d, val=%d' % (fold, len(train_idx), len(val_idx)))
        print('=' * 60)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size,
            shuffle=True, collate_fn=collate_wsi,
            num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_wsi,
            num_workers=0, pin_memory=True,
        )

        model = E2EViTCAREFusion(
            tile_size=TILE_SIZE, patch_size=PATCH_SIZE,
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, num_region_tokens=args.num_region_tokens,
            num_tasks=args.num_tasks, use_alibi=True, use_distillation=False,
            use_two_branches=True, dropout=args.dropout,
        ).to(device)
        print_model_summary(model)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01)
        trainer = E2ETrainer(
            model=model, optimizer=optimizer, scheduler=scheduler, device=device,
            num_tasks=args.num_tasks, early_stopping_patience=args.patience,
            label_smoothing=args.label_smoothing,
        )

        best_auc = 0.0
        for epoch in range(args.max_epochs):
            t0 = _time.time()
            train_results = trainer.train_epoch(train_loader, epoch)
            val_results = trainer.validate(val_loader)
            if trainer.scheduler:
                trainer.scheduler.step()
            elapsed = _time.time() - t0

            auc_d = val_results.get('auc_direct', 0.0)
            auc_a = val_results.get('auc_adaptive', 0.0)
            auc_e = val_results.get('auc_ensemble', 0.0)
            auc = auc_e  # use ensemble as main metric
            marker = ' *' if auc > best_auc else ''
            if auc > best_auc:
                best_auc = auc
                trainer.best_score = auc
                trainer.save_checkpoint(epoch, os.path.join(results_dir, 'fold_%d_best.pt' % fold))

            print('Epoch %3d | Loss: %.4f | AUC(D/A/E): %.4f/%.4f/%.4f%s | %.1fs' % (
                epoch, train_results['loss'], auc_d, auc_a, auc_e, marker, elapsed))

            if best_auc > 0 and auc < best_auc:
                trainer.counter += 1
                if trainer.counter >= trainer.patience and epoch > args.stop_epoch:
                    print('  Early stopping at epoch', epoch)
                    break

        trainer.save_checkpoint(epoch, os.path.join(results_dir, 'fold_%d_last.pt' % fold))
        all_aucs.append(best_auc)
        print('Fold %d Best AUC: %.4f' % (fold, best_auc))

    print('\n' + '=' * 60)
    print('Final Results')
    print('=' * 60)
    for i, auc in enumerate(all_aucs):
        print('  Fold %d: AUC=%.4f' % (i, auc))
    print('  Mean AUC: %.4f +/- %.4f' % (np.mean(all_aucs), np.std(all_aucs)))
    print('  Results saved to:', results_dir)


if __name__ == '__main__':
    main()
