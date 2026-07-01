# -*- coding: utf-8 -*-
"""
E2E WSI Training with small CNN backbone (no internet needed)
==============================================================
Strategy:
1. Small ResNet-18 style CNN (random init, trainable) for tile feature extraction
2. Train DRP + ARM + heads on top
3. Much faster and more stable than full ViT
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

from models.fusion_model import DynamicRegionPartition, AdaptiveRegionModeling, MultiTaskHead

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TILE_SIZE = 224
BATCH_MAX_TILES = 512


# ============================================================
# Small CNN backbone (no internet, no download)
# ============================================================

class SmallCNN(nn.Module):
    """Lightweight CNN: 2 conv blocks + GAP -> 256-dim features."""
    def __init__(self, out_dim=256):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(192, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        # x: [N, 3, 224, 224]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x  # [N, out_dim]


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
                    coords.append((int(y * scale), int(x * scale)))
                    if len(coords) >= max_tiles:
                        break
            if len(coords) >= max_tiles:
                break
        if not coords:
            n = int(np.sqrt(max_tiles))
            for i in range(n):
                for j in range(n):
                    coords.append((int(h * (i + 0.5) / n), int(w * (j + 0.5) / n)))
        rng.shuffle(coords)
        slide.close()
        return coords[:max_tiles]
    except:
        return []


def load_coords_cache(csv_path):
    cache_path = os.path.join(os.path.dirname(csv_path), 'tile_coords_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return None


# ============================================================
# Dataset
# ============================================================

class WSIDataset(Dataset):
    def __init__(self, csv_path, wsi_root, tile_size=224, max_tiles=256,
                 tissue_thresh=0.3, seed=42, coords_cache=None, verbose=True):
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
            self._coords_cache[svs] = sample_tiles_quick(
                svs, self.tile_size, self.max_tiles, self.seed + idx, self.tissue_thresh
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
        except:
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
# Fusion module: DRP + ARM + heads
# ============================================================

class FusionModule(nn.Module):
    def __init__(self, embed_dim=384, num_heads=8, num_layers=2,
                 num_region_tokens=8, num_tasks=1, dropout=0.2, use_two_branches=True):
        super().__init__()
        self.use_two_branches = use_two_branches

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout, batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        ])

        self.head_direct = nn.Linear(embed_dim, num_tasks)

        self.drp = DynamicRegionPartition(
            embed_dim, num_region_tokens, num_heads, dropout
        )
        self.arm = AdaptiveRegionModeling(embed_dim, num_heads, dropout)
        self.head_adaptive = MultiTaskHead(
            embed_dim, num_tasks, task_names=None, dropout=dropout, use_threshold=True
        )

    def forward(self, tile_tokens, coords, labels=None, label_smoothing=0.1):
        B, N, C = tile_tokens.shape

        x = tile_tokens
        for block in self.blocks:
            x = block(x)

        x_global = x.mean(dim=1)
        logits_direct = self.head_direct(x_global)
        total_loss = 0.0
        loss_dict = {}

        if labels is not None:
            smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            loss_direct = F.binary_cross_entropy_with_logits(logits_direct, smooth)
            total_loss = total_loss + loss_direct
            loss_dict['loss_direct'] = loss_direct.item()

        outputs = {'logits_direct': logits_direct, 'transformer_output': x}

        if self.use_two_branches:
            region_features, attn_weights, coverage = self.drp(
                tile_tokens=x, coords=coords, return_coverage=True,
            )
            region_embeddings, region_pooled = self.arm(
                region_features=region_features,
                tile_tokens=x,
                attn_weights=attn_weights,
            )
            adaptive_out = self.head_adaptive(region_pooled)
            logits_adaptive = adaptive_out['logits']

            outputs['logits_adaptive'] = logits_adaptive
            outputs['adaptive_probs'] = adaptive_out['probs']

            if labels is not None:
                loss_adaptive = F.binary_cross_entropy_with_logits(logits_adaptive, smooth)
                total_loss = total_loss + loss_adaptive
                loss_dict['loss_adaptive'] = loss_adaptive.item()
                loss_fusion = F.mse_loss(
                    torch.sigmoid(logits_direct),
                    torch.sigmoid(logits_adaptive),
                )
                total_loss = total_loss + 0.1 * loss_fusion
                loss_dict['loss_fusion'] = loss_fusion.item()

        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict
        return outputs


# ============================================================
# Training
# ============================================================

def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
        default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    parser.add_argument('--wsi_root', type=str,
        default=r'E:\TCGA-WSI-data\TCGA_WSI_BLCA')
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_region_tokens', type=int, default=8)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--stop_epoch', type=int, default=5)
    parser.add_argument('--max_tiles', type=int, default=64)
    parser.add_argument('--max_tile_batch', type=int, default=16)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--results_dir', type=str, default='./results_wsi_cnn')
    parser.add_argument('--testing', action='store_true')
    args = parser.parse_args()

    seed_everything(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    exp_code = 'E2E_WSI_CNN_K%d_L%d_s%d' % (
        args.num_region_tokens, args.num_layers, args.seed)
    results_dir = os.path.join(args.results_dir, exp_code)
    os.makedirs(results_dir, exist_ok=True)

    coords_cache = load_coords_cache(args.csv_path)

    print('\nLoading WSI dataset...')
    dataset = WSIDataset(
        csv_path=args.csv_path, wsi_root=args.wsi_root,
        tile_size=TILE_SIZE, max_tiles=args.max_tiles,
        tissue_thresh=0.3, seed=args.seed,
        coords_cache=coords_cache, verbose=True,
    )

    print('\nBuilding small CNN backbone (random init, trainable)...')
    backbone = SmallCNN(out_dim=args.embed_dim).to(device)
    embed_dim = args.embed_dim
    bb_params = sum(p.numel() for p in backbone.parameters())
    print('Backbone params: %d' % bb_params)

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
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
            shuffle=True, collate_fn=collate_wsi, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
            shuffle=False, collate_fn=collate_wsi, num_workers=0, pin_memory=True)

        # Fresh model per fold
        model = FusionModule(
            embed_dim=embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
            num_region_tokens=args.num_region_tokens, num_tasks=args.num_tasks,
            dropout=args.dropout, use_two_branches=True,
        ).to(device)
        backbone = SmallCNN(out_dim=args.embed_dim).to(device)  # reset per fold
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        bb_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        print('Trainable params: backbone=%d, model=%d' % (bb_params, trainable_params))

        # AdamW with lower LR for backbone
        optimizer = torch.optim.AdamW([
            {'params': backbone.parameters(), 'lr': args.lr * 0.1},  # lower LR for backbone
            {'params': model.parameters(), 'lr': args.lr},
        ], weight_decay=args.reg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01
        )
        scaler = GradScaler('cuda')
        best_auc = 0.0
        counter = 0
        epoch = 0

        for epoch in range(args.max_epochs):
            t0 = _time.time()

            # Train
            model.train()
            backbone.train()
            total_loss = 0.0
            nb = 0
            pbar = tqdm(train_loader, desc='Epoch %d [Train]' % epoch, mininterval=2.0)
            for batch in pbar:
                images, coords, labels, slide_ids, padding_mask = batch
                B, N, C, H, W = images.shape
                images_flat = images.view(B * N, C, H, W).to(device)
                coords_dev = coords.to(device)
                labels_dev = labels.to(device).float()

                # Process tiles in mini-batches to avoid OOM
                tile_batches = []
                tb = args.max_tile_batch
                for i in range(0, B * N, tb):
                    tile_batches.append(images_flat[i:i + tb])
                tile_token_chunks = []
                for tb_imgs in tile_batches:
                    with autocast(device_type='cuda'):
                        tile_token_chunks.append(backbone(tb_imgs))
                tile_tokens = torch.cat(tile_token_chunks, dim=0)
                tile_tokens = tile_tokens.view(B, N, -1)

                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    outputs = model(tile_tokens, coords_dev, labels_dev,
                                   label_smoothing=args.label_smoothing)

                loss = outputs['loss']
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(backbone.parameters()) + list(model.parameters()),
                    max_norm=1.0
                )
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                nb += 1
                pbar.set_postfix({'loss': '%.4f' % loss.item()})

            train_loss = total_loss / max(nb, 1)

            # Validate
            model.eval()
            backbone.eval()
            val_loss = 0.0
            nb_val = 0
            all_labels, all_probs_direct, all_probs_adaptive = [], [], []

            pbar2 = tqdm(val_loader, desc='[Val]', mininterval=2.0)
            with torch.no_grad():
                for batch in pbar2:
                    images, coords, labels, slide_ids, padding_mask = batch
                    B, N, C, H, W = images.shape
                    images_flat = images.view(B * N, C, H, W).to(device)
                    coords_dev = coords.to(device)
                    labels_dev = labels.to(device).float()
                    with autocast(device_type='cuda'):
                        # Tile batching to avoid OOM
                        tile_batches = []
                        tb = args.max_tile_batch
                        for i in range(0, B * N, tb):
                            tile_batches.append(images_flat[i:i + tb])
                        tile_token_chunks = []
                        for tb_imgs in tile_batches:
                            tile_token_chunks.append(backbone(tb_imgs))
                        tile_tokens = torch.cat(tile_token_chunks, dim=0)
                        tile_tokens = tile_tokens.view(B, N, -1)
                        outputs = model(tile_tokens, coords_dev, labels_dev,
                                       label_smoothing=args.label_smoothing)
                    val_loss += outputs['loss'].item()
                    nb_val += 1
                    all_labels.append(labels_dev.cpu().numpy())
                    all_probs_direct.append(
                        torch.sigmoid(outputs['logits_direct']).cpu().numpy()
                    )
                    if 'logits_adaptive' in outputs:
                        all_probs_adaptive.append(
                            torch.sigmoid(outputs['logits_adaptive']).cpu().numpy()
                        )

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
                auc_adaptive = roc_auc_score(all_labels.ravel(), all_probs_adaptive.ravel()) \
                    if all_probs_adaptive else 0.0
            except:
                auc_direct = auc_adaptive = auc_ensemble = 0.0

            auc = auc_ensemble
            elapsed = _time.time() - t0
            marker = ' *' if auc > best_auc else ''
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'epoch': epoch,
                    'backbone_state_dict': backbone.state_dict(),
                    'model_state_dict': model.state_dict(),
                    'best_score': best_auc,
                }, os.path.join(results_dir, 'fold_%d_best.pt' % fold))
            if best_auc > 0 and auc < best_auc:
                counter += 1

            print('Epoch %3d | Loss: %.4f/%.4f | AUC(D/A/E): %.4f/%.4f/%.4f%s | %.1fs' % (
                epoch, train_loss, val_loss / max(nb_val, 1),
                auc_direct, auc_adaptive, auc_ensemble, marker, elapsed))

            if counter >= args.patience and epoch > args.stop_epoch:
                print('  Early stopping at epoch', epoch)
                break

            scheduler.step()

        torch.save({
            'epoch': epoch,
            'backbone_state_dict': backbone.state_dict(),
            'model_state_dict': model.state_dict(),
            'best_score': best_auc,
        }, os.path.join(results_dir, 'fold_%d_last.pt' % fold))
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
