"""
训练脚本: MoTo-CARE (Molecularly guided Topological Adaptive Region Encoding)

支持:
  - 模拟数据集测试（直接跑通代码）
  - 真实数据集训练
  - 5折交叉验证
  - 早停机制
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

from moto_care import MoToCARE, MoToCAREConfig
from moto_care.topology import normalize_coords, grid_anchors


# ============================================================
# 模拟数据集
# ============================================================

class DummyMoToCAREDataset(Dataset):
    """模拟 WSI 数据集：生成随机 tile tokens、坐标、拓扑先验、分子tokens 和标签。"""

    def __init__(
        self,
        num_samples: int = 100,
        num_tiles_min: int = 80,
        num_tiles_max: int = 300,
        input_dim: int = 768,
        num_tasks: int = 2,
        num_regions: int = 8,
        topology_dim: int = 12,
        molecule_dim: int = 128,
        num_molecule_tokens: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_tiles_min = num_tiles_min
        self.num_tiles_max = num_tiles_max
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.num_regions = num_regions
        self.topology_dim = topology_dim
        self.molecule_dim = molecule_dim
        self.num_molecule_tokens = num_molecule_tokens
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.samples = []
        for i in range(num_samples):
            n_tiles = random.randint(num_tiles_min, num_tiles_max)

            features = torch.randn(n_tiles, input_dim) * 0.5
            coords = torch.rand(n_tiles, 2)

            # label: 随机0/1
            label = (torch.rand(num_tasks) > 0.5).float()

            # topology_prior: 每个区域的拓扑先验（用坐标聚类模拟）
            anchors = grid_anchors(num_regions, torch.device('cpu'), torch.float32)
            coords_norm = (coords - coords.amin(0)) / (coords.amax(0) - coords.amin(0)).clamp_min(0.01)
            nearest = torch.cdist(coords_norm, anchors).argmin(dim=-1)
            topology_prior = torch.zeros(num_regions, topology_dim)
            for k in range(num_regions):
                mask = nearest == k
                if mask.any():
                    pts = coords_norm[mask]
                    mn = pts.mean(0)
                    std = pts.std(0, unbiased=False)
                    topology_prior[k, :4] = torch.cat([mn, std])[:4]
                else:
                    topology_prior[k, :] = torch.randn(topology_dim) * 0.1
            topology_target = topology_prior.clone()

            # molecule_tokens: 模拟RNA/蛋白表达
            molecule_tokens = torch.randn(num_molecule_tokens, molecule_dim) * 0.5

            slide_id = f"slide_{i:04d}"

            self.samples.append({
                'features': features,
                'coords': coords,
                'label': label,
                'topology_prior': topology_prior,
                'topology_target': topology_target,
                'molecule_tokens': molecule_tokens,
                'slide_id': slide_id,
                'num_tiles': n_tiles,
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]


def dummy_collate_fn(batch):
    """Padding collate for variable-length tile sequences. 截断到 BATCH_MAX_TILES。"""
    BATCH_MAX_TILES = 4096
    labels = torch.stack([s['label'] for s in batch])
    num_tiles_list = [min(s['num_tiles'], BATCH_MAX_TILES) for s in batch]
    max_tiles = max(num_tiles_list)

    pad_features = torch.zeros(len(batch), max_tiles, batch[0]['features'].shape[1])
    pad_coords = torch.zeros(len(batch), max_tiles, 2)
    padding_mask = torch.zeros(len(batch), max_tiles, dtype=torch.bool)

    for i, s in enumerate(batch):
        n = s['num_tiles']
        pad_features[i, :n] = s['features']
        pad_coords[i, :n] = s['coords']
        padding_mask[i, n:] = True

    topology_prior = torch.stack([s['topology_prior'] for s in batch])
    topology_target = torch.stack([s['topology_target'] for s in batch])
    molecule_tokens = torch.stack([s['molecule_tokens'] for s in batch])
    slide_ids = [s['slide_id'] for s in batch]

    return pad_features, pad_coords, labels, topology_prior, topology_target, molecule_tokens, slide_ids, padding_mask


# ============================================================
# 真实数据集
# ============================================================

class RealMoToCAREDataset(Dataset):
    """真实 WSI 数据集，加载 .pt 文件。

    CSV 必须包含: slide_id, pt_path, label
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        input_dim: int = 768,
        num_tasks: int = 1,
        max_tiles: int = 4096,
        num_regions: int = 8,
        topology_dim: int = 12,
        molecule_dim: int = 128,
        num_molecule_tokens: int = 4,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.max_tiles = max_tiles
        self.num_regions = num_regions
        self.topology_dim = topology_dim
        self.molecule_dim = molecule_dim
        self.num_molecule_tokens = num_molecule_tokens

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        slide_id = str(row['slide_id'])

        if 'pt_path' in row and pd.notna(row['pt_path']):
            pt_path = str(row['pt_path']).replace('\\', '/')
        else:
            pt_path = None

        if pt_path and os.path.exists(pt_path):
            data = torch.load(pt_path, map_location='cpu')
            features = data.float() if data.dtype != torch.float32 else data
        else:
            features = torch.randn(200, self.input_dim)

        n = features.shape[0]
        if n > self.max_tiles:
            indices = sorted(random.sample(range(n), self.max_tiles))
            features = features[indices]

        n_trunc = features.shape[0]
        coords = torch.linspace(0, 1, n_trunc).unsqueeze(1).expand(-1, 2).float()

        if 'label' in self.df.columns:
            label = torch.tensor([float(row['label'])], dtype=torch.float32)
        else:
            label = torch.zeros(self.num_tasks)

        # 合成拓扑先验
        anchors = grid_anchors(self.num_regions, torch.device('cpu'), torch.float32)
        nearest = torch.cdist(coords, anchors).argmin(dim=-1)
        topology_prior = torch.zeros(self.num_regions, self.topology_dim)
        for k in range(self.num_regions):
            mask = nearest == k
            if mask.any():
                pts = coords[mask]
                mn = pts.mean(0)
                std = pts.std(0, unbiased=False)
                topology_prior[k, :4] = torch.cat([mn, std])[:4]
            else:
                topology_prior[k, :] = torch.randn(self.topology_dim) * 0.05
        topology_target = topology_prior.clone()

        molecule_tokens = torch.randn(self.num_molecule_tokens, self.molecule_dim) * 0.1

        return {
            'features': features,
            'coords': coords,
            'label': label,
            'topology_prior': topology_prior,
            'topology_target': topology_target,
            'molecule_tokens': molecule_tokens,
            'slide_id': slide_id,
            'num_tiles': features.shape[0],
        }


# ============================================================
# 训练器
# ============================================================

def smoothed_bce(logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    if smoothing <= 0:
        return F.binary_cross_entropy_with_logits(logits, labels)
    smooth_labels = labels * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, smooth_labels)


class MoToCARETrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_tasks: int = 2,
        early_stopping_patience: int = 15,
        early_stopping_stop_epoch: int = 20,
        log_interval: int = 10,
        label_smoothing: float = 0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_tasks = num_tasks
        self.patience = early_stopping_patience
        self.stop_epoch = early_stopping_stop_epoch
        self.log_interval = log_interval
        self.label_smoothing = label_smoothing
        self.scaler = GradScaler('cuda')
        self.best_score = None
        self.counter = 0

    def train_epoch(self, loader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        loss_accum = {}
        n_batches = 0

        pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            features, coords, labels, topo_prior, topo_target, mol_tokens, slide_ids, padding_mask = batch
            features = features.to(self.device)
            coords = coords.to(self.device)
            labels = labels.to(self.device)
            topo_prior = topo_prior.to(self.device)
            topo_target = topo_target.to(self.device)
            mol_tokens = mol_tokens.to(self.device)
            padding_mask = padding_mask.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = self.model(
                    features=features,
                    coords=coords,
                    labels=labels,
                    topology_prior=topo_prior,
                    topology_target=topo_target,
                    molecule_tokens=mol_tokens,
                    padding_mask=padding_mask,
                )

            loss = outputs['loss']
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            for k, v in outputs.get('losses', {}).items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v.item()
            n_batches += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / n_batches
        for k in loss_accum:
            loss_accum[k] /= n_batches
        return {'loss': avg_loss, **loss_accum}

    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        loss_accum = {}
        all_labels = []
        all_probs = []

        pbar = tqdm(loader, desc='[Val]')
        for batch in pbar:
            features, coords, labels, topo_prior, topo_target, mol_tokens, slide_ids, padding_mask = batch
            features = features.to(self.device)
            coords = coords.to(self.device)
            labels = labels.to(self.device)
            topo_prior = topo_prior.to(self.device)
            topo_target = topo_target.to(self.device)
            mol_tokens = mol_tokens.to(self.device)
            padding_mask = padding_mask.to(self.device)

            with autocast(device_type='cuda'):
                outputs = self.model(
                    features=features,
                    coords=coords,
                    labels=labels,
                    topology_prior=topo_prior,
                    topology_target=topo_target,
                    molecule_tokens=mol_tokens,
                    padding_mask=padding_mask,
                )

            total_loss += outputs['loss'].item()
            n_batches += 1
            for k, v in outputs.get('losses', {}).items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v.item()

            all_labels.append(labels.cpu().numpy())
            all_probs.append(outputs['probs'].cpu().numpy())

        avg_loss = total_loss / n_batches
        for k in loss_accum:
            loss_accum[k] /= n_batches

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)
        try:
            auc = compute_multitask_auc(all_labels, all_probs)
        except Exception:
            auc = 0.0

        results = {'loss': avg_loss, 'auc': auc}
        results.update({k: v for k, v in loss_accum.items()})
        return results

    def save_checkpoint(self, epoch: int, path: str):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.best_score = ckpt.get('best_score')
        return ckpt.get('epoch', 0)


def compute_multitask_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        if labels.shape[1] == 1:
            return roc_auc_score(labels.ravel(), probs.ravel())
        aucs = []
        for i in range(labels.shape[1]):
            if labels[:, i].sum() > 0 and labels[:, i].sum() < len(labels[:, i]):
                aucs.append(roc_auc_score(labels[:, i], probs[:, i]))
        return np.mean(aucs) if aucs else 0.0
    except Exception:
        return 0.0


# ============================================================
# 训练入口
# ============================================================

def seed_everything(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fold(fold, model, train_loader, val_loader, device, args, results_dir):
    print(f"\n{'='*60}")
    print(f"  Fold {fold}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    trainer = MoToCARETrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_tasks=args.num_tasks,
        early_stopping_patience=args.patience,
        early_stopping_stop_epoch=args.stop_epoch,
        label_smoothing=args.label_smoothing,
    )

    best_auc = 0.0
    best_results = {}

    for epoch in range(args.max_epochs):
        t0 = time.time()
        train_results = trainer.train_epoch(train_loader, epoch)
        train_time = time.time() - t0
        val_results = trainer.validate(val_loader)
        val_time = time.time() - t0 - train_time

        auc = val_results.get('auc', 0.0)
        marker = " *" if auc > best_auc else ""

        print(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_results['loss']:.4f} ({train_time:.1f}s) | "
            f"Val Loss: {val_results['loss']:.4f} ({val_time:.1f}s) | "
            f"AUC: {auc:.4f}{marker}"
        )

        if auc > best_auc:
            best_auc = auc
            best_results = val_results.copy()
            trainer.save_checkpoint(epoch, os.path.join(results_dir, f'fold_{fold}_best.pt'))
            print(f"  -> Best AUC: {best_auc:.4f}")
        elif epoch == 0:
            best_results = val_results.copy()

        score = -val_results['loss']
        if trainer.best_score is None:
            trainer.best_score = score
            trainer.counter = 0
        elif score < trainer.best_score:
            trainer.counter += 1
            print(f"  EarlyStopping: {trainer.counter}/{trainer.patience}")
            if trainer.counter >= trainer.patience and epoch > trainer.stop_epoch:
                print(f"  Early stopping at epoch {epoch}")
                break
        else:
            trainer.best_score = score
            trainer.counter = 0

    trainer.save_checkpoint(epoch, os.path.join(results_dir, f'fold_{fold}_last.pt'))
    print(f"\nFold {fold} Best AUC: {best_auc:.4f}")
    return best_results


def get_args():
    parser = argparse.ArgumentParser(description='MoTo-CARE Training')

    parser.add_argument('--dataset', type=str, default='dummy', choices=['dummy', 'real'])
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--csv_path', type=str, default='C:/Users/cwnu/Desktop/CARE-E2E-Fusion/CARE-E2E-Fusion/blca_slides.csv')
    parser.add_argument('--data_root_dir', type=str, default='E:/TCGA-data/CPathPatchFeature/blca/chief/pt_files')

    parser.add_argument('--input_dim', type=int, default=768)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--num_regions', type=int, default=8)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_tasks', type=int, default=2)
    parser.add_argument('--topology_dim', type=int, default=12)
    parser.add_argument('--molecule_dim', type=int, default=128)
    parser.add_argument('--top_k_regions', type=int, default=4)
    parser.add_argument('--assignment_temperature', type=float, default=0.35)
    parser.add_argument('--topology_weight', type=float, default=0.5)
    parser.add_argument('--molecular_weight', type=float, default=0.2)
    parser.add_argument('--entropy_weight', type=float, default=0.01)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--stop_epoch', type=int, default=5)
    parser.add_argument('--label_frac', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--exp_code', type=str, default=None)
    parser.add_argument('--testing', action='store_true')

    args = parser.parse_args()

    if args.exp_code is None:
        args.exp_code = f"{args.dataset}_MoToCARE_R{args.num_regions}_T{args.num_tasks}"
    return args


def main(args):
    seed_everything(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    args.results_dir = os.path.join(args.results_dir, args.exp_code + f'_s{args.seed}')
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = MoToCAREConfig(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_regions=args.num_regions,
        num_heads=args.num_heads,
        num_tasks=args.num_tasks,
        topology_dim=args.topology_dim,
        molecule_dim=args.molecule_dim,
        top_k_regions=args.top_k_regions,
        assignment_temperature=args.assignment_temperature,
        topology_weight=args.topology_weight,
        molecular_weight=args.molecular_weight,
        entropy_weight=args.entropy_weight,
        dropout=args.dropout,
        label_smoothing=args.label_smoothing,
    )

    if args.dataset == 'dummy':
        print(f"\nLoading Dummy Dataset ({args.num_samples} samples)...")
        full_dataset = DummyMoToCAREDataset(
            num_samples=args.num_samples,
            num_tiles_min=80,
            num_tiles_max=300,
            input_dim=args.input_dim,
            num_tasks=args.num_tasks,
            num_regions=args.num_regions,
            topology_dim=args.topology_dim,
            molecule_dim=args.molecule_dim,
            seed=args.seed,
        )
    else:
        print(f"\nLoading Real Dataset from {args.data_root_dir}...")
        full_dataset = RealMoToCAREDataset(
            csv_path=args.csv_path,
            data_root=args.data_root_dir,
            input_dim=args.input_dim,
            num_tasks=args.num_tasks,
            num_regions=args.num_regions,
            topology_dim=args.topology_dim,
            molecule_dim=args.molecule_dim,
        )

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    indices = np.arange(len(full_dataset))

    all_fold_aucs = []
    all_fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if args.testing and fold > 0:
            break

        print(f"\n{'='*60}")
        print(f"  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
        print(f"{'='*60}")

        train_subset = Subset(full_dataset, train_idx.tolist())
        val_subset = Subset(full_dataset, val_idx.tolist())

        train_loader = DataLoader(
            train_subset, batch_size=args.batch_size,
            shuffle=True, collate_fn=dummy_collate_fn,
            num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=args.batch_size,
            shuffle=False, collate_fn=dummy_collate_fn,
            num_workers=0, pin_memory=True,
        )

        fold_model = MoToCARE(cfg).to(device)

        results = train_fold(
            fold=fold, model=fold_model,
            train_loader=train_loader, val_loader=val_loader,
            device=device, args=args, results_dir=args.results_dir,
        )

        auc = results.get('auc', 0.0)
        all_fold_aucs.append(auc)
        all_fold_results.append(results)

    print(f"\n{'='*60}")
    print(f"  Final Results ({args.k} folds)")
    print(f"{'='*60}")
    for fold, (auc, res) in enumerate(zip(all_fold_aucs, all_fold_results)):
        print(f"  Fold {fold}: AUC={auc:.4f}, Loss={res['loss']:.4f}")

    mean_auc = np.mean(all_fold_aucs) if all_fold_aucs else 0.0
    std_auc = np.std(all_fold_aucs) if len(all_fold_aucs) > 1 else 0.0
    print(f"\n  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Results saved to: {args.results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    args = get_args()
    main(args)
