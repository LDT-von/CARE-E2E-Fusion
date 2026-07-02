"""
训练脚本: PathwayMorph-OT (Pathway-to-Morphology Optimal Transport)

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

from pathway_morph_ot import PathwayMorphOT, PathwayMorphOTConfig


# ============================================================
# 模拟数据集
# ============================================================

class DummyPMOTDataset(Dataset):
    """模拟数据集：随机生成区域嵌入、拓扑、空间、通路tokens和标签。

    PathwayMorph-OT 工作在区域级别（region-level），不是 tile 级别。
    上游的 tile→region 聚合（如 MoToCARE assignment 或 CARE ARM）已经完成，
    这里直接输入处理后的区域特征。
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_regions: int = 8,
        num_pathways: int = 5,
        atom_dim: int = 256,
        topology_dim: int = 12,
        spatial_dim: int = 4,
        pathway_dim: int = 64,
        num_tasks: int = 2,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_regions = num_regions
        self.num_pathways = num_pathways
        self.atom_dim = atom_dim
        self.topology_dim = topology_dim
        self.spatial_dim = spatial_dim
        self.pathway_dim = pathway_dim
        self.num_tasks = num_tasks
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.samples = []
        for i in range(num_samples):
            region_embeddings = torch.randn(num_regions, atom_dim) * 0.3
            topology = torch.rand(num_regions, topology_dim) * 0.6
            spatial = torch.rand(num_regions, spatial_dim)
            pathway_tokens = torch.randn(num_pathways, pathway_dim) * 0.4
            label = (torch.rand(num_tasks) > 0.5).float()
            slide_id = f"slide_{i:04d}"

            self.samples.append({
                'region_embeddings': region_embeddings,
                'topology': topology,
                'spatial': spatial,
                'pathway_tokens': pathway_tokens,
                'label': label,
                'slide_id': slide_id,
            })

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.samples[idx]


def dummy_collate_fn(batch):
    labels = torch.stack([s['label'] for s in batch])
    region_embeddings = torch.stack([s['region_embeddings'] for s in batch])
    topology = torch.stack([s['topology'] for s in batch])
    spatial = torch.stack([s['spatial'] for s in batch])
    pathway_tokens = torch.stack([s['pathway_tokens'] for s in batch])
    slide_ids = [s['slide_id'] for s in batch]
    return region_embeddings, topology, spatial, pathway_tokens, labels, slide_ids


# ============================================================
# 真实数据集
# ============================================================

class RealPMOTDataset(Dataset):
    """真实数据集：加载 .pt 文件，通过区域池化生成区域级特征。

    由于真实数据是 tile 级别的 .pt 文件，需要先用 K-means/Grid 聚类
    将 tile tokens 转化为 region embeddings，再输入 PathwayMorph-OT。
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        atom_dim: int = 256,
        num_regions: int = 8,
        num_pathways: int = 5,
        topology_dim: int = 12,
        spatial_dim: int = 4,
        pathway_dim: int = 64,
        num_tasks: int = 1,
        max_tiles: int = 2048,
        tile_input_dim: int = 768,
        pca_path: str = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.atom_dim = atom_dim
        self.num_regions = num_regions
        self.num_pathways = num_pathways
        self.topology_dim = topology_dim
        self.spatial_dim = spatial_dim
        self.pathway_dim = pathway_dim
        self.num_tasks = num_tasks
        self.max_tiles = max_tiles
        self.tile_input_dim = tile_input_dim

        # 将 tile 特征投影到 atom_dim 的线性层
        self.tile_proj = nn.Linear(tile_input_dim, atom_dim)

        # 加载或训练 PCA 用于把 tile 特征映射到 2D 伪空间坐标
        self.pca = None
        if pca_path and os.path.exists(pca_path):
            import pickle
            with open(pca_path, 'rb') as f:
                self.pca = pickle.load(f)

    def _get_pseudo_2d_coords(self, features: torch.Tensor) -> torch.Tensor:
        """把 tile 特征投影到 2D 伪空间坐标（用于 grid anchor 区域池化）。
        优先用预训练 PCA；若没有，用 features 的 top-2 SVD 成分作为稳定 2D 坐标。
        """
        x = features.float().cpu().numpy()
        if self.pca is not None:
            coords = self.pca.transform(x)
            return torch.from_numpy(coords).float()
        try:
            xc = x - x.mean(0)
            U, S, Vt = np.linalg.svd(xc, full_matrices=False)
            coords = xc @ Vt[:2].T
            coords = (coords - coords.min(0)) / (coords.max(0) - coords.min(0) + 1e-6)
            return torch.from_numpy(coords).float()
        except Exception:
            return torch.rand(features.shape[0], 2)

    def __len__(self):
        return len(self.df)

    def _pool_tiles_to_regions(self, features: torch.Tensor, coords: torch.Tensor):
        """使用 K-means 风格的网格锚点将 tile 聚类为 region。

        近似：用空间坐标的 grid anchors 将 tokens 分到 num_regions 个区域，
        每个区域做 mean pooling。
        """
        num_tiles = features.shape[0]
        n = self.num_regions

        # grid anchors
        side = int(np.ceil(np.sqrt(n)))
        axis = torch.linspace(0, 1, side)
        gy, gx = torch.meshgrid(axis, axis, indexing='ij')
        anchors = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)[:n]

        # 归一化坐标并确保2D
        if coords.shape[-1] == 1:
            coords = coords.expand(-1, 2)  # [N,1]->[N,2] 两个维度相同值
        coords_norm = normalize_coords_simple(coords)

        # 最近锚点分配
        dist = torch.cdist(coords_norm.float(), anchors.float())
        assignment = dist.argmin(dim=-1)  # [num_tiles]

        region_embeddings = torch.zeros(n, features.shape[1])
        topology = torch.zeros(n, self.topology_dim)
        spatial = torch.zeros(n, self.spatial_dim)

        for k in range(n):
            mask = (assignment == k)
            if mask.sum() > 0:
                region_embeddings[k] = features[mask].mean(0)
                pts = coords_norm[mask]
                topology[k, :2] = pts.mean(0)[:2]
                topology[k, 2:4] = pts.std(0, unbiased=False)[:2]
                spatial[k, :2] = pts.mean(0)[:2]
                spatial[k, 2:] = pts.std(0, unbiased=False)[:2]
            else:
                region_embeddings[k] = torch.zeros(features.shape[1])
                topology[k] = torch.zeros(self.topology_dim)
                spatial[k] = torch.zeros(self.spatial_dim)

        # 投影到 atom_dim
        region_embeddings = self.tile_proj(region_embeddings.float())

        return region_embeddings, topology, spatial

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
            features = torch.randn(200, self.tile_input_dim)

        n = features.shape[0]
        if n > self.max_tiles:
            indices = sorted(random.sample(range(n), self.max_tiles))
            features = features[indices]

        n_trunc = features.shape[0]
        # 用真实 2D PCA 伪空间坐标（替换原来的 linspace 1D 坐标）
        coords = self._get_pseudo_2d_coords(features)

        region_embeddings, topology, spatial = self._pool_tiles_to_regions(features, coords)

        pathway_tokens = torch.randn(self.num_pathways, self.pathway_dim) * 0.1

        if 'label' in self.df.columns:
            label = torch.tensor([float(row['label'])], dtype=torch.float32)
        else:
            label = torch.zeros(self.num_tasks)

        return {
            'region_embeddings': region_embeddings,
            'topology': topology,
            'spatial': spatial,
            'pathway_tokens': pathway_tokens,
            'label': label,
            'slide_id': slide_id,
        }


def normalize_coords_simple(coords: torch.Tensor):
    mins = coords.amin(dim=-2, keepdim=True)
    maxs = coords.amax(dim=-2, keepdim=True)
    return (coords - mins) / (maxs - mins).clamp_min(0.01)


# ============================================================
# 训练器
# ============================================================

class PMOTTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_tasks: int = 2,
        early_stopping_patience: int = 15,
        early_stopping_stop_epoch: int = 20,
        log_interval: int = 10,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_tasks = num_tasks
        self.patience = early_stopping_patience
        self.stop_epoch = early_stopping_stop_epoch
        self.log_interval = log_interval
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
            region_embeddings, topology, spatial, pathway_tokens, labels, slide_ids = batch
            region_embeddings = region_embeddings.to(self.device)
            topology = topology.to(self.device)
            spatial = spatial.to(self.device)
            pathway_tokens = pathway_tokens.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # NOTE: no autocast — Sinkhorn OT is numerically sensitive to fp16
            outputs = self.model(
                region_embeddings=region_embeddings,
                topology=topology,
                spatial=spatial,
                pathway_tokens=pathway_tokens,
                labels=labels,
            )

            loss = outputs['loss']
            loss.backward()
            self.optimizer.step()

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
            region_embeddings, topology, spatial, pathway_tokens, labels, slide_ids = batch
            region_embeddings = region_embeddings.to(self.device)
            topology = topology.to(self.device)
            spatial = spatial.to(self.device)
            pathway_tokens = pathway_tokens.to(self.device)
            labels = labels.to(self.device)

            # NOTE: no autocast — Sinkhorn OT is numerically sensitive to fp16
            outputs = self.model(
                region_embeddings=region_embeddings,
                topology=topology,
                spatial=spatial,
                pathway_tokens=pathway_tokens,
                labels=labels,
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
    trainer = PMOTTrainer(
        model=model, optimizer=optimizer, device=device,
        num_tasks=args.num_tasks,
        early_stopping_patience=args.patience,
        early_stopping_stop_epoch=args.stop_epoch,
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
    parser = argparse.ArgumentParser(description='PathwayMorph-OT Training')

    parser.add_argument('--dataset', type=str, default='dummy', choices=['dummy', 'real'])
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--csv_path', type=str, default=str(ROOT.parent / 'CARE-E2E-Fusion' / 'blca_slides.csv'))
    parser.add_argument('--data_root_dir', type=str, default='E:/TCGA-data/CPathPatchFeature/blca/chief/pt_files')

    parser.add_argument('--atom_dim', type=int, default=256)
    parser.add_argument('--topology_dim', type=int, default=12)
    parser.add_argument('--spatial_dim', type=int, default=4)
    parser.add_argument('--pathway_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.08)
    parser.add_argument('--tau', type=float, default=0.8)
    parser.add_argument('--ot_iters', type=int, default=40)
    parser.add_argument('--ot_cost_weight', type=float, default=0.05)
    parser.add_argument('--entropy_weight', type=float, default=0.005)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=15)
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
        args.exp_code = f"{args.dataset}_PMOT_R{args.atom_dim}_H{args.hidden_dim}_T{args.num_tasks}"
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

    cfg = PathwayMorphOTConfig(
        atom_dim=args.atom_dim,
        topology_dim=args.topology_dim,
        spatial_dim=args.spatial_dim,
        pathway_dim=args.pathway_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=args.num_tasks,
        epsilon=args.epsilon,
        tau=args.tau,
        ot_iters=args.ot_iters,
        ot_cost_weight=args.ot_cost_weight,
        entropy_weight=args.entropy_weight,
        label_smoothing=args.label_smoothing,
    )

    if args.dataset == 'dummy':
        print(f"\nLoading Dummy Dataset ({args.num_samples} samples)...")
        full_dataset = DummyPMOTDataset(
            num_samples=args.num_samples,
            num_regions=8,
            num_pathways=5,
            atom_dim=args.atom_dim,
            topology_dim=args.topology_dim,
            spatial_dim=args.spatial_dim,
            pathway_dim=args.pathway_dim,
            num_tasks=args.num_tasks,
            seed=args.seed,
        )
    else:
        print(f"\nLoading Real Dataset from {args.data_root_dir}...")
        full_dataset = RealPMOTDataset(
            csv_path=args.csv_path,
            data_root=args.data_root_dir,
            atom_dim=args.atom_dim,
            num_regions=8,
            num_pathways=5,
            topology_dim=args.topology_dim,
            spatial_dim=args.spatial_dim,
            pathway_dim=args.pathway_dim,
            num_tasks=args.num_tasks,
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

        fold_model = PathwayMorphOT(cfg).to(device)

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
