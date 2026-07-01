"""
训练脚本: E2E-ViT + CARE 融合模型

支持:
  - 模拟数据集测试（直接跑通代码）
  - 真实数据集训练
  - 5折交叉验证
  - 早停机制
"""

from __future__ import annotations

import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Dict, List, Optional, Tuple
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import Optional, List, Tuple, Dict
import random

from care_adapter import find_care_feature_file, load_care_feature


# ============================================================
# 模拟数据集（用于快速验证代码能跑通）
# ============================================================

class DummyWSIDataset(Dataset):
    """模拟 WSI 数据集：生成随机 tile tokens 和标签用于测试。

    模拟 E2E-ViT 的长条图处理：
    - 每个 WSI 有 N 个 tile（随机 100-500 个）
    - 每个 tile 的特征维度是 embed_dim
    - 多任务标签（ER/PR/HER2/LDH）
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_tiles_min: int = 100,
        num_tiles_max: int = 500,
        embed_dim: int = 768,
        num_tasks: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.num_tiles_min = num_tiles_min
        self.num_tiles_max = num_tiles_max
        self.embed_dim = embed_dim
        self.num_tasks = num_tasks
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # 预生成所有数据
        self.samples = []
        for i in range(num_samples):
            n_tiles = random.randint(num_tiles_min, num_tiles_max)
            # tile tokens: [n_tiles, embed_dim]
            tile_tokens = torch.randn(n_tiles, embed_dim)
            # 归一化坐标: [n_tiles, 2]
            coords = torch.rand(n_tiles, 2)
            # 多任务标签（0-1 之间的概率）
            label = torch.rand(num_tasks)
            # slide_id
            slide_id = f"slide_{i:04d}"
            self.samples.append({
                'tile_tokens': tile_tokens,
                'coords': coords,
                'label': label,
                'slide_id': slide_id,
                'num_tiles': n_tiles,
            })

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def dummy_collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[int]]:
    """整理 batch 数据"""
    labels = torch.stack([s['label'] for s in batch])  # [B, num_tasks]
    num_tiles_list = [s['num_tiles'] for s in batch]

    # 使用样本中最小的 tile 数量作为批次大小
    # 或者 padding 到相同长度
    min_tiles = min(num_tiles_list)

    # 所有样本都截断/填充到相同长度
    max_tiles = max(num_tiles_list)
    pad_tokens = torch.zeros(len(batch), max_tiles, batch[0]['tile_tokens'].shape[1])
    pad_coords = torch.zeros(len(batch), max_tiles, 2)
    padding_mask = torch.zeros(len(batch), max_tiles, dtype=torch.bool)

    for i, s in enumerate(batch):
        n = s['num_tiles']
        pad_tokens[i, :n] = s['tile_tokens']
        pad_coords[i, :n] = s['coords']
        padding_mask[i, n:] = True

    slide_ids = [s['slide_id'] for s in batch]

    return pad_tokens, pad_coords, labels, slide_ids, padding_mask


# ============================================================
# 真实数据集（需要替换为你的实际数据加载逻辑）
# ============================================================

class RealWSIDataset(Dataset):
    """真实 WSI 数据集，支持 .npy（CARE格式）和 .pt（CONCH格式）。

    CSV 必须包含:
      - slide_id: 幻灯片/患者ID
      - pt_path: .pt 或 .npy 文件的完整路径（或 data_root 下的相对路径）

    .pt 格式: 每文件是 [N, 768] tensor（CONCH 提取的 patch 特征），无坐标信息。
    .npy 格式: 每文件是 dict，含 feature 和 coords/index 键。
    """

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        embed_dim: int = 768,
        num_tasks: int = 4,
        tile_size: int = 256,
        max_tiles: int = 4096,
    ):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.embed_dim = embed_dim
        self.num_tasks = num_tasks
        self.tile_size = tile_size
        self.max_tiles = max_tiles

    def __len__(self) -> int:
        return len(self.df)

    def _load_pt_features(self, pt_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """加载 .pt 文件（CONCH patch features）。

        .pt 文件: [N, 768] tensor，无坐标信息。
        坐标用 1D 位置编码模拟: [0/N, 1/N, ..., (N-1)/N]。
        """
        data = torch.load(pt_path, map_location='cpu')
        tokens = data.float() if data.dtype != torch.float32 else data

        # 生成模拟坐标（1D 位置）
        n = tokens.shape[0]
        coords = torch.linspace(0, 1, n).unsqueeze(1).float()
        return tokens, coords

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        slide_id = row['slide_id']

        # 支持两种路径写法：
        # 1. CSV 中有 pt_path 列（完整路径）
        # 2. CSV 中只有 slide_id，需要在 data_root 下查找
        if 'pt_path' in row and pd.notna(row['pt_path']):
            pt_path = str(row['pt_path'])
            # 转换反斜杠为正斜杠（Windows 兼容）
            pt_path = pt_path.replace('\\', '/')
        else:
            pt_path = find_care_feature_file(
                data_root=self.data_root,
                slide_id=str(slide_id),
                tile_size=self.tile_size,
            )

        # 根据后缀判断加载方式
        if pt_path.endswith('.pt'):
            tile_tokens, coords = self._load_pt_features(pt_path)
        else:
            tile_tokens, coords = load_care_feature(pt_path)

        # 限制最大 tile 数
        if tile_tokens.shape[0] > self.max_tiles:
            indices = sorted(random.sample(range(tile_tokens.shape[0]), self.max_tiles))
            tile_tokens = tile_tokens[indices]
            coords = coords[indices]

        # 标签
        label_cols = [c for c in self.df.columns if c.startswith('label_')]
        if label_cols:
            label = torch.tensor([row[c] for c in label_cols], dtype=torch.float32)
        elif 'label' in self.df.columns:
            label = torch.tensor([row['label']], dtype=torch.float32)
        else:
            label = torch.zeros(self.num_tasks)

        return {
            'tile_tokens': tile_tokens,
            'coords': coords,
            'label': label,
            'slide_id': slide_id,
            'num_tiles': tile_tokens.shape[0],
        }


# ============================================================
# 训练器
# ============================================================

def smoothed_bce(logits: torch.Tensor, labels: torch.Tensor, smoothing: float = 0.1) -> torch.Tensor:
    """带 label smoothing 的 BCE loss。"""
    if smoothing <= 0:
        return F.binary_cross_entropy_with_logits(logits, labels)
    smooth_labels = labels * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, smooth_labels)


class FusionTrainer:
    """融合模型训练器"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_tasks: int = 4,
        task_names: Optional[List[str]] = None,
        early_stopping_patience: int = 15,
        early_stopping_stop_epoch: int = 20,
        log_interval: int = 10,
        label_smoothing: float = 0.1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]
        self.patience = early_stopping_patience
        self.stop_epoch = early_stopping_stop_epoch
        self.log_interval = log_interval
        self.label_smoothing = label_smoothing
        self.scaler = GradScaler('cuda')
        self.best_score = None
        self.counter = 0

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        loss_accum = {}
        n_batches = 0

        pbar = tqdm(loader, desc=f'Epoch {epoch} [Train]')
        for batch_idx, batch in enumerate(pbar):
            pad_tokens, pad_coords, labels, slide_ids, padding_mask = batch
            pad_tokens = pad_tokens.to(self.device)
            pad_coords = pad_coords.to(self.device)
            labels = labels.to(self.device).float()
            padding_mask = padding_mask.to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type='cuda'):
                # 构造模拟的长条图（因为 Dummy 数据集是 tile tokens，需要转换）
                # 如果是 DummyWSIDataset，用 tile tokens 直接构建 strip_image
                # 这里做了一个简化：把 tile_tokens 当作 strip_image 的特征
                # 真实场景下应该用真实的图像数据
                outputs = self._forward_with_tiles(
                    pad_tokens, pad_coords, labels, padding_mask
                )

            loss = outputs['loss']
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            for k, v in outputs.get('loss_dict', {}).items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v
            n_batches += 1

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / n_batches
        for k in loss_accum:
            loss_accum[k] /= n_batches

        return {'loss': avg_loss, **loss_accum}

    def _forward_with_tiles(
        self,
        tile_tokens: torch.Tensor,
        coords: torch.Tensor,
        labels: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """用 tile tokens 构建 strip_image 并前向传播。

        由于我们用的是预提取的 tile 特征，需要模拟 E2E-ViT 的处理流程。
        这里做了一个简化：把 tile tokens 当作已经过 Transformer 处理的结果，
        直接用于动态区域划分和 ARM。

        真实场景: strip_image 应该是 [B, 3, H, W] 的图像
        """
        B, N, C = tile_tokens.shape

        # 模拟：把 tile_tokens 当作已经过 patch_embed + patch_merger + transformer 的输出
        # 直接用于后续的双层自适应分支
        x = tile_tokens

        # ---- 分支1: 直接分类 ----
        x_global = x.mean(dim=1)  # [B, C]
        logits_direct = self.model.head_direct(x_global)  # [B, num_tasks]

        outputs = {
            'tile_tokens': tile_tokens,
            'transformer_output': x,
            'logits_direct': logits_direct,
        }

        total_loss = 0.0
        loss_dict = {}

        if labels is not None:
            loss_direct = smoothed_bce(logits_direct, labels, self.label_smoothing)
            total_loss = total_loss + loss_direct
            loss_dict['loss_direct'] = loss_direct.item()

        # ---- 分支2: 双层自适应 ----
        if self.model.use_two_branches:
            # 第一层: 动态区域划分
            region_features, attn_weights, coverage = self.model.dynamic_region_partition(
                tile_tokens=x,
                coords=coords,
                return_coverage=True,
            )

            # 第二层: ARM 区域聚合
            region_embeddings, region_pooled = self.model.arm(
                region_features=region_features,
                tile_tokens=x,
                attn_weights=attn_weights,
            )

            # 多任务预测
            adaptive_output = self.model.head_adaptive(region_pooled)
            logits_adaptive = adaptive_output['logits']  # [B, num_tasks]

            outputs['region_features'] = region_features
            outputs['region_embeddings'] = region_embeddings
            outputs['attn_weights'] = attn_weights
            outputs['coverage'] = coverage
            outputs['logits_adaptive'] = logits_adaptive
            outputs['adaptive_probs'] = adaptive_output['probs']
            outputs['adaptive_preds'] = adaptive_output['preds']

            if labels is not None:
                loss_adaptive = smoothed_bce(logits_adaptive, labels, self.label_smoothing)
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

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        loss_accum = {}

        all_labels = []
        all_probs_direct = []
        all_probs_adaptive = []

        pbar = tqdm(loader, desc='[Val]')
        for batch in pbar:
            pad_tokens, pad_coords, labels, slide_ids, padding_mask = batch
            pad_tokens = pad_tokens.to(self.device)
            pad_coords = pad_coords.to(self.device)
            labels = labels.to(self.device).float()
            padding_mask = padding_mask.to(self.device)

            with autocast(device_type='cuda'):
                outputs = self._forward_with_tiles(
                    pad_tokens, pad_coords, labels, padding_mask
                )

            total_loss += outputs['loss'].item()
            n_batches += 1
            for k, v in outputs.get('loss_dict', {}).items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v

            all_labels.append(labels.cpu().numpy())
            all_probs_direct.append(torch.sigmoid(outputs['logits_direct']).cpu().numpy())
            if 'logits_adaptive' in outputs:
                all_probs_adaptive.append(torch.sigmoid(outputs['logits_adaptive']).cpu().numpy())

        avg_loss = total_loss / n_batches
        for k in loss_accum:
            loss_accum[k] /= n_batches

        all_labels = np.concatenate(all_labels)
        all_probs_direct = np.concatenate(all_probs_direct)

        # 计算 AUC（防御性：始终确保有值）
        try:
            auc_direct = compute_multitask_auc(all_labels, all_probs_direct)
        except Exception:
            auc_direct = 0.0
        results = {'loss': avg_loss, 'auc_direct': auc_direct}
        results.update({k: v for k, v in loss_accum.items()})

        if all_probs_adaptive:
            all_probs_adaptive = np.concatenate(all_probs_adaptive)
            try:
                auc_adaptive = compute_multitask_auc(all_labels, all_probs_adaptive)
            except Exception:
                auc_adaptive = 0.0
            results['auc_adaptive'] = auc_adaptive

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
    """计算多任务 AUC"""
    try:
        from sklearn.metrics import roc_auc_score
        if labels.shape[1] == 1:
            return roc_auc_score(labels.ravel(), probs.ravel())
        else:
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
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fold(
    fold: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args,
    results_dir: str,
) -> Dict[str, float]:
    print(f"\n{'='*60}")
    print(f"  Fold {fold}")
    print(f"{'='*60}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

    trainer = FusionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_tasks=args.num_tasks,
        task_names=args.task_names,
        early_stopping_patience=args.patience,
        early_stopping_stop_epoch=args.stop_epoch,
        label_smoothing=args.label_smoothing,
    )

    best_auc = 0.0
    best_results = {}

    for epoch in range(args.max_epochs):
        t0 = time.time()

        # 训练
        train_results = trainer.train_epoch(train_loader, epoch)
        train_time = time.time() - t0

        # 验证
        val_results = trainer.validate(val_loader)
        val_time = time.time() - t0 - train_time

        auc = val_results.get('auc_adaptive', val_results.get('auc_direct', 0.0))
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

        # 早停
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

    # 保存最后一个检查点
    trainer.save_checkpoint(epoch, os.path.join(results_dir, f'fold_{fold}_last.pt'))

    print(f"\nFold {fold} Best AUC: {best_auc:.4f}")
    return best_results


# ============================================================
# 主函数
# ============================================================

def get_args():
    parser = argparse.ArgumentParser(description='E2E-ViT + CARE Fusion Training')

    # 数据配置
    parser.add_argument('--dataset', type=str, default='dummy', choices=['dummy', 'real'],
                        help='使用模拟数据集还是真实数据集')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='模拟数据集样本数')
    parser.add_argument('--csv_path', type=str, default='blca_slides.csv')
    parser.add_argument('--data_root_dir', type=str, default='E:/TCGA-data/CPathPatchFeature/blca/chief/pt_files')

    # 模型配置
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_region_tokens', type=int, default=8,
                        help='动态区域数量（K）')
    parser.add_argument('--tile_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--use_alibi', type=bool, default=True)
    parser.add_argument('--use_distillation', type=bool, default=False,
                        help='模拟数据集下关闭蒸馏（没有教师）')
    parser.add_argument('--use_two_branches', type=bool, default=True)

    # 预测配置
    parser.add_argument('--num_tasks', type=int, default=4)
    parser.add_argument('--task_names', type=str, default='ER,PR,HER2,LDH')

    # 训练配置
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--reg', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=15)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--k', type=int, default=5, help='折数')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--stop_epoch', type=int, default=5)
    parser.add_argument('--label_frac', type=float, default=1.0)
    parser.add_argument('--label_smoothing', type=float, default=0.1)

    # 其他配置
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--results_dir', type=str, default='./results')
    parser.add_argument('--exp_code', type=str, default=None)
    parser.add_argument('--testing', action='store_true')

    args = parser.parse_args()

    # 解析 task_names
    if isinstance(args.task_names, str):
        args.task_names = args.task_names.split(',')

    # 默认 exp_code
    if args.exp_code is None:
        args.exp_code = f"{args.dataset}_E2E_CARE_K{args.num_region_tokens}_L{args.num_layers}_T{args.num_tasks}"

    return args


def main(args):
    # 设置随机种子
    seed_everything(args.seed)

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 结果目录
    args.results_dir = os.path.join(args.results_dir, args.exp_code + f'_s{args.seed}')
    os.makedirs(args.results_dir, exist_ok=True)

    # 导入模型
    from models.fusion_model import E2EViTCAREFusion
    from models.utils import print_model_summary

    # 创建模型
    print("\nCreating model...")
    model = E2EViTCAREFusion(
        tile_size=args.tile_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_region_tokens=args.num_region_tokens,
        num_tasks=args.num_tasks,
        task_names=args.task_names,
        use_alibi=args.use_alibi,
        use_distillation=args.use_distillation,
        use_two_branches=args.use_two_branches,
        dropout=args.dropout,
    )

    print_model_summary(model)
    model = model.to(device)

    # 加载数据集
    if args.dataset == 'dummy':
        print(f"\nLoading Dummy WSI Dataset ({args.num_samples} samples)...")
        full_dataset = DummyWSIDataset(
            num_samples=args.num_samples,
            num_tiles_min=100,
            num_tiles_max=500,
            embed_dim=args.embed_dim,
            num_tasks=args.num_tasks,
            seed=args.seed,
        )
    else:
        print(f"\nLoading Real WSI Dataset from {args.data_root_dir}...")
        full_dataset = RealWSIDataset(
            csv_path=args.csv_path,
            data_root=args.data_root_dir,
            embed_dim=args.embed_dim,
            num_tasks=args.num_tasks,
            tile_size=args.tile_size,
        )

    # 5折交叉验证
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

        # 为每个 fold 创建新模型（参数共享但状态独立）
        fold_model = E2EViTCAREFusion(
            tile_size=args.tile_size,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            num_region_tokens=args.num_region_tokens,
            num_tasks=args.num_tasks,
            task_names=args.task_names,
            use_alibi=args.use_alibi,
            use_distillation=False,  # 模拟数据集无蒸馏
            use_two_branches=args.use_two_branches,
            dropout=args.dropout,
        )
        fold_model = fold_model.to(device)

        results = train_fold(
            fold=fold,
            model=fold_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            args=args,
            results_dir=args.results_dir,
        )

        auc = results.get('auc_adaptive', results.get('auc_direct', 0.0))
        all_fold_aucs.append(auc)
        all_fold_results.append(results)

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"  Final Results ({args.k} folds)")
    print(f"{'='*60}")
    for fold, (auc, res) in enumerate(zip(all_fold_aucs, all_fold_results)):
        print(f"  Fold {fold}: AUC={auc:.4f}, Loss={res['loss']:.4f}")
    print(f"\n  Mean AUC: {np.mean(all_fold_aucs):.4f} +/- {np.std(all_fold_aucs):.4f}")
    print(f"  Results saved to: {args.results_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    args = get_args()
    main(args)
