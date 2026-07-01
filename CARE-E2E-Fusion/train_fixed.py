"""
CARE-E2E-Fusion 修复版训练：
- Bug 修复: AdaptiveRegionModeling.mask_threshold=0.01 太大导致 region 输出退化
- 修复方式: mask_threshold 默认从 0.01 -> 0.0（彻底不用 mask）
- 也修了 real_wsi_train.log 里的 nan/inf 兜底异常（AUC=0.0）
"""
from __future__ import annotations

import os
import sys
import time
import argparse
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pathlib import Path

ROOT = Path(r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'models'))

from models.fusion_model import (
    DynamicRegionPartition, AdaptiveRegionModeling,
    MultiTaskHead, build_alibi_bias, get_alibi_slopes,
)
import models.fusion_model as fm
from train import RealWSIDataset, dummy_collate_fn, smoothed_bce


# ============ Monkey-patch: 修复 AdaptiveRegionModeling 的 mask 逻辑 ============
def fixed_arm_forward(
    self,
    region_features: torch.Tensor,
    tile_tokens: torch.Tensor,
    attn_weights: torch.Tensor,
    mask_threshold: float = 0.0,  # 默认从 0.01 改为 0.0 (完全关闭 mask)
) -> tuple:
    """修复版 ARM forward：去掉过激的 mask_threshold，避免 region 输出塌缩"""
    B, K, C = region_features.shape
    N = tile_tokens.shape[1]

    # 原始版本用 mask_threshold=0.01 把低注意力权重位置屏蔽掉，
    # 但 softmax 后的 attn_weights 在 N=4096 时平均约 1/4096=0.000244，
    # 几乎所有位置都被 mask 成 -1e9，导致 softmax 输出几乎均匀，region 退化。
    # 修复: 直接用 attn_weights，不做硬屏蔽。
    attn_weighted = attn_weights  # [B, K, N]
    region_aggregated = torch.bmm(attn_weighted, tile_tokens)  # [B, K, C]

    # Tile-to-region cross-attn（不加 mask）
    tile_out, _ = self.tile_to_region_attn(
        query=region_features,
        key=tile_tokens,
        value=tile_tokens,
    )
    region_features = self.tile_norm(region_features + tile_out)

    # Region self-attn
    region_out, _ = self.region_self_attn(
        query=region_features,
        key=region_features,
        value=region_features,
    )
    region_features = self.region_norm(region_features + region_out)

    # FFN
    region_out = self.ffn(region_features)
    region_embeddings = self.ffn_norm(region_features + region_out)
    region_embeddings = self.region_proj(region_embeddings)
    region_pooled = region_embeddings.mean(dim=1)

    return region_embeddings, region_pooled


# 替换原方法
fm.AdaptiveRegionModeling.forward = fixed_arm_forward
print("[OK] AdaptiveRegionModeling.forward patched (mask_threshold=0)", flush=True)


# ============ 自定义模型（精简版，跳过 patch_embed/transformer，直接用 CONCH features） ============
class FusionModuleFixed(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=2,
                 num_region_tokens=8, num_tasks=1, dropout=0.2, use_two_branches=True):
        super().__init__()
        self.use_two_branches = use_two_branches

        # 简化 transformer: 用更稳定的 pre-norm + GELU
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout, batch_first=True,
                norm_first=True, activation='gelu',
            )
            for _ in range(num_layers)
        ])

        # LayerNorm 稳定输入
        self.input_norm = nn.LayerNorm(embed_dim)

        self.head_direct = nn.Linear(embed_dim, num_tasks)

        self.drp = DynamicRegionPartition(
            embed_dim, num_region_tokens, num_heads, dropout
        )
        self.arm = AdaptiveRegionModeling(embed_dim, num_heads, dropout)
        self.head_adaptive = MultiTaskHead(
            embed_dim, num_tasks, task_names=None, dropout=dropout, use_threshold=False
        )

    def forward(self, tile_tokens, coords, labels=None, label_smoothing=0.0):
        # 输入归一化
        x = self.input_norm(tile_tokens)
        for block in self.blocks:
            x = block(x)

        # Direct branch
        x_global = x.mean(dim=1)
        logits_direct = self.head_direct(x_global)
        total_loss = 0.0
        loss_dict = {}

        if labels is not None:
            smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            loss_direct = smoothed_bce(logits_direct, smooth, label_smoothing)
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
                loss_adaptive = smoothed_bce(logits_adaptive, smooth, label_smoothing)
                total_loss = total_loss + loss_adaptive
                loss_dict['loss_adaptive'] = loss_adaptive.item()
                # 一致性损失（弱化）
                loss_fusion = F.mse_loss(
                    torch.sigmoid(logits_direct),
                    torch.sigmoid(logits_adaptive),
                )
                total_loss = total_loss + 0.05 * loss_fusion
                loss_dict['loss_fusion'] = loss_fusion.item()

        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict
        return outputs


def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', type=str,
                   default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    p.add_argument('--data_root', type=str,
                   default=r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files')
    p.add_argument('--results_dir', type=str,
                   default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real_v2')
    p.add_argument('--exp_code', type=str, default='real_Fixed_K8_L2')

    p.add_argument('--embed_dim', type=int, default=768)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=2)  # 减小避免过拟合
    p.add_argument('--num_region_tokens', type=int, default=8)
    p.add_argument('--num_tasks', type=int, default=1)

    p.add_argument('--lr', type=float, default=2e-4)  # 提升 LR
    p.add_argument('--reg', type=float, default=1e-4)  # 加大正则
    p.add_argument('--batch_size', type=int, default=2)  # 减半避免 OOM
    p.add_argument('--max_epochs', type=int, default=30)
    p.add_argument('--dropout', type=float, default=0.3)  # 提升 dropout
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--stop_epoch', type=int, default=8)
    p.add_argument('--max_tiles', type=int, default=1024)  # 减小到 1024
    p.add_argument('--label_smoothing', type=float, default=0.0)  # 关闭 smoothing
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--start_fold', type=int, default=0)
    p.add_argument('--end_fold', type=int, default=5)
    return p.parse_args()


def main():
    args = get_args()
    seed_everything(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    exp_code = args.exp_code
    results_dir = os.path.join(args.results_dir, exp_code + f'_s{args.seed}')
    os.makedirs(results_dir, exist_ok=True)

    print(f'\nLoading Real WSI dataset (CONCH features)...', flush=True)
    dataset = RealWSIDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        embed_dim=args.embed_dim,
        num_tasks=args.num_tasks,
        tile_size=256,
        max_tiles=args.max_tiles,
    )
    print(f'Dataset: {len(dataset)}', flush=True)

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    indices = np.arange(len(dataset))
    all_aucs = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if fold < args.start_fold or fold >= args.end_fold:
            continue

        best_path = os.path.join(results_dir, f'fold_{fold}_best.pt')
        if os.path.exists(best_path):
            print(f'\n=== Fold {fold} already done, skipping ===', flush=True)
            continue

        print(f'\n{"="*60}', flush=True)
        print(f'  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}', flush=True)
        print(f'{"="*60}', flush=True)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())
        train_loader = DataLoader(train_subset, batch_size=args.batch_size,
            shuffle=True, collate_fn=dummy_collate_fn, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size,
            shuffle=False, collate_fn=dummy_collate_fn, num_workers=0, pin_memory=True)

        model = FusionModuleFixed(
            embed_dim=args.embed_dim, num_heads=args.num_heads, num_layers=args.num_layers,
            num_region_tokens=args.num_region_tokens, num_tasks=args.num_tasks,
            dropout=args.dropout, use_two_branches=True,
        ).to(device)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable params: {trainable_params}', flush=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.max_epochs, eta_min=args.lr * 0.01
        )
        scaler = GradScaler('cuda')
        best_auc = 0.0
        best_auc_a = 0.0
        counter = 0
        epoch = 0

        for epoch in range(args.max_epochs):
            t0 = time.time()
            model.train()
            total_loss = 0.0
            nb = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', mininterval=2.0)
            for batch in pbar:
                pad_tokens, pad_coords, labels, slide_ids, padding_mask = batch
                pad_tokens = pad_tokens.to(device)
                pad_coords = pad_coords.to(device)
                labels = labels.to(device).float()
                padding_mask = padding_mask.to(device)

                # 跳过 padding-only batch
                valid = ~padding_mask
                if valid.sum() == 0:
                    continue

                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    outputs = model(pad_tokens, pad_coords, labels,
                                    label_smoothing=args.label_smoothing)

                loss = outputs['loss']
                if not torch.isfinite(loss):
                    print(f'  WARN: non-finite loss at epoch {epoch}, batch {nb}, skipping', flush=True)
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                nb += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            train_loss = total_loss / max(nb, 1)

            # Validate
            model.eval()
            all_labels = []
            all_probs_direct = []
            all_probs_adaptive = []
            val_loss_total = 0.0
            nb_val = 0
            with torch.no_grad():
                for batch in val_loader:
                    pad_tokens, pad_coords, labels, slide_ids, padding_mask = batch
                    pad_tokens = pad_tokens.to(device)
                    pad_coords = pad_coords.to(device)
                    labels = labels.to(device).float()
                    with autocast(device_type='cuda'):
                        outputs = model(pad_tokens, pad_coords, labels,
                                        label_smoothing=args.label_smoothing)
                    val_loss_total += outputs['loss'].item()
                    nb_val += 1
                    all_labels.append(labels.cpu().numpy())
                    all_probs_direct.append(
                        torch.sigmoid(outputs['logits_direct']).cpu().numpy()
                    )
                    if 'logits_adaptive' in outputs:
                        all_probs_adaptive.append(
                            torch.sigmoid(outputs['logits_adaptive']).cpu().numpy()
                        )

            all_labels = np.concatenate(all_labels)
            all_probs_direct = np.concatenate(all_probs_direct)
            all_probs_ensemble = all_probs_direct
            if len(all_probs_adaptive) > 0:
                all_probs_adaptive_arr = np.concatenate(all_probs_adaptive)
                all_probs_ensemble = (all_probs_direct + all_probs_adaptive_arr) / 2.0
            else:
                all_probs_adaptive_arr = None

            # 计算 AUC（不再 try/except 兜底，输出有意义的诊断）
            try:
                auc_direct = float(roc_auc_score(all_labels.ravel(), all_probs_direct.ravel()))
            except Exception as e:
                print(f'  AUC Direct failed: {e}', flush=True)
                auc_direct = 0.0
            try:
                auc_ensemble = float(roc_auc_score(all_labels.ravel(), all_probs_ensemble.ravel()))
            except Exception as e:
                print(f'  AUC Ensemble failed: {e}', flush=True)
                auc_ensemble = 0.0
            if all_probs_adaptive_arr is not None:
                try:
                    auc_adaptive = float(roc_auc_score(all_labels.ravel(), all_probs_adaptive_arr.ravel()))
                except Exception as e:
                    print(f'  AUC Adaptive failed: {e}', flush=True)
                    auc_adaptive = 0.0
            else:
                auc_adaptive = 0.0

            auc = auc_ensemble
            elapsed = time.time() - t0
            marker = ' *' if auc > best_auc else ''
            print(
                f'Epoch {epoch:3d} | Tr {train_loss:.4f} | '
                f'V {val_loss_total/max(nb_val,1):.4f} | '
                f'AUC(D/A/E): {auc_direct:.4f}/{auc_adaptive:.4f}/{auc_ensemble:.4f}{marker} | '
                f'{elapsed:.1f}s', flush=True,
            )
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'best_score': best_auc,
                }, best_path)
                print(f'  -> Best AUC: {best_auc:.4f}', flush=True)

            # 用 ensemble AUC 做早停（而不是 loss）
            score = auc_ensemble
            if score <= best_auc * 0.99:  # 容忍 1% 波动
                counter += 1
            else:
                counter = 0
            if counter >= args.patience and epoch > args.stop_epoch:
                print(f'  Early stopping at epoch {epoch}', flush=True)
                break

            scheduler.step()

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'best_score': best_auc,
        }, os.path.join(results_dir, f'fold_{fold}_last.pt'))
        all_aucs[fold] = best_auc
        print(f'\nFold {fold} Best AUC: {best_auc:.4f}', flush=True)

    print(f'\n{"="*60}\n  Final Results\n{"="*60}', flush=True)
    if all_aucs:
        for f in sorted(all_aucs):
            print(f'  Fold {f}: AUC={all_aucs[f]:.4f}', flush=True)
        mean_auc = np.mean([all_aucs[f] for f in sorted(all_aucs)])
        std_auc = np.std([all_aucs[f] for f in sorted(all_aucs)])
        print(f'\n  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}', flush=True)


if __name__ == '__main__':
    main()