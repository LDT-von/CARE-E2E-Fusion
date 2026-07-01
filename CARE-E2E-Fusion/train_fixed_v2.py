"""
CARE-E2E-Fusion v2 训练脚本 —— 全面改进版
===========================================
改动（按优先级）:
  1. [蒸馏] 新增 distill_loss：student transformer tokens → teacher CONCH features KL divergence
  2. [容量] num_layers 2→4, num_region_tokens 8→16
  3. [集成] 改为 learned sigmoid gating（替代固定 0.5 平均）
  4. [训练] Label smoothing 0.05 + Linear warmup + CosineAnnealing
  5. [正则] dropout 0.3→0.35, weight_decay 1e-4→2e-4
  6. [Bug] 修复 ARM mask_threshold=0.01 → 0（继承自 train_fixed.py）
"""
from __future__ import annotations

import os, sys, time, argparse, pickle, random
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


# ============ Monkey-patch: 修复 AdaptiveRegionModeling ============
def fixed_arm_forward(self, region_features, tile_tokens, attn_weights,
                      mask_threshold=0.0):
    B, K, C = region_features.shape
    attn_weighted = attn_weights
    region_aggregated = torch.bmm(attn_weighted, tile_tokens)
    tile_out, _ = self.tile_to_region_attn(
        query=region_features, key=tile_tokens, value=tile_tokens)
    region_features = self.tile_norm(region_features + tile_out)
    region_out, _ = self.region_self_attn(
        query=region_features, key=region_features, value=region_features)
    region_features = self.region_norm(region_features + region_out)
    region_out = self.ffn(region_features)
    region_embeddings = self.ffn_norm(region_features + region_out)
    region_embeddings = self.region_proj(region_embeddings)
    region_pooled = region_embeddings.mean(dim=1)
    return region_embeddings, region_pooled

fm.AdaptiveRegionModeling.forward = fixed_arm_forward
print("[OK] AdaptiveRegionModeling.forward patched (mask_threshold=0)", flush=True)


# ============ Distillation Loss ============
class DistillKL(nn.Module):
    """KL-divergence based knowledge distillation at tile-token level."""
    def __init__(self, temperature=3.0):
        super().__init__()
        self.T = temperature

    def forward(self, student_tokens, teacher_tokens):
        """student/teacher: [B, N, C]"""
        s = student_tokens / self.T
        t = teacher_tokens / self.T
        # mean-reduced KL for numerical stability
        kl = F.kl_div(
            F.log_softmax(s, dim=-1),
            F.softmax(t.detach(), dim=-1),
            reduction='batchmean',
        ) * (self.T ** 2)
        return kl


# ============ Learned Gating Ensemble ============
class GatedEnsemble(nn.Module):
    """Learnable gating between direct and adaptive branches.
    gate = sigmoid(w_d * h_d + w_a * h_a + b)
    final = gate * direct + (1 - gate) * adaptive
    """
    def __init__(self, embed_dim=768, dropout=0.3):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, direct_features, adaptive_features):
        """
        direct_features:  [B, embed_dim]  — from mean-pool after transformer
        adaptive_features: [B, embed_dim]  — from ARM region-pooled
        Returns [B, 1] sigmoid gate weight for direct branch.
        """
        concat = torch.cat([direct_features, adaptive_features], dim=-1)
        gate = torch.sigmoid(self.gate_net(concat))          # [B, 1]
        return gate


# ============ 模型定义 ============
class FusionModuleV2(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, num_layers=4,
                 num_region_tokens=16, num_tasks=1, dropout=0.35,
                 use_two_branches=True, distill_weight=0.5,
                 use_distillation=False):
        super().__init__()
        self.use_two_branches = use_two_branches
        self.distill_weight = distill_weight
        self.use_distillation = use_distillation

        # ---- Backbone ----
        self.input_norm = nn.LayerNorm(embed_dim)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout, batch_first=True,
                norm_first=True, activation='gelu',
            )
            for _ in range(num_layers)
        ])

        # ---- Direct branch ----
        self.head_direct = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_tasks),
        )

        # ---- Adaptive branch ----
        self.drp = DynamicRegionPartition(embed_dim, num_region_tokens, num_heads, dropout)
        self.arm = AdaptiveRegionModeling(embed_dim, num_heads, dropout)
        self.head_adaptive = MultiTaskHead(
            embed_dim, num_tasks, task_names=None,
            dropout=dropout, use_threshold=False,
        )

        # ---- Gating ensemble ----
        if use_two_branches:
            self.gate = GatedEnsemble(embed_dim, dropout)

        # ---- Distillation ----
        if use_distillation:
            self.distill_kl = DistillKL(temperature=3.0)

    def forward(self, tile_tokens, coords, labels=None,
                label_smoothing=0.05, teacher_tokens=None):
        """
        tile_tokens:    [B, N, 768]  — CONCH features
        coords:         [B, N, 2]     — tile coordinates
        teacher_tokens: [B, N, 768]  — CONCH features (same as input, for distillation)
        """
        B = tile_tokens.size(0)
        x = self.input_norm(tile_tokens)
        for block in self.blocks:
            x = block(x)

        # ---- Direct ----
        x_global = x.mean(dim=1)                     # [B, 768]
        logits_direct = self.head_direct(x_global)     # [B, num_tasks]

        total_loss = 0.0
        loss_dict = {}
        if labels is not None:
            smooth = labels * (1 - label_smoothing) + 0.5 * label_smoothing
            loss_direct = smoothed_bce(logits_direct, smooth, label_smoothing)
            total_loss = total_loss + loss_direct
            loss_dict['loss_direct'] = loss_direct.item()

        outputs = {
            'logits_direct': logits_direct,
            'transformer_output': x,
            'direct_features': x_global,
            'loss': total_loss,
            'loss_dict': loss_dict,
        }

        # ---- Adaptive ----
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
            outputs['adaptive_features'] = region_pooled

            if labels is not None:
                loss_adaptive = smoothed_bce(logits_adaptive, smooth, label_smoothing)
                total_loss = total_loss + loss_adaptive
                loss_dict['loss_adaptive'] = loss_adaptive.item()

                # Learned gating ensemble loss
                gate = self.gate(x_global, region_pooled)       # [B, 1]
                logits_ensemble = gate * torch.sigmoid(logits_direct) + \
                                  (1 - gate) * torch.sigmoid(logits_adaptive)
                logits_ensemble = logits_ensemble.clamp(1e-6, 1 - 1e-6)
                logits_ensemble = torch.logit(logits_ensemble)
                loss_ensemble = F.binary_cross_entropy_with_logits(
                    logits_ensemble, smooth)
                total_loss = total_loss + 0.05 * loss_ensemble
                loss_dict['loss_ensemble'] = loss_ensemble.item()
                loss_dict['gate_mean'] = gate.mean().item()

            # ---- Distillation ----
            if self.use_distillation and teacher_tokens is not None:
                distill_loss = self.distill_kl(x, teacher_tokens)
                total_loss = total_loss + self.distill_weight * distill_loss
                loss_dict['loss_distill'] = distill_loss.item()

            outputs['loss'] = total_loss
            outputs['loss_dict'] = loss_dict

        return outputs


def get_ensemble_probs(outputs):
    """Inference 时用 learned gate 做 ensemble（不依赖 labels）。"""
    logits_d = outputs['logits_direct']
    if 'logits_adaptive' not in outputs:
        return torch.sigmoid(logits_d)

    logits_a = outputs['logits_adaptive']
    gate = outputs.get('_gate', None)
    if gate is not None:
        probs = gate * torch.sigmoid(logits_d) + (1 - gate) * torch.sigmoid(logits_a)
    else:
        probs = (torch.sigmoid(logits_d) + torch.sigmoid(logits_a)) * 0.5
    return probs


# ============ 工具函数 ============
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument('--csv_path', type=str,
        default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    p.add_argument('--data_root', type=str,
        default=r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files')
    p.add_argument('--results_dir', type=str,
        default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real_v3')
    p.add_argument('--exp_code', type=str, default='real_v2_K16_L4_Distill')
    # 模型
    p.add_argument('--embed_dim', type=int, default=768)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--num_layers', type=int, default=4)        # 2→4
    p.add_argument('--num_region_tokens', type=int, default=16) # 8→16
    p.add_argument('--num_tasks', type=int, default=1)
    # 训练
    p.add_argument('--lr', type=float, default=3e-4)           # 2e-4→3e-4
    p.add_argument('--warmup_epochs', type=int, default=3)     # 新增 warmup
    p.add_argument('--reg', type=float, default=2e-4)           # 1e-4→2e-4
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--max_epochs', type=int, default=40)
    p.add_argument('--dropout', type=float, default=0.35)        # 0.3→0.35
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--stop_epoch', type=int, default=10)
    # 蒸馏
    p.add_argument('--use_distillation', type=bool, default=True)  # 开启！
    p.add_argument('--distill_weight', type=float, default=0.5)  # distill loss 权重
    # 其他
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--max_tiles', type=int, default=1024)
    p.add_argument('--label_smoothing', type=float, default=0.05)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--start_fold', type=int, default=0)
    p.add_argument('--end_fold', type=int, default=5)
    return p.parse_args()


def get_scheduler(optimizer, warmup_epochs, max_epochs, steps_per_epoch):
    """Linear warmup → CosineAnnealing."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = max_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(warmup_steps, 1))
        progress = float(step - warmup_steps) / float(max(total_steps - warmup_steps, 1))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============ 训练循环 ============
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, label_smoothing):
    model.train()
    total_loss = 0.0
    nb = 0
    for batch in loader:
        pad_tokens, pad_coords, labels, _, padding_mask = batch
        pad_tokens = pad_tokens.to(device)
        pad_coords = pad_coords.to(device)
        labels = labels.to(device).float()
        teacher_tokens = pad_tokens  # CONCH features 作为 teacher

        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            outputs = model(
                pad_tokens, pad_coords, labels,
                label_smoothing=label_smoothing,
                teacher_tokens=teacher_tokens if model.use_distillation else None,
            )

        loss = outputs['loss']
        if not torch.isfinite(loss):
            print(f'  WARN: non-finite loss, skipping', flush=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += loss.item()
        nb += 1

    return total_loss / max(nb, 1)


@torch.no_grad()
def validate(model, loader, device, label_smoothing):
    model.eval()
    all_labels, all_probs_d, all_probs_a, all_probs_e = [], [], [], []
    val_loss_total = 0.0
    nb_val = 0

    for batch in loader:
        pad_tokens, pad_coords, labels, _, padding_mask = batch
        pad_tokens = pad_tokens.to(device)
        pad_coords = pad_coords.to(device)
        labels = labels.to(device).float()

        with autocast(device_type='cuda'):
            outputs = model(pad_tokens, pad_coords, labels,
                            label_smoothing=label_smoothing)

        val_loss_total += outputs['loss'].item()
        nb_val += 1
        all_labels.append(labels.cpu().numpy())

        pd_val = torch.sigmoid(outputs['logits_direct']).cpu().numpy()
        all_probs_d.append(pd_val)

        if 'logits_adaptive' in outputs:
            pa_val = torch.sigmoid(outputs['logits_adaptive']).cpu().numpy()
            all_probs_a.append(pa_val)
            # Inference gate
            gate = model.gate(outputs['direct_features'],
                              outputs['adaptive_features']).cpu().detach().numpy()
            pe_val = gate * pd_val + (1 - gate) * pa_val
            all_probs_e.append(pe_val)

    all_labels = np.concatenate(all_labels)
    all_probs_d = np.concatenate(all_probs_d)
    all_probs_a = np.concatenate(all_probs_a) if all_probs_a else None
    all_probs_e = np.concatenate(all_probs_e) if all_probs_e else all_probs_d

    try:
        auc_d = roc_auc_score(all_labels.ravel(), all_probs_d.ravel())
    except:
        auc_d = 0.0
    try:
        auc_e = roc_auc_score(all_labels.ravel(), all_probs_e.ravel())
    except:
        auc_e = 0.0
    auc_a = 0.0
    if all_probs_a is not None:
        try:
            auc_a = roc_auc_score(all_labels.ravel(), all_probs_a.ravel())
        except:
            auc_a = 0.0

    return auc_d, auc_a, auc_e, val_loss_total / max(nb_val, 1)


# ============ 主函数 ============
def main():
    args = get_args()
    seed_everything(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}', flush=True)

    results_dir = os.path.join(args.results_dir, args.exp_code + f'_s{args.seed}')
    os.makedirs(results_dir, exist_ok=True)
    print(f'\n[Config]', flush=True)
    print(f'  num_layers={args.num_layers}, num_region_tokens={args.num_region_tokens}', flush=True)
    print(f'  dropout={args.dropout}, lr={args.lr}, reg={args.reg}', flush=True)
    print(f'  use_distillation={args.use_distillation}, distill_weight={args.distill_weight}', flush=True)
    print(f'  label_smoothing={args.label_smoothing}, warmup={args.warmup_epochs}ep', flush=True)

    dataset = RealWSIDataset(
        csv_path=args.csv_path, data_root=args.data_root,
        embed_dim=args.embed_dim, num_tasks=args.num_tasks,
        tile_size=256, max_tiles=args.max_tiles,
    )
    print(f'Dataset: {len(dataset)}', flush=True)

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    indices = np.arange(len(dataset))
    all_results = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if fold < args.start_fold or fold >= args.end_fold:
            continue

        best_path = os.path.join(results_dir, f'fold_{fold}_best.pt')
        if os.path.exists(best_path):
            print(f'\n=== Fold {fold} already done, skipping ===', flush=True)
            ckpt = torch.load(best_path, map_location='cpu', weights_only=False)
            all_results[fold] = ckpt.get('best_score', 0.0)
            continue

        print(f'\n{"="*60}', flush=True)
        print(f'  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}', flush=True)
        print(f'{"="*60}', flush=True)

        train_loader = DataLoader(
            Subset(dataset, train_idx.tolist()),
            batch_size=args.batch_size, shuffle=True,
            collate_fn=dummy_collate_fn, num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx.tolist()),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=dummy_collate_fn, num_workers=0, pin_memory=True,
        )

        model = FusionModuleV2(
            embed_dim=args.embed_dim, num_heads=args.num_heads,
            num_layers=args.num_layers, num_region_tokens=args.num_region_tokens,
            num_tasks=args.num_tasks, dropout=args.dropout,
            use_two_branches=True,
            distill_weight=args.distill_weight,
            use_distillation=args.use_distillation,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Trainable params: {n_params:,}', flush=True)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.reg,
        )
        steps_per_epoch = len(train_loader)
        scheduler = get_scheduler(optimizer, args.warmup_epochs,
                                  args.max_epochs, steps_per_epoch)
        scaler = GradScaler('cuda')

        best_auc = 0.0
        counter = 0

        for epoch in range(args.max_epochs):
            t0 = time.time()
            train_loss = train_one_epoch(
                model, train_loader, optimizer, scheduler, scaler,
                device, args.label_smoothing,
            )
            auc_d, auc_a, auc_e, val_loss = validate(
                model, val_loader, device, args.label_smoothing,
            )
            elapsed = time.time() - t0
            marker = ' *' if auc_e > best_auc else ''
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f'Ep {epoch:3d} | TrL {train_loss:.4f} | VL {val_loss:.4f} | '
                f'AUC(D/A/E): {auc_d:.4f}/{auc_a:.4f}/{auc_e:.4f}{marker} | '
                f'LR {lr_now:.2e} | {elapsed:.1f}s',
                flush=True,
            )

            if auc_e > best_auc:
                best_auc = auc_e
                torch.save({
                    'epoch': epoch, 'fold': fold,
                    'model_state_dict': model.state_dict(),
                    'best_score': best_auc,
                    'auc_d': auc_d, 'auc_a': auc_a,
                }, best_path)
                print(f'  -> Best AUC: {best_auc:.4f}', flush=True)
                counter = 0
            else:
                counter += 1

            if counter >= args.patience and epoch > args.stop_epoch:
                print(f'  Early stopping at epoch {epoch}', flush=True)
                break

        all_results[fold] = best_auc
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'best_score': best_auc,
        }, os.path.join(results_dir, f'fold_{fold}_last.pt'))
        print(f'Fold {fold} Best AUC: {best_auc:.4f}', flush=True)

    # ---- 汇总 ----
    print(f'\n{"="*60}\n  Final Results\n{"="*60}', flush=True)
    for f in sorted(all_results):
        print(f'  Fold {f}: AUC={all_results[f]:.4f}', flush=True)
    mean_auc = np.mean(list(all_results.values()))
    std_auc  = np.std(list(all_results.values()))
    print(f'\n  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}', flush=True)
    print(f'  Results saved to: {results_dir}', flush=True)


if __name__ == '__main__':
    main()
