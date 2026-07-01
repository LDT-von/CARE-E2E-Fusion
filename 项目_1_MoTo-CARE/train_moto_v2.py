"""
MoTo-CARE 改进版训练
- 增加 num_regions (8 → 16)
- 增加 region attention 深度
- 增加 embed_dim (256 → 384)
- 25 epochs
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
from torch.utils.data import Subset, DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import KFold
from tqdm import tqdm

from train import (
    RealMoToCAREDataset, dummy_collate_fn, seed_everything,
    compute_multitask_auc, smoothed_bce,
)
from moto_care import MoToCARE, MoToCAREConfig
from moto_care.topology import grid_anchors


# ============ 改进模型 ============

class DeepRegionAttn(nn.Module):
    """2 层 region self-attention + FFN"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        a, _ = self.attn1(x, x, x, need_weights=False)
        x = self.norm1(x + a)
        f = self.ffn(x)
        x = self.norm2(x + f)
        return x


class ImprovedMoToCARE(nn.Module):
    """MoTo-CARE 改进版：
    - 更大 num_regions (16)
    - 更大 embed_dim (384)
    - 2 层 region attention + FFN
    - 保留所有原始功能 (topology, molecule, gate)
    """
    def __init__(self, cfg: MoToCAREConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_proj = nn.Sequential(
            nn.LayerNorm(cfg.input_dim),
            nn.Linear(cfg.input_dim, cfg.embed_dim),
            nn.GELU(),
        )
        from moto_care.model import TopologyAwareAssignment
        self.assignment = TopologyAwareAssignment(cfg)
        self.region_attn = DeepRegionAttn(cfg.embed_dim, cfg.num_heads, cfg.dropout)
        self.topology_head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim), nn.Linear(cfg.embed_dim, cfg.topology_dim)
        )
        self.gate = nn.Linear(cfg.embed_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.embed_dim), nn.Linear(cfg.embed_dim, cfg.num_tasks)
        )
        self.mol_proj = nn.Linear(cfg.molecule_dim, cfg.embed_dim)

    def forward(self, features, coords, labels=None, topology_prior=None,
                topology_target=None, molecule_tokens=None, padding_mask=None):
        from moto_care.losses import molecular_contrastive_loss, topology_loss
        tokens = self.feature_proj(features)
        regions, assignment = self.assignment(tokens, coords, topology_prior, padding_mask)
        regions = self.region_attn(regions)
        topology_pred = self.topology_head(regions)
        gates = F.softmax(self.gate(regions).squeeze(-1), dim=-1)
        slide = torch.einsum("bk,bkd->bd", gates, regions)
        logits = self.head(slide)

        losses = {}
        if labels is not None:
            losses["task"] = smoothed_bce(logits, labels.float(), self.cfg.label_smoothing)
        if topology_target is not None:
            losses["topology"] = topology_loss(topology_pred, topology_target)
        if molecule_tokens is not None:
            mol = self.mol_proj(molecule_tokens)
            align = F.softmax(
                torch.einsum("bld,bkd->blk", mol, regions) / (regions.shape[-1] ** 0.5),
                dim=-1,
            )
            evidence = torch.einsum("blk,bkd->bld", align, regions)
            losses["molecular"] = molecular_contrastive_loss(evidence, mol)

        token_entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(-1)
        if padding_mask is not None:
            entropy = token_entropy.masked_fill(padding_mask, 0.0).sum() / (~padding_mask).sum().clamp_min(1)
        else:
            entropy = token_entropy.mean()
        total = losses.get("task", torch.zeros((), device=features.device))
        total = total + self.cfg.entropy_weight * entropy
        if "topology" in losses:
            total = total + self.cfg.topology_weight * losses["topology"]
        if "molecular" in losses:
            total = total + self.cfg.molecular_weight * losses["molecular"]
        return {
            "logits": logits, "probs": torch.sigmoid(logits),
            "assignment": assignment, "region_tokens": regions,
            "topology_pred": topology_pred, "loss": total, "losses": losses,
        }


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', type=str,
                   default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    p.add_argument('--data_root', type=str,
                   default=r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files')
    p.add_argument('--results_root', type=str,
                   default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\results_v2')
    p.add_argument('--exp_code', type=str, default='real_ImpMoToCARE_R16_T1')

    p.add_argument('--input_dim', type=int, default=768)
    p.add_argument('--embed_dim', type=int, default=384)
    p.add_argument('--num_regions', type=int, default=16)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--top_k_regions', type=int, default=8)
    p.add_argument('--topology_dim', type=int, default=12)
    p.add_argument('--molecule_dim', type=int, default=128)
    p.add_argument('--num_molecule_tokens', type=int, default=4)

    p.add_argument('--assignment_temperature', type=float, default=0.35)
    p.add_argument('--topology_weight', type=float, default=0.5)
    p.add_argument('--molecular_weight', type=float, default=0.2)
    p.add_argument('--entropy_weight', type=float, default=0.01)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--label_smoothing', type=float, default=0.1)

    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--reg', type=float, default=1e-5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_epochs', type=int, default=25)
    p.add_argument('--patience', type=int, default=12)
    p.add_argument('--stop_epoch', type=int, default=8)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--start_fold', type=int, default=0)
    p.add_argument('--end_fold', type=int, default=5)
    p.add_argument('--max_tiles', type=int, default=4096)
    return p.parse_args()


def main(args):
    seed_everything(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}", flush=True)
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    args.results_dir = os.path.join(args.results_root, args.exp_code + f'_s{args.seed}')
    os.makedirs(args.results_dir, exist_ok=True)

    cfg = MoToCAREConfig(
        input_dim=args.input_dim,
        embed_dim=args.embed_dim,
        num_regions=args.num_regions,
        num_heads=args.num_heads,
        num_tasks=1,
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

    print(f"\nLoading Real Dataset from {args.data_root}...", flush=True)
    dataset = RealMoToCAREDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        input_dim=args.input_dim,
        num_tasks=1,
        max_tiles=args.max_tiles,
        num_regions=args.num_regions,
        topology_dim=args.topology_dim,
        molecule_dim=args.molecule_dim,
        num_molecule_tokens=args.num_molecule_tokens,
    )
    print(f"Dataset: {len(dataset)}", flush=True)

    kfold = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
    indices = np.arange(len(dataset))
    fold_aucs = {}

    for fold, (train_idx, val_idx) in enumerate(kfold.split(indices)):
        if fold < args.start_fold or fold >= args.end_fold:
            continue
        best_path = os.path.join(args.results_dir, f'fold_{fold}_best.pt')
        if os.path.exists(best_path):
            print(f"\n=== Fold {fold} already done, skipping ===", flush=True)
            continue

        print(f"\n{'='*60}\n  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}\n{'='*60}", flush=True)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())
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

        model = ImprovedMoToCARE(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)
        scaler = GradScaler('cuda')

        best_auc = 0.0
        best_score = None
        counter = 0
        for epoch in range(args.max_epochs):
            t0 = time.time()
            # train
            model.train()
            total_loss = 0.0
            n_batches = 0
            pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
            for batch in pbar:
                features, coords, labels, topo_prior, topo_target, mol_tokens, slide_ids, padding_mask = batch
                features = features.to(device)
                coords = coords.to(device)
                labels = labels.to(device)
                topo_prior = topo_prior.to(device)
                topo_target = topo_target.to(device)
                mol_tokens = mol_tokens.to(device)
                padding_mask = padding_mask.to(device)
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    out = model(features, coords, labels, topo_prior, topo_target, mol_tokens, padding_mask)
                scaler.scale(out['loss']).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += out['loss'].item()
                n_batches += 1
                pbar.set_postfix({'loss': f'{out["loss"].item():.4f}'})
            train_loss = total_loss / n_batches
            train_t = time.time() - t0

            # validate
            model.eval()
            val_loss = 0.0
            n_v = 0
            all_labels = []
            all_probs = []
            with torch.no_grad():
                for batch in val_loader:
                    features, coords, labels, topo_prior, topo_target, mol_tokens, slide_ids, padding_mask = batch
                    features = features.to(device)
                    coords = coords.to(device)
                    labels = labels.to(device)
                    topo_prior = topo_prior.to(device)
                    topo_target = topo_target.to(device)
                    mol_tokens = mol_tokens.to(device)
                    padding_mask = padding_mask.to(device)
                    with autocast(device_type='cuda'):
                        out = model(features, coords, labels, topo_prior, topo_target, mol_tokens, padding_mask)
                    val_loss += out['loss'].item()
                    n_v += 1
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(out['probs'].cpu().numpy())
            val_loss /= n_v
            val_t = time.time() - t0 - train_t

            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            auc = compute_multitask_auc(all_labels, all_probs)

            marker = " *" if auc > best_auc else ""
            print(
                f"  Epoch {epoch:3d} | Tr {train_loss:.4f} ({train_t:.0f}s) | "
                f"Val {val_loss:.4f} ({val_t:.0f}s) | AUC {auc:.4f}{marker}",
                flush=True,
            )
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'auc': auc,
                }, best_path)
                print(f"    -> Best AUC: {best_auc:.4f}", flush=True)

            score = -val_loss
            if best_score is None or score > best_score:
                best_score = score
                counter = 0
            else:
                counter += 1
                if counter >= args.patience and epoch > args.stop_epoch:
                    print(f"    Early stopping at epoch {epoch}", flush=True)
                    break

        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), 'auc': best_auc,
        }, os.path.join(args.results_dir, f'fold_{fold}_last.pt'))
        fold_aucs[fold] = best_auc
        print(f"\nFold {fold} Best AUC: {best_auc:.4f}", flush=True)

    print(f"\n{'='*60}\n  Final Results\n{'='*60}", flush=True)
    if fold_aucs:
        for f in sorted(fold_aucs):
            print(f"  Fold {f}: AUC={fold_aucs[f]:.4f}", flush=True)
        aucs_list = [fold_aucs[f] for f in sorted(fold_aucs)]
        print(f"\n  Mean AUC: {np.mean(aucs_list):.4f} +/- {np.std(aucs_list):.4f}", flush=True)


if __name__ == '__main__':
    main(get_args())