"""
PathwayMorph-OT 改进版训练：
- 用 PCA 2D 伪空间坐标替代 linspace 1D 坐标
- Region pooling 用真正的 2D grid anchor
- 提升 lr / max_epochs / 加 label smoothing
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
from sklearn.model_selection import KFold
from tqdm import tqdm

from train import RealPMOTDataset, dummy_collate_fn, PMOTTrainer, seed_everything
from pathway_morph_ot import PathwayMorphOT, PathwayMorphOTConfig


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', type=str,
                   default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    p.add_argument('--data_root', type=str,
                   default=r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files')
    p.add_argument('--results_root', type=str,
                   default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_2_PathwayMorph-OT\results_real_v2')
    p.add_argument('--exp_code', type=str, default='real_PMOTv2_R256_H128_T1')

    # model
    p.add_argument('--atom_dim', type=int, default=256)
    p.add_argument('--hidden_dim', type=int, default=128)
    p.add_argument('--num_regions', type=int, default=8)
    p.add_argument('--num_pathways', type=int, default=5)
    p.add_argument('--pathway_dim', type=int, default=64)
    p.add_argument('--topology_dim', type=int, default=12)
    p.add_argument('--spatial_dim', type=int, default=4)
    p.add_argument('--epsilon', type=float, default=0.08)
    p.add_argument('--tau', type=float, default=0.8)
    p.add_argument('--ot_iters', type=int, default=40)
    p.add_argument('--ot_cost_weight', type=float, default=0.05)
    p.add_argument('--entropy_weight', type=float, default=0.005)
    p.add_argument('--label_smoothing', type=float, default=0.1)

    # train
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--reg', type=float, default=1e-5)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--max_epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--stop_epoch', type=int, default=5)
    p.add_argument('--k', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_tiles', type=int, default=2048)
    p.add_argument('--gpu', type=str, default='0')
    p.add_argument('--start_fold', type=int, default=0)
    p.add_argument('--end_fold', type=int, default=5)
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

    cfg = PathwayMorphOTConfig(
        atom_dim=args.atom_dim,
        topology_dim=args.topology_dim,
        spatial_dim=args.spatial_dim,
        pathway_dim=args.pathway_dim,
        hidden_dim=args.hidden_dim,
        num_tasks=1,
        epsilon=args.epsilon,
        tau=args.tau,
        ot_iters=args.ot_iters,
        ot_cost_weight=args.ot_cost_weight,
        entropy_weight=args.entropy_weight,
        label_smoothing=args.label_smoothing,
    )

    print(f"\nLoading Real Dataset from {args.data_root}...", flush=True)
    dataset = RealPMOTDataset(
        csv_path=args.csv_path,
        data_root=args.data_root,
        atom_dim=args.atom_dim,
        num_regions=args.num_regions,
        num_pathways=args.num_pathways,
        topology_dim=args.topology_dim,
        spatial_dim=args.spatial_dim,
        pathway_dim=args.pathway_dim,
        num_tasks=1,
        max_tiles=args.max_tiles,
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

        model = PathwayMorphOT(cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

        trainer = PMOTTrainer(
            model=model, optimizer=optimizer, device=device,
            num_tasks=1,
            early_stopping_patience=args.patience,
            early_stopping_stop_epoch=args.stop_epoch,
        )

        best_auc = 0.0
        for epoch in range(args.max_epochs):
            t0 = time.time()
            tr = trainer.train_epoch(train_loader, epoch)
            tr_t = time.time() - t0
            vr = trainer.validate(val_loader)
            vr_t = time.time() - t0 - tr_t

            auc = vr.get('auc', 0.0)
            marker = " *" if auc > best_auc else ""
            print(
                f"  Epoch {epoch:3d} | Tr Loss {tr['loss']:.4f} ({tr_t:.0f}s) | "
                f"Val Loss {vr['loss']:.4f} ({vr_t:.0f}s) | AUC {auc:.4f}{marker}",
                flush=True,
            )
            if auc > best_auc:
                best_auc = auc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'auc': auc,
                }, best_path)
                print(f"    -> Best AUC: {best_auc:.4f}", flush=True)

            score = -vr['loss']
            if trainer.best_score is None or score > trainer.best_score:
                trainer.best_score = score
                trainer.counter = 0
            else:
                trainer.counter += 1
                if trainer.counter >= trainer.patience and epoch > trainer.stop_epoch:
                    print(f"    Early stopping at epoch {epoch}", flush=True)
                    break

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'auc': best_auc,
        }, os.path.join(args.results_dir, f'fold_{fold}_last.pt'))

        fold_aucs[fold] = best_auc
        print(f"\nFold {fold} Best AUC: {best_auc:.4f}", flush=True)

    print(f"\n{'='*60}\n  Final Results\n{'='*60}", flush=True)
    if fold_aucs:
        aucs_list = [fold_aucs[f] for f in sorted(fold_aucs)]
        for f in sorted(fold_aucs):
            print(f"  Fold {f}: AUC={fold_aucs[f]:.4f}", flush=True)
        print(f"\n  Mean AUC: {np.mean(aucs_list):.4f} +/- {np.std(aucs_list):.4f}", flush=True)


if __name__ == '__main__':
    args = get_args()
    main(args)