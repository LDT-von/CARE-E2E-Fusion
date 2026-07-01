# -*- coding: utf-8 -*-
"""
Full CARE-E2E-Fusion evaluation on all 5 folds (L4 model)
"""
from __future__ import print_function
import sys as _sys
import os as _os
import glob as _glob
import re as _re
_sys.path.insert(0, r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion")

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

from train import RealWSIDataset, dummy_collate_fn, compute_multitask_auc
from models.fusion_model import E2EViTCAREFusion

device = torch.device("cuda:0")

csv_path = r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv"
data_root = r"E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files"
dataset = RealWSIDataset(
    csv_path=csv_path,
    data_root=data_root,
    embed_dim=768, num_tasks=1, tile_size=256, max_tiles=4096,
)
print("Dataset loaded:", len(dataset))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(dataset))))

result_dir = r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42"
best_files = sorted(_glob.glob(_os.path.join(result_dir, "fold_*_best.pt")))
print("Found", len(best_files), "best checkpoints")

results = []
for bf in best_files:
    fold_match = _re.search(r"fold_(\d)_", bf)
    fold = int(fold_match.group(1))
    ckpt = torch.load(bf, map_location="cpu")
    ep = ckpt.get("epoch", "?")
    print("\nFold", fold, ": epoch=", ep, flush=True)

    model = E2EViTCAREFusion(
        tile_size=256, patch_size=16, embed_dim=768,
        num_heads=4, num_layers=4, num_region_tokens=8,
        num_tasks=1, dropout=0.25,
        use_two_branches=True, use_distillation=False, use_alibi=True,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, val_idx = splits[fold]
    val_subset = Subset(dataset, val_idx.tolist())
    loader = DataLoader(val_subset, batch_size=1, shuffle=False, collate_fn=dummy_collate_fn, num_workers=0)

    all_labels, all_probs_direct, all_probs_adaptive = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Fold " + str(fold), leave=False):
            pad_tokens, pad_coords, labels, _, padding_mask = batch
            pad_tokens = pad_tokens.to(device)
            pad_coords = pad_coords.to(device)
            labels = labels.to(device).float()

            x_global = pad_tokens.mean(dim=1)
            logits_direct = model.head_direct(x_global)

            region_features, attn_weights, _ = model.dynamic_region_partition(
                tile_tokens=pad_tokens, coords=pad_coords, return_coverage=True,
            )
            region_embeddings, region_pooled = model.arm(
                region_features=region_features,
                tile_tokens=pad_tokens,
                attn_weights=attn_weights,
            )
            adaptive_out = model.head_adaptive(region_pooled)
            logits_adaptive = adaptive_out["logits"]

            all_labels.append(labels.cpu().numpy())
            all_probs_direct.append(torch.sigmoid(logits_direct).cpu().numpy())
            all_probs_adaptive.append(torch.sigmoid(logits_adaptive).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs_direct = np.concatenate(all_probs_direct)
    all_probs_adaptive = np.concatenate(all_probs_adaptive)

    auc_direct = compute_multitask_auc(all_labels, all_probs_direct)
    auc_adaptive = compute_multitask_auc(all_labels, all_probs_adaptive)
    auc_ensemble = compute_multitask_auc(
        all_labels, (all_probs_direct + all_probs_adaptive) / 2.0
    )

    results.append((fold, ep, auc_direct, auc_adaptive, auc_ensemble))
    print("  Direct:", round(auc_direct, 4),
          " Adaptive:", round(auc_adaptive, 4),
          " Ensemble:", round(auc_ensemble, 4), flush=True)

if results:
    print("\n" + "=" * 60)
    print("=== CARE-E2E-Fusion (L4) ===")
    for fold, ep, d, a, e in results:
        print("  Fold", fold, ": Direct=", round(d, 4),
              " Adaptive=", round(a, 4), " Ensemble=", round(e, 4))
    mean_d = np.mean([r[2] for r in results])
    mean_a = np.mean([r[3] for r in results])
    mean_e = np.mean([r[4] for r in results])
    std_d = np.std([r[2] for r in results])
    std_a = np.std([r[3] for r in results])
    std_e = np.std([r[4] for r in results])
    print("  Mean Direct:  ", round(mean_d, 4), "+/-", round(std_d, 4))
    print("  Mean Adaptive:", round(mean_a, 4), "+/-", round(std_a, 4))
    print("  Mean Ensemble:", round(mean_e, 4), "+/-", round(std_e, 4))
    print("=" * 60)
