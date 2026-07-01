"""
Complete CARE-E2E-Fusion evaluation (fixed version)
"""
import sys, os, glob, re
sys.path.insert(0, r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion")

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

from train import RealWSIDataset, dummy_collate_fn, compute_multitask_auc
from models.fusion_model import E2EViTCAREFusion

device = torch.device("cuda:0")

dataset = RealWSIDataset(
    csv_path=r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv",
    data_root=r"E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files",
    embed_dim=768, num_tasks=1, tile_size=256, max_tiles=4096,
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(dataset))))

# L12 model
result_dir = r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L12_T1_s42"
best_files = sorted(glob.glob(os.path.join(result_dir, "fold_*_best.pt")))
print(f"Found {len(best_files)} checkpoints in {result_dir}")

results = []
for bf in best_files:
    fold = int(re.search(r"fold_(\d)_", bf).group(1))
    ckpt = torch.load(bf, map_location="cpu")
    ep = ckpt.get("epoch", "?")
    print(f"\nFold {fold}: epoch={ep}", flush=True)

    model = E2EViTCAREFusion(
        tile_size=256, patch_size=16, embed_dim=768,
        num_heads=12, num_layers=12, num_region_tokens=8,
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
        for batch in tqdm(loader, desc=f"Fold {fold}", leave=False):
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
    print(f"  Direct: {auc_direct:.4f}  Adaptive: {auc_adaptive:.4f}  Ensemble: {auc_ensemble:.4f}", flush=True)

if results:
    print("\n" + "=" * 50)
    print("=== CARE-E2E-Fusion (L12) Results ===")
    for fold, ep, d, a, e in results:
        print(f"  Fold {fold}: Direct={d:.4f} Adaptive={a:.4f} Ensemble={e:.4f}")
    print(f"  Mean Direct:   {np.mean([r[2] for r in results]):.4f}")
    print(f"  Mean Adaptive: {np.mean([r[3] for r in results]):.4f}")
    print(f"  Mean Ensemble: {np.mean([r[4] for r in results]):.4f}")
    print("=" * 50)
