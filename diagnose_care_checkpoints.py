"""
诊断脚本：检查现有 CARE-E2E-Fusion L4 checkpoints 的预测分布
看 label_smoothing=0.1 是否把 sigmoid 输出推到 0.5 附近
"""
import sys as _sys
import os as _os
import glob as _glob

_sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from train import RealWSIDataset, dummy_collate_fn
from models.fusion_model import E2EViTCAREFusion

device = torch.device('cuda:0')

csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
data_root = r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files'

dataset = RealWSIDataset(
    csv_path=csv_path, data_root=data_root,
    embed_dim=768, num_tasks=1, tile_size=256, max_tiles=4096,
)
print('Dataset:', len(dataset))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(dataset))))

result_dir = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42'
best_files = sorted(_glob.glob(_os.path.join(result_dir, 'fold_*_best.pt')))
print('Checkpoints:', len(best_files))

for bf in best_files:
    fold = int(_os.path.basename(bf).split('_')[1])
    print(f'\n=== Fold {fold} ===', flush=True)
    ckpt = torch.load(bf, map_location='cpu')
    print(f"  epoch={ckpt.get('epoch','?')}", flush=True)
    print(f"  keys in ckpt:", list(ckpt.keys())[:5], flush=True)
    if 'best_score' in ckpt:
        print(f"  best_score (during training): {ckpt['best_score']}", flush=True)

    model = E2EViTCAREFusion(
        tile_size=256, patch_size=16, embed_dim=768,
        num_heads=4, num_layers=4, num_region_tokens=8,
        num_tasks=1, dropout=0.25,
        use_two_branches=True, use_distillation=False, use_alibi=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, val_idx = splits[fold]
    val_subset = Subset(dataset, val_idx.tolist())
    loader = DataLoader(val_subset, batch_size=1, shuffle=False,
                         collate_fn=dummy_collate_fn, num_workers=0)

    all_labels = []
    all_probs_d = []
    all_probs_a = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Fold {fold}', leave=False):
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
                region_features=region_features, tile_tokens=pad_tokens,
                attn_weights=attn_weights,
            )
            adaptive_out = model.head_adaptive(region_pooled)
            logits_adaptive = adaptive_out['logits']

            all_labels.append(labels.cpu().numpy().ravel())
            all_probs_d.append(torch.sigmoid(logits_direct).cpu().numpy().ravel())
            all_probs_a.append(torch.sigmoid(logits_adaptive).cpu().numpy().ravel())

    all_labels = np.concatenate(all_labels)
    all_probs_d = np.concatenate(all_probs_d)
    all_probs_a = np.concatenate(all_probs_a)
    print(f"  labels: pos={all_labels.sum():.0f}/{len(all_labels)}", flush=True)
    print(f"  direct probs: mean={all_probs_d.mean():.4f} std={all_probs_d.std():.4f} "
          f"min={all_probs_d.min():.4f} max={all_probs_d.max():.4f}", flush=True)
    print(f"  adapt probs:  mean={all_probs_a.mean():.4f} std={all_probs_a.std():.4f} "
          f"min={all_probs_a.min():.4f} max={all_probs_a.max():.4f}", flush=True)

    try:
        auc_d = roc_auc_score(all_labels, all_probs_d)
    except Exception as e:
        auc_d = f'ERR({e})'
    try:
        auc_a = roc_auc_score(all_labels, all_probs_a)
    except Exception as e:
        auc_a = f'ERR({e})'
    print(f"  AUC Direct={auc_d}", flush=True)
    print(f"  AUC Adaptive={auc_a}", flush=True)

    # logits 分布
    print(f"  Direct logits: mean={(all_probs_d - 0.5).mean():.4f} "
          f"(should be near 0.5 if model predicts majority class)", flush=True)