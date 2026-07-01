"""
Cross-Architecture Ensemble: MoTo-CARE + CARE-E2E-Fusion (Adaptive)
对每个样本用 5 折交叉验证中对应的 fold 模型预测，然后融合概率。
"""
from __future__ import annotations
import sys as _sys
import os as _os
import glob as _glob
import re as _re

_sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE')
_sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# ----- MoTo-CARE -----
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('moto_train',
    r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\train.py')
moto_train = _ilu.module_from_spec(_spec)
_sys.modules['moto_train'] = moto_train
_spec.loader.exec_module(moto_train)
RealMoToCAREDataset = moto_train.RealMoToCAREDataset
moto_collate = moto_train.dummy_collate_fn
from moto_care import MoToCARE, MoToCAREConfig

# ----- CARE-E2E-Fusion -----
_spec = _ilu.spec_from_file_location('care_train',
    r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\train.py')
care_train = _ilu.module_from_spec(_spec)
_sys.modules['care_train'] = care_train
_spec.loader.exec_module(care_train)
RealWSIDataset = care_train.RealWSIDataset
care_collate = care_train.dummy_collate_fn
from models.fusion_model import E2EViTCAREFusion

device = torch.device('cuda:0')

csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
data_root = r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files'

# 两个 dataset 用相同 KFold split，但必须保证一致：dataset 顺序对齐
moto_dataset = RealMoToCAREDataset(csv_path=csv_path, data_root=data_root,
                                    input_dim=768, num_tasks=1, max_tiles=4096,
                                    num_regions=8, topology_dim=12,
                                    molecule_dim=128, num_molecule_tokens=4)
care_dataset = RealWSIDataset(csv_path=csv_path, data_root=data_root,
                                embed_dim=768, num_tasks=1, tile_size=256,
                                max_tiles=4096)

# 用相同的 KFold 划分 (random_state=42)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(moto_dataset))))
assert len(moto_dataset) == len(care_dataset), 'datasets size mismatch!'

moto_dir = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\results_real\real_MoToCARE_R8_T1_s42'
care_dir = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42'

moto_bests = sorted(_glob.glob(_os.path.join(moto_dir, 'fold_*_best.pt')))
care_bests = sorted(_glob.glob(_os.path.join(care_dir, 'fold_*_best.pt')))
print(f'MoTo-CARE checkpoints: {len(moto_bests)}')
print(f'CARE-E2E checkpoints: {len(care_bests)}')

moto_cfg = MoToCAREConfig(
    input_dim=768, embed_dim=256, num_regions=8, num_heads=4,
    num_tasks=1, topology_dim=12, molecule_dim=128, top_k_regions=4,
    assignment_temperature=0.35, topology_weight=0.5,
    molecular_weight=0.2, entropy_weight=0.01,
    dropout=0.1, label_smoothing=0.1,
)

all_per_fold = []
for fold, (train_idx, val_idx) in enumerate(splits):
    print(f'\n=== Fold {fold} (val={len(val_idx)}) ===', flush=True)

    # ----- MoTo-CARE prediction -----
    mb = _os.path.join(moto_dir, f'fold_{fold}_best.pt')
    moto_ckpt = torch.load(mb, map_location='cpu')
    moto_model = MoToCARE(moto_cfg).to(device)
    moto_model.load_state_dict(moto_ckpt['model_state_dict'])
    moto_model.eval()

    moto_val = Subset(moto_dataset, val_idx.tolist())
    moto_loader = DataLoader(moto_val, batch_size=1, shuffle=False,
                              collate_fn=moto_collate, num_workers=0)
    moto_probs = []
    moto_labels = []
    with torch.no_grad():
        for batch in moto_loader:
            features, coords, labels, topo_prior, topo_target, mol_tokens, slide_ids, padding_mask = batch
            features = features.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
            topo_prior = topo_prior.to(device)
            topo_target = topo_target.to(device)
            mol_tokens = mol_tokens.to(device)
            padding_mask = padding_mask.to(device)
            out = moto_model(features, coords, labels, topo_prior, topo_target, mol_tokens, padding_mask)
            moto_probs.append(out['probs'].cpu().numpy().ravel())
            moto_labels.append(labels.cpu().numpy().ravel())
    moto_probs = np.concatenate(moto_probs)
    moto_labels = np.concatenate(moto_labels)
    moto_auc = roc_auc_score(moto_labels, moto_probs)
    print(f'  MoTo-CARE: AUC={moto_auc:.4f}', flush=True)
    del moto_model
    torch.cuda.empty_cache()

    # ----- CARE-E2E-Fusion Adaptive prediction -----
    cb = _os.path.join(care_dir, f'fold_{fold}_best.pt')
    care_ckpt = torch.load(cb, map_location='cpu')
    care_model = E2EViTCAREFusion(
        tile_size=256, patch_size=16, embed_dim=768,
        num_heads=4, num_layers=4, num_region_tokens=8,
        num_tasks=1, dropout=0.25,
        use_two_branches=True, use_distillation=False, use_alibi=True,
    ).to(device)
    care_model.load_state_dict(care_ckpt['model_state_dict'])
    care_model.eval()

    care_val = Subset(care_dataset, val_idx.tolist())
    care_loader = DataLoader(care_val, batch_size=1, shuffle=False,
                              collate_fn=care_collate, num_workers=0)
    care_probs = []
    care_labels = []
    with torch.no_grad():
        for batch in care_loader:
            pad_tokens, pad_coords, labels, _, padding_mask = batch
            pad_tokens = pad_tokens.to(device)
            pad_coords = pad_coords.to(device)
            labels = labels.to(device).float()

            region_features, attn_weights, _ = care_model.dynamic_region_partition(
                tile_tokens=pad_tokens, coords=pad_coords, return_coverage=True,
            )
            region_embeddings, region_pooled = care_model.arm(
                region_features=region_features,
                tile_tokens=pad_tokens,
                attn_weights=attn_weights,
            )
            adaptive_out = care_model.head_adaptive(region_pooled)
            logits_adaptive = adaptive_out['logits']
            care_probs.append(torch.sigmoid(logits_adaptive).cpu().numpy().ravel())
            care_labels.append(labels.cpu().numpy().ravel())
    care_probs = np.concatenate(care_probs)
    care_labels = np.concatenate(care_labels)
    care_auc = roc_auc_score(care_labels, care_probs)
    print(f'  CARE Adaptive: AUC={care_auc:.4f}', flush=True)
    del care_model
    torch.cuda.empty_cache()

    assert np.allclose(moto_labels, care_labels), 'label order mismatch!'

    # ----- Ensemble -----
    # 尝试不同权重
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ens = w * moto_probs + (1 - w) * care_probs
        ens_auc = roc_auc_score(care_labels, ens)
        print(f'  Ensemble w_moto={w:.1f}: AUC={ens_auc:.4f}', flush=True)

    ens = 0.5 * moto_probs + 0.5 * care_probs
    ens_auc = roc_auc_score(care_labels, ens)
    all_per_fold.append((fold, moto_auc, care_auc, ens_auc))
    print(f'  ==> Ensemble(0.5/0.5): AUC={ens_auc:.4f}', flush=True)

print('\n' + '=' * 60)
print('=== Cross-Architecture Ensemble (5 folds) ===')
for fold, m, c, e in all_per_fold:
    print(f'  Fold {fold}: MoTo={m:.4f} CARE={c:.4f} Ensemble={e:.4f}')
print(f'  Mean MoTo:    {np.mean([x[1] for x in all_per_fold]):.4f} +/- {np.std([x[1] for x in all_per_fold]):.4f}')
print(f'  Mean CARE:    {np.mean([x[2] for x in all_per_fold]):.4f} +/- {np.std([x[2] for x in all_per_fold]):.4f}')
print(f'  Mean Ensemble:{np.mean([x[3] for x in all_per_fold]):.4f} +/- {np.std([x[3] for x in all_per_fold]):.4f}')
print('=' * 60)