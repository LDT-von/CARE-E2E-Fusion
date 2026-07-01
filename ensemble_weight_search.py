"""
Weight optimization: 寻找 MoTo-CARE + CARE-E2E 最优融合权重
"""
import sys as _sys
import os as _os
import glob as _glob
import numpy as np
from sklearn.metrics import roc_auc_score

_sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE')
_sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')

import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location('moto_train',
    r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\train.py')
moto_train = _ilu.module_from_spec(_spec)
_sys.modules['moto_train'] = moto_train
_spec.loader.exec_module(moto_train)
RealMoToCAREDataset = moto_train.RealMoToCAREDataset
moto_collate = moto_train.dummy_collate_fn
from moto_care import MoToCARE, MoToCAREConfig

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

moto_dataset = RealMoToCAREDataset(csv_path=csv_path, data_root=data_root,
                                    input_dim=768, num_tasks=1, max_tiles=4096,
                                    num_regions=8, topology_dim=12,
                                    molecule_dim=128, num_molecule_tokens=4)
care_dataset = RealWSIDataset(csv_path=csv_path, data_root=data_root,
                                embed_dim=768, num_tasks=1, tile_size=256,
                                max_tiles=4096)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(moto_dataset))))

moto_dir = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\results_real\real_MoToCARE_R8_T1_s42'
care_dir = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42'

moto_cfg = MoToCAREConfig(
    input_dim=768, embed_dim=256, num_regions=8, num_heads=4,
    num_tasks=1, topology_dim=12, molecule_dim=128, top_k_regions=4,
    assignment_temperature=0.35, topology_weight=0.5,
    molecular_weight=0.2, entropy_weight=0.01,
    dropout=0.1, label_smoothing=0.1,
)

# 收集所有 fold 的概率和标签
fold_data = []
for fold, (train_idx, val_idx) in enumerate(splits):
    print(f'\nFold {fold}...', flush=True)
    # MoTo-CARE
    mb = _os.path.join(moto_dir, f'fold_{fold}_best.pt')
    moto_ckpt = torch.load(mb, map_location='cpu')
    moto_model = MoToCARE(moto_cfg).to(device)
    moto_model.load_state_dict(moto_ckpt['model_state_dict'])
    moto_model.eval()
    moto_val = Subset(moto_dataset, val_idx.tolist())
    moto_loader = DataLoader(moto_val, batch_size=1, shuffle=False,
                              collate_fn=moto_collate, num_workers=0)
    moto_probs, labels_list = [], []
    with torch.no_grad():
        for batch in moto_loader:
            features, coords, labels, topo_prior, topo_target, mol_tokens, slide_ids, padding_mask = batch
            features = features.to(device); coords = coords.to(device)
            labels = labels.to(device); topo_prior = topo_prior.to(device)
            topo_target = topo_target.to(device); mol_tokens = mol_tokens.to(device)
            padding_mask = padding_mask.to(device)
            out = moto_model(features, coords, labels, topo_prior, topo_target, mol_tokens, padding_mask)
            moto_probs.append(out['probs'].cpu().numpy().ravel())
            labels_list.append(labels.cpu().numpy().ravel())
    moto_probs = np.concatenate(moto_probs)
    labels = np.concatenate(labels_list)
    del moto_model; torch.cuda.empty_cache()

    # CARE Adaptive
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
    with torch.no_grad():
        for batch in care_loader:
            pad_tokens, pad_coords, labs, _, padding_mask = batch
            pad_tokens = pad_tokens.to(device); pad_coords = pad_coords.to(device)
            labs = labs.to(device).float()
            region_features, attn_weights, _ = care_model.dynamic_region_partition(
                tile_tokens=pad_tokens, coords=pad_coords, return_coverage=True,
            )
            region_embeddings, region_pooled = care_model.arm(
                region_features=region_features, tile_tokens=pad_tokens,
                attn_weights=attn_weights,
            )
            adaptive_out = care_model.head_adaptive(region_pooled)
            care_probs.append(torch.sigmoid(adaptive_out['logits']).cpu().numpy().ravel())
    care_probs = np.concatenate(care_probs)
    del care_model; torch.cuda.empty_cache()

    # CARE Direct (x_global mean)
    care_model = E2EViTCAREFusion(
        tile_size=256, patch_size=16, embed_dim=768,
        num_heads=4, num_layers=4, num_region_tokens=8,
        num_tasks=1, dropout=0.25,
        use_two_branches=True, use_distillation=False, use_alibi=True,
    ).to(device)
    care_model.load_state_dict(care_ckpt['model_state_dict'])
    care_model.eval()
    care_direct_probs = []
    with torch.no_grad():
        for batch in care_loader:
            pad_tokens, pad_coords, labs, _, padding_mask = batch
            pad_tokens = pad_tokens.to(device); pad_coords = pad_coords.to(device)
            x_global = pad_tokens.mean(dim=1)
            logits_direct = care_model.head_direct(x_global)
            care_direct_probs.append(torch.sigmoid(logits_direct).cpu().numpy().ravel())
    care_direct_probs = np.concatenate(care_direct_probs)
    del care_model; torch.cuda.empty_cache()

    fold_data.append({
        'fold': fold, 'labels': labels,
        'moto': moto_probs, 'care_a': care_probs, 'care_d': care_direct_probs,
    })
    print(f'  MoTo={roc_auc_score(labels, moto_probs):.4f} '
          f'CARE-A={roc_auc_score(labels, care_probs):.4f} '
          f'CARE-D={roc_auc_score(labels, care_direct_probs):.4f}', flush=True)

# Concatenate all folds
all_labels = np.concatenate([f['labels'] for f in fold_data])
all_moto = np.concatenate([f['moto'] for f in fold_data])
all_care_a = np.concatenate([f['care_a'] for f in fold_data])
all_care_d = np.concatenate([f['care_d'] for f in fold_data])

print('\n=== Concatenated Pooled AUC ===')
print(f'  MoTo-CARE:           {roc_auc_score(all_labels, all_moto):.4f}')
print(f'  CARE Adaptive:       {roc_auc_score(all_labels, all_care_a):.4f}')
print(f'  CARE Direct:         {roc_auc_score(all_labels, all_care_d):.4f}')

# Grid search weights
print('\n=== Weight Grid Search (w_moto, w_care_a, w_care_d) ===')
best = (0, None)
for w_moto in np.linspace(0.0, 1.0, 11):
    for w_a in np.linspace(0.0, 1.0 - w_moto, 11):
        w_d = 1.0 - w_moto - w_a
        if w_d < -1e-6: continue
        ens = w_moto * all_moto + w_a * all_care_a + w_d * all_care_d
        auc = roc_auc_score(all_labels, ens)
        if auc > best[0]:
            best = (auc, (w_moto, w_a, w_d))
print(f'  Best weights: w_moto={best[1][0]:.2f} w_care_a={best[1][1]:.2f} w_care_d={best[1][2]:.2f}')
print(f'  Best pooled AUC: {best[0]:.4f}')

# Per-fold best
print('\n=== Per-Fold Mean of Best-Weight Ensemble ===')
fold_aucs_best = []
for f in fold_data:
    ens = best[1][0] * f['moto'] + best[1][1] * f['care_a'] + best[1][2] * f['care_d']
    fold_aucs_best.append(roc_auc_score(f['labels'], ens))
print(f'  Mean: {np.mean(fold_aucs_best):.4f} +/- {np.std(fold_aucs_best):.4f}')