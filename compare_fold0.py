"""
Fold 0 性能对比：修复前 vs 修复后 + MoTo-CARE + Ensemble
"""
import sys as _sys
import os as _os
import glob as _glob
import importlib.util as _ilu
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

_sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')
_sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE')

# CARE-E2E-Fusion imports
_spec = _ilu.spec_from_file_location('care_train',
    r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\train.py')
care_train = _ilu.module_from_spec(_spec)
_sys.modules['care_train'] = care_train
_spec.loader.exec_module(care_train)
RealWSIDataset = care_train.RealWSIDataset
care_collate = care_train.dummy_collate_fn
from models.fusion_model import E2EViTCAREFusion

# MoTo-CARE imports
_spec = _ilu.spec_from_file_location('moto_train',
    r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\train.py')
moto_train = _ilu.module_from_spec(_spec)
_sys.modules['moto_train'] = moto_train
_spec.loader.exec_module(moto_train)
RealMoToCAREDataset = moto_train.RealMoToCAREDataset
moto_collate = moto_train.dummy_collate_fn
from moto_care import MoToCARE, MoToCAREConfig

device = torch.device('cuda:0')

csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
data_root = r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files'

care_dataset = RealWSIDataset(csv_path=csv_path, data_root=data_root,
                              embed_dim=768, num_tasks=1, tile_size=256, max_tiles=4096)
moto_dataset = RealMoToCAREDataset(csv_path=csv_path, data_root=data_root,
                                    input_dim=768, num_tasks=1, max_tiles=4096,
                                    num_regions=8, topology_dim=12,
                                    molecule_dim=128, num_molecule_tokens=4)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(care_dataset))))
_, val_idx = splits[0]

# ---- 1) 修复前 CARE-E2E-Fusion L4 ----
print('\n[1] Old CARE-E2E-Fusion L4 (broken) ...', flush=True)
old_care = E2EViTCAREFusion(
    tile_size=256, patch_size=16, embed_dim=768,
    num_heads=4, num_layers=4, num_region_tokens=8,
    num_tasks=1, dropout=0.25,
    use_two_branches=True, use_distillation=False, use_alibi=True,
).to(device)
old_care_ckpt = torch.load(r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42\fold_0_best.pt', map_location='cpu')
old_care.load_state_dict(old_care_ckpt['model_state_dict'])
old_care.eval()

care_loader = DataLoader(Subset(care_dataset, val_idx.tolist()), batch_size=1,
                          shuffle=False, collate_fn=care_collate, num_workers=0)
old_labels, old_d, old_a = [], [], []
with torch.no_grad():
    for batch in care_loader:
        pad_tokens, pad_coords, labels, _, _ = batch
        pad_tokens = pad_tokens.to(device); pad_coords = pad_coords.to(device)
        labels = labels.to(device).float()
        x_global = pad_tokens.mean(dim=1)
        logits_direct = old_care.head_direct(x_global)
        region_features, attn_weights, _ = old_care.dynamic_region_partition(
            tile_tokens=pad_tokens, coords=pad_coords, return_coverage=True,
        )
        region_embeddings, region_pooled = old_care.arm(
            region_features=region_features, tile_tokens=pad_tokens, attn_weights=attn_weights,
        )
        adaptive_out = old_care.head_adaptive(region_pooled)
        old_labels.append(labels.cpu().numpy().ravel())
        old_d.append(torch.sigmoid(logits_direct).cpu().numpy().ravel())
        old_a.append(torch.sigmoid(adaptive_out['logits']).cpu().numpy().ravel())
old_labels = np.concatenate(old_labels)
old_d = np.concatenate(old_d)
old_a = np.concatenate(old_a)
print(f'  Old Direct AUC: {roc_auc_score(old_labels, old_d):.4f}', flush=True)
print(f'  Old Adaptive AUC: {roc_auc_score(old_labels, old_a):.4f}', flush=True)
print(f'  Old Ensemble AUC: {roc_auc_score(old_labels, (old_d + old_a) / 2):.4f}', flush=True)
del old_care; torch.cuda.empty_cache()

care_train_sys = _sys.modules['care_train']
_spec = _ilu.spec_from_file_location('train_fixed',
    r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\train_fixed.py')
train_fixed = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(train_fixed)
FusionModuleFixed = train_fixed.FusionModuleFixed

# ---- 2) 修复后 CARE-E2E-Fusion (fold 0) ----
print('\n[2] Fixed CARE-E2E-Fusion (fold 0) ...', flush=True)
new_care = FusionModuleFixed(
    embed_dim=768, num_heads=8, num_layers=2,
    num_region_tokens=8, num_tasks=1, dropout=0.3, use_two_branches=True,
).to(device)
new_care_ckpt = torch.load(r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real_v2\real_Fixed_K8_L2_s42\fold_0_best.pt', map_location='cpu')
new_care.load_state_dict(new_care_ckpt['model_state_dict'])
new_care.eval()

new_d, new_a = [], []
with torch.no_grad():
    for batch in care_loader:
        pad_tokens, pad_coords, labels, _, _ = batch
        pad_tokens = pad_tokens.to(device); pad_coords = pad_coords.to(device)
        labels = labels.to(device).float()
        # 用同样的 forward 路径
        out = new_care(pad_tokens, pad_coords, labels, label_smoothing=0.0)
        new_d.append(torch.sigmoid(out['logits_direct']).cpu().numpy().ravel())
        if 'logits_adaptive' in out:
            new_a.append(torch.sigmoid(out['logits_adaptive']).cpu().numpy().ravel())
new_d = np.concatenate(new_d)
new_a = np.concatenate(new_a)
print(f'  New Direct AUC: {roc_auc_score(old_labels, new_d):.4f}', flush=True)
print(f'  New Adaptive AUC: {roc_auc_score(old_labels, new_a):.4f}', flush=True)
print(f'  New Ensemble AUC: {roc_auc_score(old_labels, (new_d + new_a) / 2):.4f}', flush=True)
del new_care; torch.cuda.empty_cache()

# ---- 3) MoTo-CARE ----
print('\n[3] MoTo-CARE ...', flush=True)
moto_cfg = MoToCAREConfig(
    input_dim=768, embed_dim=256, num_regions=8, num_heads=4,
    num_tasks=1, topology_dim=12, molecule_dim=128, top_k_regions=4,
    assignment_temperature=0.35, topology_weight=0.5,
    molecular_weight=0.2, entropy_weight=0.01,
    dropout=0.1, label_smoothing=0.1,
)
moto_model = MoToCARE(moto_cfg).to(device)
moto_ckpt = torch.load(r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\results_real\real_MoToCARE_R8_T1_s42\fold_0_best.pt', map_location='cpu')
moto_model.load_state_dict(moto_ckpt['model_state_dict'])
moto_model.eval()

moto_loader = DataLoader(Subset(moto_dataset, val_idx.tolist()), batch_size=1,
                          shuffle=False, collate_fn=moto_collate, num_workers=0)
moto_probs = []
with torch.no_grad():
    for batch in moto_loader:
        features, coords, labels, topo_prior, topo_target, mol_tokens, slide_ids, padding_mask = batch
        features = features.to(device); coords = coords.to(device)
        labels = labels.to(device); topo_prior = topo_prior.to(device)
        topo_target = topo_target.to(device); mol_tokens = mol_tokens.to(device)
        padding_mask = padding_mask.to(device)
        out = moto_model(features, coords, labels, topo_prior, topo_target, mol_tokens, padding_mask)
        moto_probs.append(out['probs'].cpu().numpy().ravel())
moto_probs = np.concatenate(moto_probs)
print(f'  MoTo AUC: {roc_auc_score(old_labels, moto_probs):.4f}', flush=True)

# ---- 4) 3-way Ensemble: Old CARE + New CARE + MoTo ----
print('\n[4] 3-way Ensembles ...', flush=True)
ens_old_new_moto = (old_d + new_a + moto_probs) / 3
print(f'  Old_Direct + New_Adaptive + MoTo AUC: {roc_auc_score(old_labels, ens_old_new_moto):.4f}', flush=True)

ens_new_d_a_moto = (new_d + new_a + moto_probs) / 3
print(f'  New_Direct + New_Adaptive + MoTo AUC: {roc_auc_score(old_labels, ens_new_d_a_moto):.4f}', flush=True)

ens_new_a_moto = (new_a + moto_probs) / 2
print(f'  New_Adaptive + MoTo AUC: {roc_auc_score(old_labels, ens_new_a_moto):.4f}', flush=True)

ens_new_e_moto = ((new_d + new_a) / 2 + moto_probs) / 2
print(f'  New_Ensemble + MoTo AUC: {roc_auc_score(old_labels, ens_new_e_moto):.4f}', flush=True)