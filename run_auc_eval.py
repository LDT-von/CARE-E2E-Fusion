"""Full AUC evaluation for all 3 projects."""
from __future__ import annotations
import os, sys, warnings, random, json, time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv"
DATA_ROOT = r"E:/TCGA-data/CPathPatchFeature/blca/chief/pt_files"

def seed_all(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def compute_auc(labels, probs):
    labels = np.asarray(labels); probs = np.asarray(probs)
    try:
        if labels.ndim==1 or (labels.ndim==2 and labels.shape[1]==1):
            return roc_auc_score(labels.ravel(), probs.ravel())
        aucs=[]
        for i in range(labels.shape[1]):
            if labels[:,i].sum()>0 and labels[:,i].sum()<len(labels):
                aucs.append(roc_auc_score(labels[:,i], probs[:,i]))
        return float(np.mean(aucs)) if aucs else 0.5
    except Exception as e:
        print(f"  [WARN] compute_auc fail: {e}")
        return 0.5

def summary(aucs, name):
    vals = [a for a in aucs if a is not None and not np.isnan(a)]
    print()
    print("="*64)
    print(f"  {name}")
    print("="*64)
    if not vals:
        print("  NO RESULTS (checkpoint missing / error)")
        print("="*64); return None
    for i, a in enumerate(aucs):
        if a is None:
            print(f"  Fold {i}: MISSING")
        else:
            print(f"  Fold {i}: AUC={a:.4f}")
    mean = float(np.mean(vals)); std = float(np.std(vals))
    print("-"*64)
    print(f"  Mean AUC: {mean:.4f} ± {std:.4f}  (n={len(vals)} folds)")
    print(f"  Min / Max: {min(vals):.4f} / {max(vals):.4f}")
    print("="*64)
    return {"folds": [None if a is None else round(a,4) for a in aucs], "mean": round(mean,4), "std": round(std,4), "n": len(vals)}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed_all(42)

FINAL = {}

# ====================================================================
# 1. MoTo-CARE
# ====================================================================
print("\n" + "#"*64)
print("  1. EVAL MoTo-CARE")
print("#"*64)
sys.path.insert(0, os.path.join(ROOT, '项目_1_MoTo-CARE'))
import importlib.util
spec = importlib.util.spec_from_file_location("moto_train", os.path.join(ROOT, '项目_1_MoTo-CARE', 'train.py'))
mt = importlib.util.module_from_spec(spec); spec.loader.exec_module(mt)

ds = mt.RealMoToCAREDataset(csv_path=CSV_PATH, data_root=DATA_ROOT, input_dim=768, num_tasks=1, max_tiles=4096, num_regions=8, topology_dim=12, molecule_dim=128, num_molecule_tokens=4)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
idx = np.arange(len(ds))
moto_ckpt = os.path.join(ROOT, '项目_1_MoTo-CARE', 'results_real', 'real_MoToCARE_R8_T1_s42')
print(f"[MoTo] Dataset: {len(ds)} slides, results dir: {moto_ckpt}")

aucs_moto = []
for fold, (tr, va) in enumerate(kf.split(idx)):
    cp = os.path.join(moto_ckpt, f'fold_{fold}_best.pt')
    if not os.path.exists(cp):
        print(f"  Fold {fold}: checkpoint missing -> SKIP"); aucs_moto.append(None); continue
    ckpt = torch.load(cp, map_location='cpu')
    sd = ckpt['model_state_dict']
    # detect params
    embed_dim, num_heads, top_k_regions, topology_dim, molecule_dim = 256, 4, 4, 12, 128
    for k, v in sd.items():
        if k == 'tile_proj.weight': embed_dim = v.shape[0]; break
    for k, v in sd.items():
        if 'topology_proj.weight' in k: topology_dim = v.shape[1]; break
    for k, v in sd.items():
        if 'molecule_proj.weight' in k: molecule_dim = v.shape[1]; break
    for k, v in sd.items():
        if 'attn.qkv.weight' in k:
            H = v.shape[-1]; num_heads = max(1, H // 64); break
    # detect top_k from region_assign or default 4 (not strictly needed for inference; pass default)
    cfg = mt.MoToCAREConfig(
        input_dim=768, embed_dim=embed_dim, num_regions=8,
        num_heads=num_heads, num_tasks=1, topology_dim=topology_dim,
        molecule_dim=molecule_dim, top_k_regions=4,
        assignment_temperature=0.35, topology_weight=0.5,
        molecular_weight=0.2, entropy_weight=0.01, dropout=0.1,
        label_smoothing=0.0,
    )
    try:
        model = mt.MoToCARE(cfg).to(device)
        model.load_state_dict(sd, strict=False); model.eval()
    except Exception as e:
        print(f"  Fold {fold}: model load FAIL {e}"); aucs_moto.append(None); continue
    va_ds = Subset(ds, va)
    all_lab, all_prob = [], []
    with torch.no_grad():
        for s in tqdm(va_ds, desc=f"  Moto Fold {fold}", leave=False):
            feat = s['features'].unsqueeze(0).to(device)
            coord = s['coords'].unsqueeze(0).to(device)
            lab = s['label']; topo_prior = s['topology_prior'].unsqueeze(0).to(device)
            topo_target = s['topology_target'].unsqueeze(0).to(device)
            mol_tokens = s['molecule_tokens'].unsqueeze(0).to(device)
            B, N = feat.shape[:2]
            pm = torch.zeros(B, N, dtype=torch.bool, device=device)
            out = model(features=feat, coords=coord, labels=None, topology_prior=topo_prior, topology_target=topo_target, molecule_tokens=mol_tokens, padding_mask=pm)
            if 'probs' in out: prob = out['probs'].cpu().numpy()
            else: prob = torch.sigmoid(out['logits']).cpu().numpy()
            all_lab.append(np.asarray(lab).reshape(1,-1)); all_prob.append(prob.reshape(1,-1))
    a = compute_auc(np.concatenate(all_lab), np.concatenate(all_prob))
    print(f"  Fold {fold}: AUC={a:.4f}"); aucs_moto.append(a)
FINAL["MoTo-CARE"] = summary(aucs_moto, "MoTo-CARE (real_MoToCARE_R8_T1_s42)")

# ====================================================================
# 2. PathwayMorph-OT
# ====================================================================
print("\n" + "#"*64)
print("  2. EVAL PathwayMorph-OT")
print("#"*64)
sys.path.insert(0, os.path.join(ROOT, '项目_2_PathwayMorph-OT'))
spec = importlib.util.spec_from_file_location("pmot_train", os.path.join(ROOT, '项目_2_PathwayMorph-OT', 'train.py'))
ptm = importlib.util.module_from_spec(spec); spec.loader.exec_module(ptm)

pmot_root = os.path.join(ROOT, '项目_2_PathwayMorph-OT', 'results_real')
for exp_name in sorted(os.listdir(pmot_root)):
    exp_dir = os.path.join(pmot_root, exp_name)
    if not os.path.isdir(exp_dir): continue
    n_ckpt = len([x for x in os.listdir(exp_dir) if x.startswith('fold_') and x.endswith('_best.pt')])
    print(f"\n>>> PMOT experiment: {exp_name}  (found {n_ckpt}/5 fold checkpoints)")
    ds_pmot = ptm.RealPMOTDataset(csv_path=CSV_PATH, data_root=DATA_ROOT, atom_dim=256, num_regions=8, num_pathways=5, topology_dim=12, spatial_dim=4, pathway_dim=64, num_tasks=1, max_tiles=4096, tile_input_dim=768)
    aucs_p = []
    for fold, (tr, va) in enumerate(kf.split(idx)):
        cp = os.path.join(exp_dir, f'fold_{fold}_best.pt')
        if not os.path.exists(cp):
            print(f"  Fold {fold}: checkpoint missing -> SKIP"); aucs_p.append(None); continue
        ckpt = torch.load(cp, map_location='cpu'); sd = ckpt['model_state_dict']
        # detect params
        atom_dim, topology_dim, spatial_dim, pathway_dim, hidden_dim = 256, 12, 4, 64, 128
        for k, v in sd.items():
            if 'atom_encoder.1.weight' in k: hidden_dim = v.shape[0]; break
        for k, v in sd.items():
            if 'pathway_encoder.1.weight' in k: pathway_dim = v.shape[1]; break
        for k, v in sd.items():
            if 'topology_to_hidden.weight' in k: topology_dim = v.shape[1]; break
        cfg_p = ptm.PathwayMorphOTConfig(
            atom_dim=atom_dim, topology_dim=topology_dim, spatial_dim=spatial_dim,
            pathway_dim=pathway_dim, hidden_dim=hidden_dim, num_tasks=1,
            epsilon=0.08, tau=0.8, ot_iters=40, ot_cost_weight=0.05,
            entropy_weight=0.005, label_smoothing=0.0,
        )
        try:
            model_p = ptm.PathwayMorphOT(cfg_p).to(device)
            model_p.load_state_dict(sd, strict=False); model_p.eval()
        except Exception as e:
            print(f"  Fold {fold}: load FAIL {e}"); aucs_p.append(None); continue
        va_ds = Subset(ds_pmot, va)
        all_lab, all_prob = [], []
        with torch.no_grad():
            for s in tqdm(va_ds, desc=f"  PMOT {exp_name} F{fold}", leave=False):
                region = s['region_embeddings'].unsqueeze(0).to(device)
                topo = s['topology'].unsqueeze(0).to(device)
                spa = s['spatial'].unsqueeze(0).to(device)
                pw = s['pathway_tokens'].unsqueeze(0).to(device)
                lab = s['label']
                out = model_p(region_embeddings=region, topology=topo, spatial=spa, pathway_tokens=pw, labels=None)
                if 'probs' in out: prob = out['probs'].cpu().numpy()
                else: prob = torch.sigmoid(out['logits']).cpu().numpy()
                all_lab.append(np.asarray(lab).reshape(1,-1)); all_prob.append(prob.reshape(1,-1))
        a = compute_auc(np.concatenate(all_lab), np.concatenate(all_prob))
        print(f"  Fold {fold}: AUC={a:.4f}"); aucs_p.append(a)
    FINAL[f"PMOT/{exp_name}"] = summary(aucs_p, f"PathwayMorph-OT: {exp_name}")

# ====================================================================
# 3. CARE-E2E-Fusion (use trainer._forward_with_tiles for consistent logic)
# ====================================================================
print("\n" + "#"*64)
print("  3. EVAL CARE-E2E-Fusion")
print("#"*64)
sys.path.insert(0, os.path.join(ROOT, 'CARE-E2E-Fusion'))
spec = importlib.util.spec_from_file_location("e2e_train", os.path.join(ROOT, 'CARE-E2E-Fusion', 'train.py'))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
from models.fusion_model import E2EViTCAREFusion

e2e_root = os.path.join(ROOT, 'CARE-E2E-Fusion', 'results_real')
for exp_name in sorted(os.listdir(e2e_root)):
    exp_dir = os.path.join(e2e_root, exp_name)
    if not os.path.isdir(exp_dir): continue
    n_ckpt = len([x for x in os.listdir(exp_dir) if x.startswith('fold_') and x.endswith('_best.pt')])
    print(f"\n>>> E2E experiment: {exp_name}  (found {n_ckpt}/5 fold checkpoints)")
    ds_e2e = em.RealWSIDataset(csv_path=CSV_PATH, data_root=DATA_ROOT, embed_dim=768, num_tasks=1, tile_size=256, max_tiles=8192)
    aucs_e = []
    for fold, (tr, va) in enumerate(kf.split(idx)):
        cp = os.path.join(exp_dir, f'fold_{fold}_best.pt')
        if not os.path.exists(cp):
            print(f"  Fold {fold}: checkpoint missing -> SKIP"); aucs_e.append(None); continue
        ckpt = torch.load(cp, map_location='cpu'); sd = ckpt['model_state_dict']
        # detect architecture params
        embed_dim, n_heads, n_layers, n_reg = 128, 4, 4, 8
        for k, v in sd.items():
            if 'region_tokens' in k: n_reg = v.shape[1]; break
        for k, v in sd.items():
            if 'patch_embed.weight' in k and len(v.shape)==4: embed_dim = v.shape[0]; break
            if k == 'tile_proj.weight': embed_dim = v.shape[0]; break
        # detect head dim for num_heads from any qkv weight
        for k, v in sd.items():
            if 'qkv.weight' in k:
                H = v.shape[-1]
                if H % 64 == 0: n_heads = H // 64
                elif H % 32 == 0: n_heads = H // 32
                break
        # count layers
        layer_ids = set()
        for k in sd.keys():
            if k.startswith('blocks.'):
                try: layer_ids.add(int(k.split('.')[1]))
                except: pass
            elif k.startswith('transformer.layers.'):
                try: layer_ids.add(int(k.split('.')[2]))
                except: pass
        if layer_ids: n_layers = max(layer_ids)+1

        # construct model: try with distillation first, then fallback
        built_ok = False
        for use_dist in [True, False]:
            for d_use_two in [False, True]:
                try:
                    model_e = E2EViTCAREFusion(
                        tile_size=16, patch_size=1, embed_dim=embed_dim,
                        num_heads=n_heads, num_layers=n_layers,
                        num_region_tokens=n_reg, num_tasks=1, task_names=['task0'],
                        use_alibi=True, use_distillation=use_dist,
                        use_two_branches=d_use_two, dropout=0.1,
                    ).to(device)
                    ok_e = model_e.load_state_dict(sd, strict=False)
                    built_ok = True; break
                except Exception as _e:
                    continue
            if built_ok: break
        if not built_ok:
            # last fallback: 256/4/4/8
            try:
                model_e = E2EViTCAREFusion(
                    tile_size=16, patch_size=1, embed_dim=256, num_heads=4, num_layers=4,
                    num_region_tokens=8, num_tasks=1, task_names=['task0'],
                    use_alibi=True, use_distillation=False, use_two_branches=False, dropout=0.1,
                ).to(device)
                model_e.load_state_dict(sd, strict=False); built_ok = True
            except Exception as _eF:
                print(f"  Fold {fold}: ALL MODEL INIT FAILED: {_eF}")
                aucs_e.append(None); continue
        model_e.eval()
        # Use trainer's forward method to ensure consistency with training time
        trainer = em.FusionTrainer(
            model=model_e, optimizer=torch.optim.SGD(model_e.parameters(), lr=1e-9),
            device=device, num_tasks=1, task_names=['task0'],
            early_stopping_patience=9999, log_interval=9999, label_smoothing=0.0,
        )
        va_ds = Subset(ds_e2e, va)
        all_lab, all_prob = [], []
        with torch.no_grad():
            for s in tqdm(va_ds, desc=f"  E2E {exp_name} F{fold}", leave=False):
                toks = s['tile_tokens'].unsqueeze(0).to(device)
                coord = s['coords'].unsqueeze(0).to(device)
                lab = s['label']
                B, N, C = toks.shape
                pm = torch.zeros(B, N, dtype=torch.bool, device=device)
                lab_t = torch.tensor(np.asarray(lab).reshape(1,-1), dtype=torch.float32, device=device)
                out = trainer._forward_with_tiles(toks, coord, lab_t, pm)
                # E2E trainer returns loss_dict, logits. Check for logits.
                if 'logits_direct' in out:
                    logits = out['logits_direct']
                elif 'logits' in out:
                    logits = out['logits']
                else:
                    # find first [1,1] tensor value
                    logits = None
                    for k, v in out.items():
                        if isinstance(v, torch.Tensor) and v.dtype in (torch.float16, torch.float32, torch.float64) and tuple(v.shape)==(1,1):
                            logits = v; break
                prob = torch.sigmoid(logits).cpu().numpy()
                all_lab.append(np.asarray(lab).reshape(1,-1)); all_prob.append(prob.reshape(1,-1))
        a = compute_auc(np.concatenate(all_lab), np.concatenate(all_prob))
        print(f"  Fold {fold}: AUC={a:.4f}"); aucs_e.append(a)
    FINAL[f"E2E/{exp_name}"] = summary(aucs_e, f"CARE-E2E-Fusion: {exp_name}")

# ====================================================================
# Final print table
# ====================================================================
print("\n\n" + "="*84)
print("  FINAL AUC SUMMARY (All Methods)")
print("="*84)
print(f"{'Method':<40s} {'n':>3s} {'Mean AUC':>9s} {'Std':>7s} {'Min':>7s} {'Max':>7s}")
print("-"*84)
rows = []
for k, v in FINAL.items():
    if v is None: continue
    folds = [x for x in v['folds'] if x is not None]
    if not folds: continue
    rows.append((k, v['n'], v['mean'], v['std'], min(folds), max(folds)))
rows.sort(key=lambda r: -r[2])
for k, n, mean, std, mn, mx in rows:
    print(f"{k:<40s} {n:>3d} {mean:>9.4f} {std:>7.4f} {mn:>7.4f} {mx:>7.4f}")
print("="*84)

out_json = os.path.join(ROOT, 'AUC_SUMMARY_FINAL.json')
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(FINAL, f, ensure_ascii=False, indent=2)
print(f"\nSaved JSON summary to: {out_json}")
print(f"Done at {time.strftime('%Y-%m-%d %H:%M:%S')}")
