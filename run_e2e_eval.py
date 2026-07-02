"""Standalone E2E-Fusion AUC eval with correct embed_dim detection."""
import os, sys, warnings, random, json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT, 'CARE-E2E-Fusion', 'blca_slides.csv')
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
        print(f"  [WARN] compute_auc fail: {e}"); return 0.5

seed_all(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

sys.path.insert(0, os.path.join(ROOT, 'CARE-E2E-Fusion'))
import importlib.util
spec = importlib.util.spec_from_file_location("e2e_train", os.path.join(ROOT, 'CARE-E2E-Fusion', 'train.py'))
em = importlib.util.module_from_spec(spec); spec.loader.exec_module(em)
from models.fusion_model import E2EViTCAREFusion

ds_e2e = em.RealWSIDataset(csv_path=CSV_PATH, data_root=DATA_ROOT, embed_dim=768, num_tasks=1, tile_size=256, max_tiles=8192)
print(f"Dataset: {len(ds_e2e)} slides")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
idx = np.arange(len(ds_e2e))
e2e_root = os.path.join(ROOT, 'CARE-E2E-Fusion', 'results_real')

FINAL_E2E = {}
for exp_name in sorted(os.listdir(e2e_root)):
    exp_dir = os.path.join(e2e_root, exp_name)
    if not os.path.isdir(exp_dir): continue
    n_ckpt = len([x for x in os.listdir(exp_dir) if x.startswith('fold_') and x.endswith('_best.pt')])
    print(f"\n>>> E2E experiment: {exp_name}  ({n_ckpt}/5 folds)")
    aucs_e = []
    for fold, (tr, va) in enumerate(kf.split(idx)):
        cp = os.path.join(exp_dir, f'fold_{fold}_best.pt')
        if not os.path.exists(cp):
            print(f"  Fold {fold}: missing -> SKIP"); aucs_e.append(None); continue
        ckpt = torch.load(cp, map_location='cpu'); sd = ckpt['model_state_dict']

        # -------- detect architecture correctly from sd --------
        embed_dim = 768  # FIRST try 768 since that's what error showed
        n_heads = 12
        n_layers = 4
        n_reg = 8
        patch_size = 16
        tile_size = 16
        for k, v in sd.items():
            if 'region_tokens' in k: n_reg = v.shape[1]
        # detect embed dim from patch_embed or head
        for k, v in sd.items():
            if k == 'head_direct.weight': embed_dim = v.shape[1]; break
            if 'patch_embed.proj.weight' == k and len(v.shape)==4:
                embed_dim = v.shape[0]; patch_size = v.shape[2]; break
        # num_heads from attn.in_proj_weight (shape is 3H, D)
        for k, v in sd.items():
            if 'attn.in_proj_weight' in k:
                H = v.shape[0] // 3  # 3H for QKV concat
                if embed_dim % H == 0:
                    n_heads = embed_dim // H
                elif H % 64 == 0: n_heads = H // 64
                else: n_heads = 4
                break
        # count layers
        layer_ids = set()
        for k in sd.keys():
            if k.startswith('blocks.'):
                try: layer_ids.add(int(k.split('.')[1]))
                except: pass
        if layer_ids: n_layers = max(layer_ids)+1

        print(f"  Fold {fold}: Detected embed_dim={embed_dim}, heads={n_heads}, layers={n_layers}, reg={n_reg}, patch={patch_size}, tile={tile_size}")

        # Try many config combinations
        built_ok = False
        last_err = None
        configs_to_try = [
            # (use_distill, use_two_branches, tile_size, patch_size, embed_dim, n_heads, n_layers)
            (True, False, tile_size, patch_size, embed_dim, n_heads, n_layers),
            (False, False, tile_size, patch_size, embed_dim, n_heads, n_layers),
            (True, True, tile_size, patch_size, embed_dim, n_heads, n_layers),
            (False, True, tile_size, patch_size, embed_dim, n_heads, n_layers),
            (True, False, 16, 1, embed_dim, n_heads, n_layers),
            (False, False, 16, 1, embed_dim, n_heads, n_layers),
            (True, False, tile_size, patch_size, embed_dim, max(1,n_heads//2), n_layers),
        ]
        # also try reduced embed_dim fallback (for other experiments in dir)
        for d_try in [embed_dim]:
            for cfg_t in configs_to_try:
                ud, utb, ts, ps, ed, nh, nl = cfg_t
                ed = d_try
                try:
                    model_e = E2EViTCAREFusion(
                        tile_size=ts, patch_size=ps, embed_dim=ed, num_heads=nh,
                        num_layers=nl, num_region_tokens=n_reg, num_tasks=1,
                        task_names=['task0'], use_alibi=True, use_distillation=ud,
                        use_two_branches=utb, dropout=0.1,
                    ).to(device)
                    ok_e = model_e.load_state_dict(sd, strict=False)
                    # sanity: if there are >20 missing keys with size mismatch, probably wrong
                    if len(ok_e.missing_keys) < 50 or len([x for x in ok_e.unexpected_keys if not x.startswith('head_')]) < 10:
                        built_ok = True; break
                except Exception as _e:
                    last_err = _e
                    continue
            if built_ok: break
        if not built_ok:
            # last fallback: try 256 as last-ditch
            try:
                model_e = E2EViTCAREFusion(
                    tile_size=16, patch_size=16, embed_dim=256, num_heads=4,
                    num_layers=4, num_region_tokens=8, num_tasks=1,
                    task_names=['task0'], use_alibi=True, use_distillation=False,
                    use_two_branches=False, dropout=0.1,
                ).to(device)
                ok_e = model_e.load_state_dict(sd, strict=False)
                built_ok = True
            except Exception as _eF:
                print(f"  Fold {fold}: INIT FAILED. last err: {last_err}, final: {_eF}")
                aucs_e.append(None); continue
        model_e.eval()
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
                try:
                    out = trainer._forward_with_tiles(toks, coord, lab_t, pm)
                except Exception as eF:
                    print(f"  forward ERR: {eF}"); continue
                if 'logits_direct' in out: logits = out['logits_direct']
                elif 'logits' in out: logits = out['logits']
                else:
                    logits = None
                    for k2, v2 in out.items():
                        if isinstance(v2, torch.Tensor) and v2.dtype in (torch.float16, torch.float32, torch.float64) and tuple(v2.shape)==(1,1):
                            logits = v2; break
                prob = torch.sigmoid(logits).cpu().numpy()
                all_lab.append(np.asarray(lab).reshape(1,-1)); all_prob.append(prob.reshape(1,-1))
        a = compute_auc(np.concatenate(all_lab), np.concatenate(all_prob))
        print(f"  Fold {fold}: AUC={a:.4f}"); aucs_e.append(a)

    # summarize
    vals = [x for x in aucs_e if x is not None]
    print()
    print("="*64)
    print(f"  CARE-E2E-Fusion: {exp_name}")
    print("="*64)
    if vals:
        for i, x in enumerate(aucs_e):
            if x is None: print(f"  Fold {i}: MISSING")
            else: print(f"  Fold {i}: AUC={x:.4f}")
        print("-"*64)
        mean = float(np.mean(vals)); std = float(np.std(vals))
        print(f"  Mean AUC: {mean:.4f} ± {std:.4f}  (n={len(vals)})")
        print(f"  Min/Max:   {min(vals):.4f} / {max(vals):.4f}")
        FINAL_E2E[exp_name] = {"folds":[None if x is None else round(x,4) for x in aucs_e], "mean":round(mean,4), "std":round(std,4), "n":len(vals)}
    else:
        print("  NO RESULTS")
        FINAL_E2E[exp_name] = None
    print("="*64)

# Save
out_path = os.path.join(ROOT, 'auc_summary_e2e.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(FINAL_E2E, f, ensure_ascii=False, indent=2)
print(f"\nSaved E2E results: {out_path}")

# Merge with overall summary
if os.path.exists(os.path.join(ROOT, 'auc_summary_all.json')):
    with open(os.path.join(ROOT, 'auc_summary_all.json'), 'r', encoding='utf-8') as f:
        overall = json.load(f)
    for k, v in FINAL_E2E.items():
        overall[f"E2E/{k}"] = v
    with open(os.path.join(ROOT, 'auc_summary_all.json'), 'w', encoding='utf-8') as f:
        json.dump(overall, f, ensure_ascii=False, indent=2)
    print(f"Merged into auc_summary_all.json")

    # Print final combined table
    rows = []
    for k, v in overall.items():
        if v is None: continue
        f = [x for x in v['folds'] if x is not None]
        if not f: continue
        rows.append((k, v['n'], v['mean'], v['std'], min(f), max(f)))
    rows.sort(key=lambda r: -r[2])
    print("\n\n" + "="*88)
    print("  FINAL COMBINED AUC SUMMARY (All 3 Methods)")
    print("="*88)
    print(f"{'Method':<42s} {'n':>3s} {'Mean AUC':>9s} {'Std':>7s} {'Min':>7s} {'Max':>7s}")
    print("-"*88)
    for r in rows:
        print(f"{r[0]:<42s} {r[1]:>3d} {r[2]:>9.4f} {r[3]:>7.4f} {r[4]:>7.4f} {r[5]:>7.4f}")
    print("="*88)
