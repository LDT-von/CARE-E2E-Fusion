"""
Evaluate MoTo-CARE and PathwayMorph-OT checkpoints
"""
import sys, os, glob, re
import numpy as np
import torch
from tqdm import tqdm
from sklearn.model_selection import KFold

sys.path.insert(0, r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE")

# ============================================================
# MoTo-CARE evaluation
# ============================================================
from train import RealMoToCAREDataset, dummy_collate_fn, compute_multitask_auc
from moto_care import MoToCARE, MoToCAREConfig

device = torch.device("cuda:0")
torch.backends.cudnn.benchmark = False

print("=" * 60)
print("MoTo-CARE Evaluation")
print("=" * 60)

dataset = RealMoToCAREDataset(
    csv_path=r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv",
    data_root=r"E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files",
    input_dim=768, num_tasks=1, num_regions=8
)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(dataset))))

moto_dirs = [
    r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\results_real",
]

for results_root in moto_dirs:
    if not os.path.exists(results_root):
        continue
    for subdir in sorted(os.listdir(results_root)):
        result_dir = os.path.join(results_root, subdir)
        if not os.path.isdir(result_dir):
            continue
        best_files = sorted(glob.glob(os.path.join(result_dir, "fold_*_best.pt")))
        if not best_files:
            print(f"\n=== {subdir} ===\n  No checkpoints found")
            continue

        print(f"\n=== {subdir} ===")
        cfg = MoToCAREConfig(input_dim=768, embed_dim=256, num_regions=8, num_heads=4, num_tasks=1)
        aucs = []

        for bf in best_files:
            fold = int(re.search(r"fold_(\d)_", bf).group(1))
            ckpt = torch.load(bf, map_location="cpu")
            ep = ckpt.get("epoch", "?")
            print(f"\nFold {fold}: epoch={ep}", flush=True)

            model = MoToCARE(cfg).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            _, val_idx = splits[fold]
            val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())
            loader = torch.utils.data.DataLoader(
                val_subset, batch_size=2, shuffle=False,
                collate_fn=dummy_collate_fn, num_workers=0
            )

            all_labels, all_probs = [], []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Fold {fold}", leave=False):
                    features, coords, labels, topo_prior, topo_target, mol_tokens, _, padding_mask = batch
                    features = features.to(device)
                    coords = coords.to(device)
                    labels = labels.to(device).float()
                    topo_prior = topo_prior.to(device)
                    topo_target = topo_target.to(device)
                    mol_tokens = mol_tokens.to(device)
                    padding_mask = padding_mask.to(device)

                    out = model(
                        features=features, coords=coords, labels=labels,
                        topology_prior=topo_prior, topology_target=topo_target,
                        molecule_tokens=mol_tokens, padding_mask=padding_mask
                    )
                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(out["probs"].cpu().numpy())

            auc = compute_multitask_auc(
                np.concatenate(all_labels),
                np.concatenate(all_probs)
            )
            aucs.append((fold, ep, auc))
            print(f"  AUC: {auc:.4f}", flush=True)

        if aucs:
            mean_auc = np.mean([a for _, _, a in aucs])
            std_auc = np.std([a for _, _, a in aucs])
            print(f"\n  Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

print("\n\n" + "=" * 60)
print("Done!")
print("=" * 60)
