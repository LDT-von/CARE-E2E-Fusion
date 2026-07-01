# -*- coding: utf-8 -*-
"""
Evaluate PathwayMorph-OT checkpoints
"""
from __future__ import print_function
import sys, os, glob, re
sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_2_PathwayMorph-OT')

import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

from train import RealPMOTDataset, dummy_collate_fn
from pathway_morph_ot import PathwayMorphOT, PathwayMorphOTConfig

device = torch.device('cuda:0')

dataset = RealPMOTDataset(
    csv_path=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv',
    data_root=r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files',
    tile_input_dim=768, num_tasks=1, num_regions=8,
)
print('Dataset:', len(dataset))

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(dataset))))

result_dirs = [
    r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_2_PathwayMorph-OT\results_real',
]

all_results = {}

for results_root in result_dirs:
    if not os.path.exists(results_root):
        print('Dir not found:', results_root)
        continue

    for subdir in sorted(os.listdir(results_root)):
        result_dir = os.path.join(results_root, subdir)
        if not os.path.isdir(result_dir):
            continue
        best_files = sorted(glob.glob(os.path.join(result_dir, 'fold_*_best.pt')))
        if not best_files:
            print('\n=== %s ===\n  No checkpoints' % subdir)
            continue

        print('\n=== %s ===' % subdir)
        cfg = PathwayMorphOTConfig()
        fold_aucs = []

        for bf in best_files:
            fold = int(re.search(r'fold_(\d)_', bf).group(1))
            ckpt = torch.load(bf, map_location='cpu')
            ep = ckpt.get('epoch', '?')
            print('\nFold %d: epoch=%s' % (fold, ep), flush=True)

            model = PathwayMorphOT(cfg).to(device)
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()

            _, val_idx = splits[fold]
            val_subset = Subset(dataset, val_idx.tolist())
            loader = DataLoader(val_subset, batch_size=2, shuffle=False,
                               collate_fn=dummy_collate_fn, num_workers=0)

            all_labels, all_probs = [], []
            with torch.no_grad():
                for batch in tqdm(loader, desc='Fold %d' % fold, leave=False):
                    features, topo_prior, spatial_info, pathway_tokens, labels, slide_ids = batch
                    features = features.to(device)
                    topo_prior = topo_prior.to(device)
                    spatial_info = spatial_info.to(device)
                    pathway_tokens = pathway_tokens.to(device)
                    labels = labels.to(device).float()

                    out = model(
                        region_embeddings=features,
                        topology=topo_prior,
                        spatial=spatial_info,
                        pathway_tokens=pathway_tokens,
                        labels=labels,
                    )

                    all_labels.append(labels.cpu().numpy())
                    all_probs.append(out['probs'].cpu().numpy())

            all_labels = np.concatenate(all_labels)
            all_probs = np.concatenate(all_probs)
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(all_labels.ravel(), all_probs.ravel())
            except:
                auc = 0.5
            fold_aucs.append((fold, ep, auc))
            print('  AUC: %.4f' % auc, flush=True)

        if fold_aucs:
            mean_auc = np.mean([x[2] for x in fold_aucs])
            std_auc = np.std([x[2] for x in fold_aucs])
            all_results[subdir] = (fold_aucs, mean_auc, std_auc)
            print('\n  Mean AUC: %.4f +/- %.4f' % (mean_auc, std_auc))

print('\n' + '=' * 60)
print('Summary of All PathwayMorph-OT Results')
print('=' * 60)
for name, (folds, mean_auc, std_auc) in sorted(all_results.items()):
    print('\n%s' % name)
    for fold, ep, auc in folds:
        print('  Fold %d: AUC=%.4f' % (fold, auc))
    print('  Mean: %.4f +/- %.4f' % (mean_auc, std_auc))
