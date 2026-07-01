import sys, os, re, glob
sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE')
import torch, numpy as np
from torch.utils.data import DataLoader, Subset
from train import RealMoToCAREDataset, dummy_collate_fn, compute_multitask_auc
from moto_care import MoToCARE, MoToCAREConfig
from sklearn.model_selection import KFold

device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = False

print('Loading dataset...', flush=True)
dataset = RealMoToCAREDataset(
    csv_path=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv',
    data_root=r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files',
    input_dim=768, num_tasks=1, num_regions=8
)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(dataset))))

result_dir = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\results_real\real_MoToCARE_R8_T1_s42'
best_files = sorted(glob.glob(os.path.join(result_dir, 'fold_*_best.pt')))

cfg = MoToCAREConfig(input_dim=768, embed_dim=256, num_regions=8, num_heads=4, num_tasks=1)

aucs = []
for bf in best_files:
    fold = int(re.search(r'fold_(\d)_', bf).group(1))
    print(f'Fold {fold}: loading...', flush=True)
    ckpt = torch.load(bf, map_location='cpu')
    ep = ckpt.get('epoch', '?')

    model = MoToCARE(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    _, val_idx = splits[fold]
    val_subset = Subset(dataset, val_idx.tolist())
    loader = DataLoader(val_subset, batch_size=2, shuffle=False, collate_fn=dummy_collate_fn, num_workers=0)

    all_labels, all_probs = [], []
    total = len(loader)
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            features, coords, labels, topo_prior, topo_target, mol_tokens, _, padding_mask = batch
            features = features.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
            topo_prior = topo_prior.to(device)
            topo_target = topo_target.to(device)
            mol_tokens = mol_tokens.to(device)
            padding_mask = padding_mask.to(device)

            out = model(features=features, coords=coords, labels=labels,
                        topology_prior=topo_prior, topology_target=topo_target,
                        molecule_tokens=mol_tokens, padding_mask=padding_mask)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(out['probs'].cpu().numpy())
            if bi % 10 == 0:
                print(f'  [{bi}/{total}]', flush=True)

    auc = compute_multitask_auc(np.concatenate(all_labels), np.concatenate(all_probs))
    aucs.append((fold, ep, auc))
    print(f'  => Fold {fold} epoch={ep} AUC={auc:.4f}', flush=True)

print(f'\n=== MoTo-CARE ===', flush=True)
for fold, ep, auc in aucs:
    print(f'Fold {fold}: AUC={auc:.4f} (epoch={ep})', flush=True)
mean_auc = np.mean([a for _,_,a in aucs])
std_auc = np.std([a for _,_,a in aucs])
print(f'Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f} ({len(aucs)} folds)', flush=True)
