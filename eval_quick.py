import sys, os, glob, re
sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')
import torch, numpy as np
from torch.utils.data import DataLoader, Subset
from train import RealWSIDataset, dummy_collate_fn, compute_multitask_auc
from models.fusion_model import E2EViTCAREFusion
from sklearn.model_selection import KFold
from tqdm import tqdm

device = torch.device('cuda:0')

print('Loading dataset...')
dataset = RealWSIDataset(
    csv_path=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv',
    data_root=r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files',
    embed_dim=768, num_tasks=1
)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
splits = list(kfold.split(range(len(dataset))))
print(f'{len(dataset)} slides, 5 folds')

result_dir = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42'
best_files = sorted(glob.glob(os.path.join(result_dir, 'fold_*_best.pt')))

for bf in best_files:
    fold = int(re.search(r'fold_(\d)_', bf).group(1))
    ckpt = torch.load(bf, map_location='cpu')
    ep = ckpt.get('epoch', '?')
    print(f'\nFold {fold} (epoch={ep}): loading model...')

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
    loader = DataLoader(val_subset, batch_size=2, shuffle=False, collate_fn=dummy_collate_fn, num_workers=0)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for pad_tokens, pad_coords, labels, _, padding_mask in tqdm(loader, desc='Val'):
            x = pad_tokens.to(device)
            labels = labels.to(device).float()
            logits = model.head_direct(x.mean(dim=1))
            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    auc = compute_multitask_auc(np.concatenate(all_labels), np.concatenate(all_probs))
    print(f' => AUC: {auc:.4f}')
