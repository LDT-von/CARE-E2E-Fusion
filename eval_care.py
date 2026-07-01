"""快速评估已有checkpoint的AUC - CARE-E2E-Fusion"""
import sys, os, glob, re
sys.path.insert(0, r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion")
import torch, numpy as np
from torch.utils.data import DataLoader
from train import RealWSIDataset, dummy_collate_fn, compute_multitask_auc
from models.fusion_model import E2EViTCAREFusion
from tqdm import tqdm

device = torch.device("cuda:0")

# 加载数据集
csv_path = r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv"
data_root = r"E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files"
dataset = RealWSIDataset(csv_path=csv_path, data_root=data_root, embed_dim=768, num_tasks=1)
print(f"Dataset: {len(dataset)} slides")

# 5折索引（同seed=42）
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
indices = np.arange(len(dataset))
splits = list(kfold.split(indices))

result_dir = r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42"
best_files = sorted(glob.glob(os.path.join(result_dir, "fold_*_best.pt")))

results = []
for bf in best_files:
    fold = int(re.search(r"fold_(\d)_", bf).group(1))
    ckpt = torch.load(bf, map_location="cpu")
    print(f"\nFold {fold}: loaded epoch={ckpt.get('epoch','?')}, best_score={ckpt.get('best_score','?')}")

    # 创建模型
    model = E2EViTCAREFusion(
        tile_size=256, patch_size=16, embed_dim=768,
        num_heads=4, num_layers=4, num_region_tokens=8,
        num_tasks=1, dropout=0.25,
        use_two_branches=True, use_distillation=False, use_alibi=True,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # 验证集
    _, val_idx = splits[fold]
    from torch.utils.data import Subset
    val_subset = Subset(dataset, val_idx.tolist())
    loader = DataLoader(val_subset, batch_size=2, shuffle=False, collate_fn=dummy_collate_fn, num_workers=0)

    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Fold {fold} val"):
            pad_tokens, pad_coords, labels, _, padding_mask = batch
            pad_tokens, pad_coords = pad_tokens.to(device), pad_coords.to(device)
            labels = labels.to(device).float()
            padding_mask = padding_mask.to(device)

            # 直接分类分支
            x = pad_tokens
            x_global = x.mean(dim=1)
            logits_direct = model.head_direct(x_global)

            # 自适应分支
            region_features, _, _ = model.dynamic_region_partition(tile_tokens=x, coords=pad_coords, return_coverage=True)
            region_embeddings, region_pooled = model.arm(region_features=region_features, tile_tokens=x, attn_weights=None)
            # region_pooled should be [B, hidden_dim], need to get it from head_adaptive
            adaptive_out = model.head_adaptive(region_pooled)

            all_labels.append(labels.cpu().numpy())
            all_probs.append(torch.sigmoid(logits_direct).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    auc = compute_multitask_auc(all_labels, all_probs)
    results.append(auc)
    print(f"  AUC: {auc:.4f}")

print(f"\n=== Final ===")
for i, auc in enumerate(results):
    print(f"Fold {i}: AUC={auc:.4f}")
print(f"Mean AUC: {np.mean(results):.4f} +/- {np.std(results):.4f}")
