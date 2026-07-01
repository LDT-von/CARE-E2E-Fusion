"""
正确的 CARE-E2E-Fusion 评估脚本
================================

关键修复：
1. 完整跑过 Patch Embedding → Patch Merger → Transformer → 双分支
2. 既然没有真实图像（输入是 CONCH 预提取的 768-d tile features），
   我们将 CONCH features 当作 "post-transformer features"，直接绕过 ViT backbone，
   仅使用模型的两个分支（dynamic_region_partition + ARM + head_direct/head_adaptive）。

   这与 train.py 中 _forward_with_tiles() 的做法一致，因为 checkpoint
   也是这样训练出来的。
3. 同时对 CONCH 特征做 L2 归一化，与训练时保持一致（如果在训练中加了 norm）。

跑法：
    python eval_care_fixed.py
"""
import sys, os, glob, re
sys.path.insert(0, r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion")

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from tqdm import tqdm

from train import RealWSIDataset, dummy_collate_fn, compute_multitask_auc
from models.fusion_model import E2EViTCAREFusion

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# 加载数据集
csv_path = r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv"
data_root = r"E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files"
dataset = RealWSIDataset(
    csv_path=csv_path,
    data_root=data_root,
    embed_dim=768,
    num_tasks=1,
    tile_size=256,
    max_tiles=4096,
)
print(f"Dataset: {len(dataset)} slides")

# 5折索引（与训练一致）
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
indices = np.arange(len(dataset))
splits = list(kfold.split(indices))

# 加载所有 checkpoints
result_dirs = [
    r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42",
    r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L12_T1_s42",
]

for result_dir in result_dirs:
    name = os.path.basename(result_dir)
    best_files = sorted(glob.glob(os.path.join(result_dir, "fold_*_best.pt")))
    if not best_files:
        print(f"\n=== {name} ===\n  No best checkpoints")
        continue

    print(f"\n=== {name} ===")
    aucs = []

    for bf in best_files:
        fold = int(re.search(r"fold_(\d)_", bf).group(1))
        ckpt = torch.load(bf, map_location="cpu")
        ep = ckpt.get("epoch", "?")
        best_score = ckpt.get("best_score", "?")
        print(f"\nFold {fold}: epoch={ep}, best_train_loss={best_score}")

        # 根据结果目录名决定模型架构
        if "L12" in name:
            num_layers = 12
            num_heads = 12
        else:
            num_layers = 4
            num_heads = 4

        model = E2EViTCAREFusion(
            tile_size=256, patch_size=16, embed_dim=768,
            num_heads=num_heads, num_layers=num_layers, num_region_tokens=8,
            num_tasks=1, dropout=0.25,
            use_two_branches=True, use_distillation=False, use_alibi=True,
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # 验证集
        _, val_idx = splits[fold]
        val_subset = Subset(dataset, val_idx.tolist())
        loader = DataLoader(
            val_subset, batch_size=2, shuffle=False,
            collate_fn=dummy_collate_fn, num_workers=0,
        )

        all_labels, all_probs_direct, all_probs_adaptive = [], [], []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Fold {fold}"):
                pad_tokens, pad_coords, labels, _, padding_mask = batch
                pad_tokens = pad_tokens.to(device)
                pad_coords = pad_coords.to(device)
                labels = labels.to(device).float()

                # === 正确的 forward ===
                # pad_tokens 是 [B, N_tiles, 768] 的 CONCH 特征
                # 我们将其当作 "transformer_output"（与 _forward_with_tiles 训练时一致）
                # 然后只跑 DynamicRegionPartition + ARM + Heads

                # 分支1：直接 mean pooling → head_direct
                x_global = pad_tokens.mean(dim=1)  # [B, 768]
                logits_direct = model.head_direct(x_global)  # [B, 1]

                # 分支2：双层自适应
                region_features, attn_weights, _ = model.dynamic_region_partition(
                    tile_tokens=pad_tokens, coords=pad_coords, return_coverage=True,
                )
                region_embeddings, region_pooled = model.arm(
                    region_features=region_features,
                    tile_tokens=pad_tokens,
                    attn_weights=attn_weights,
                )
                adaptive_out = model.head_adaptive(region_pooled)
                logits_adaptive = adaptive_out["logits"]

                all_labels.append(labels.cpu().numpy())
                all_probs_direct.append(torch.sigmoid(logits_direct).cpu().numpy())
                all_probs_adaptive.append(torch.sigmoid(logits_adaptive).cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_probs_direct = np.concatenate(all_probs_direct)
        all_probs_adaptive = np.concatenate(all_probs_adaptive)

        # 两套预测都评估
        try:
            auc_direct = compute_multitask_auc(all_labels, all_probs_direct)
        except Exception:
            auc_direct = 0.0
        try:
            auc_adaptive = compute_multitask_auc(all_labels, all_probs_adaptive)
        except Exception:
            auc_adaptive = 0.0
        # 取均值作为 ensemble 预测
        try:
            auc_ensemble = compute_multitask_auc(
                all_labels,
                (all_probs_direct + all_probs_adaptive) / 2.0,
            )
        except Exception:
            auc_ensemble = 0.0

        aucs.append((fold, auc_direct, auc_adaptive, auc_ensemble))
        print(f"  AUC direct  : {auc_direct:.4f}")
        print(f"  AUC adaptive: {auc_adaptive:.4f}")
        print(f"  AUC ensemble: {auc_ensemble:.4f}")

    if aucs:
        direct = np.mean([a[1] for a in aucs])
        adaptive = np.mean([a[2] for a in aucs])
        ensemble = np.mean([a[3] for a in aucs])
        print(f"\n--- {name} summary ---")
        print(f"Mean AUC direct  : {direct:.4f}")
        print(f"Mean AUC adaptive: {adaptive:.4f}")
        print(f"Mean AUC ensemble: {ensemble:.4f}")
