"""提取已有checkpoint的验证结果"""
import torch, os, glob, re
import numpy as np

project_dirs = [
    r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\results_real\real_E2E_CARE_K8_L4_T1_s42",
    r"C:\Users\cwnu\Desktop\CARE-E2E-Fusion\项目_1_MoTo-CARE\results_real\real_MoToCARE_R8_T1_s42",
]

for d in project_dirs:
    name = d.split("\\")[-2]
    print(f"\n=== {name} ===")
    bests = sorted(glob.glob(os.path.join(d, "fold_*_best.pt")))
    if not bests:
        print("  No results")
        continue
    # 从文件名提取 fold 号和 epoch
    folds = []
    for f in bests:
        ckpt = torch.load(f, map_location="cpu")
        fold = re.search(r"fold_(\d)_", f).group(1)
        epoch = ckpt.get("epoch", "?")
        best_score = ckpt.get("best_score", "?")
        folds.append((int(fold), epoch, best_score))
        print(f"  Fold {fold}: epoch={epoch}, best_score={best_score}")
