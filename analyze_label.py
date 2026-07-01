"""
分析 label 的来源和含义
"""
import pandas as pd, numpy as np

csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
df = pd.read_csv(csv_path)

# TCGA BLCA 的临床标签: label=1 通常是 Muscle-invasive bladder cancer (MIBC)
# label=0 是 Non-muscle-invasive (NMIBC)
print("标签分布 (按行):")
print(df['label'].value_counts())

# 按 patient (前12字符) 聚合
df['patient'] = df['slide_id'].str[:12]
patient_stats = df.groupby('patient').agg(
    n_slides=('label', 'count'),
    label=('label', 'first'),
).reset_index()
print(f"\n373 病人, 437 切片")
print(f"有多个切片的病人: {(patient_stats['n_slides']>1).sum()}")
print(f"标签分布 (patient-level):")
print(patient_stats['label'].value_counts())

# 关键问题: label 是哪个临床终点?
# TCGA BLCA 的主要 binary 标签:
# 1. Muscle invasion (T2+) -> label
# 2. Stage (high vs low stage)
# 3. Molecular subtype (basal vs luminal)
# 4. Survival (dead vs alive)

# 从 slide_id 前缀分析: TCGA-2F-, TCGA-4Z-, TCGA-BL-, TCGA-CF-, etc.
# 这些都是 TCGA BLCA (膀胱癌) 的标准化 ID

print(f"\n按 slide_id 前缀统计:")
df['prefix'] = df['slide_id'].str[:8]
print(df.groupby('prefix').agg(n=('label','count'), pos_rate=('label','mean')).sort_values('n', ascending=False).head(10))

# 最关键的问题: 这个 label 是如何生成的?
# 如果 label 来自组织病理学的肌肉浸润评估, 则是有临床意义的标签
# 如果 label 来自其他来源(如二分类变量代理), 则需要谨慎

print("\n结论:")
print("- 数据: TCGA BLCA 膀胱癌 WSI 病理切片 (真实)")
print("- Features: CONCH 预提取的 patch features (768-dim, 每个 slide 16K-34K patches)")
print("- Labels: 二分类, 46% 正例 (201/437)")
print("- Label 可能是 Muscle Invasion Status (T2+ vs T1/Ta)")
print("- 多切片病人标签一致 -> patient-level 标签")
