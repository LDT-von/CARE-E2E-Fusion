"""
验证 CARE-E2E-Fusion 数据真实性
1. PT 文件内容 (维度、数值范围)
2. 标签分布
3. 标签是什么（肌肉浸润？分子标志物？）
"""
import os, pickle, numpy as np, pandas as pd, torch

csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
df = pd.read_csv(csv_path)
print(f'CSV: {len(df)} rows, {df.columns.tolist()}')

# 标签分布
print(f'\n标签分布:')
print(df['label'].value_counts())
print(f'正例比例: {df["label"].mean():.3f}')
print(f'唯一 slide_id: {df["slide_id"].nunique()}')
print(f'唯一 .pt 文件: {df["pt_path"].nunique()}')

# 检查 PT 文件是否真的存在
pt_exists = df['pt_path'].apply(os.path.exists)
print(f'\n.pt 文件存在: {pt_exists.sum()}/{len(df)}')

# 加载几个 PT 文件检查内容
sample_pts = df['pt_path'].dropna().unique()[:5]
print(f'\n前 5 个 .pt 文件检查:')
for pt in sample_pts:
    pt = pt.replace('\\', '/')
    if os.path.exists(pt):
        try:
            d = torch.load(pt, map_location='cpu')
            if isinstance(d, dict):
                shape = d.get('features', list(d.values())[0]).shape
            elif isinstance(d, torch.Tensor):
                shape = d.shape
            else:
                shape = np.array(d).shape
            print(f'  {os.path.basename(pt)[:60]}: shape={shape}')
        except Exception as e:
            print(f'  {os.path.basename(pt)[:60]}: ERROR {e}')
    else:
        print(f'  {pt}: FILE NOT FOUND')

# 检查 tile_coords_cache
cache_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\tile_coords_cache.pkl'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    print(f'\ntile_coords_cache: {len(cache)} entries')
    # 看前 2 个
    for i, (k, v) in enumerate(list(cache.items())[:2]):
        print(f'  key={os.path.basename(str(k))[:50]}, coords shape={np.array(v).shape if v is not None else None}')

# 检查 label 的来源: 可能是 muscle invasion (TCGA BLCA 的临床标签)
# 看是否与多倍体 (同病人多张切片) 一致
print(f'\n同病人多切片:')
dup_patients = df.groupby('slide_id').size()
print(f'  最多: {dup_patients.max()} 张切片 (同 slide_id)')
# 按 label 检查同一病人的切片是否一致 label
patient_label = df.groupby(df['slide_id'].str[:12])['label'].agg(['mean', 'count'])
patient_label['all_same'] = (patient_label['mean'].isin([0, 1]))
multi_slice = patient_label[patient_label['count'] > 1]
print(f'  有 {len(multi_slice)} 个病人有多张切片')
inconsistent = multi_slice[~multi_slice['all_same']]
print(f'  其中标签不一致的: {len(inconsistent)}')
if len(inconsistent) > 0:
    print(f'  -> 标签可能是 slide-level (组织块级别) 而非 patient-level')
else:
    print(f'  -> 所有多切片病人的标签一致: 标签可能是 patient-level')
