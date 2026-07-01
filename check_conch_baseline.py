"""
Quick sanity check: can CONCH features alone distinguish tumor vs normal?
Uses a simple MLP + 5-fold CV to get a rough upper-bound AUC.
"""
import os, sys, glob, re
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

DATA_ROOT = r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files'
CSV_PATH = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'

# 加载 CSV
df = pd.read_csv(CSV_PATH)
print(f'CSV: {len(df)} rows, {df["label"].sum():.0f} positive')

# 建立 slide_id -> pt_path 映射
# CSV 中的 pt_path 列有完整路径
df['pt_path_clean'] = df['pt_path'].str.replace('\\', '/', regex=False)

# 加载特征和标签
X_list = []
y_list = []
missing = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc='Loading features'):
    pt = row['pt_path_clean']
    if not os.path.exists(pt):
        missing += 1
        continue
    try:
        feat = torch.load(pt, map_location='cpu', weights_only=True).float()
        if feat.ndim == 2:
            # Mean pool across patches -> WSI-level feature
            x = feat.mean(dim=0).numpy()  # [768]
        else:
            x = feat.numpy()
        X_list.append(x)
        y_list.append(row['label'])
    except Exception as e:
        print(f'Error loading {pt}: {e}')
        missing += 1

print(f'Loaded: {len(X_list)}, missing: {missing}')
X = np.stack(X_list)
y = np.array(y_list)
print(f'X shape: {X.shape}, y: {y.sum():.0f} pos / {(1-y).sum():.0f} neg')

# 标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 5-fold CV with Logistic Regression
kf = KFold(n_splits=5, shuffle=True, random_state=42)
aucs = []
for fold, (tr, va) in enumerate(kf.split(X)):
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X[tr], y[tr])
    proba = model.predict_proba(X[va])[:, 1]
    try:
        auc = roc_auc_score(y[va], proba)
    except:
        auc = 0.5
    aucs.append(auc)
    print(f'Fold {fold}: AUC={auc:.4f}')

print(f'\n=== CONCH Features + Mean Pool + LogReg ===')
print(f'Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}')

# Also try: first few PCA components
from sklearn.decomposition import PCA
pca = PCA(n_components=32)
X_pca = pca.fit_transform(X)
print(f'PCA explained variance (32 comps): {pca.explained_variance_ratio_.sum():.4f}')

aucs_pca = []
for fold, (tr, va) in enumerate(kf.split(X_pca)):
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_pca[tr], y[tr])
    proba = model.predict_proba(X_pca[va])[:, 1]
    try:
        auc = roc_auc_score(y[va], proba)
    except:
        auc = 0.5
    aucs_pca.append(auc)

print(f'\nCONCH Features + PCA(32) + LogReg')
print(f'Mean AUC: {np.mean(aucs_pca):.4f} +/- {np.std(aucs_pca):.4f}')
