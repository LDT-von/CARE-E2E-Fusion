# -*- coding: utf-8 -*-
"""Debug: verify data loading is correct."""
from __future__ import print_function
import os, sys, pickle
from pathlib import Path
import numpy as np
import torch
import pandas as pd

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

# Load cache
cache_path = r'c:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\tile_coords_cache.pkl'
with open(cache_path, 'rb') as f:
    coords_cache = pickle.load(f)
print('Cache: %d entries' % len(coords_cache))

# Load CSV
csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
df = pd.read_csv(csv_path)
print('CSV: %d rows' % len(df))

# Check first few rows
df['pt_path_norm'] = df['pt_path'].str.replace('\\', '/')
for i in range(5):
    row = df.iloc[i]
    pt_path = row['pt_path_norm']
    slide_id = row['slide_id']
    label = row['label']
    
    # Load features
    try:
        tokens = torch.load(pt_path, map_location='cpu')
        if isinstance(tokens, dict):
            tokens = tokens.get('features', list(tokens.values())[0])
        tokens = tokens.float()
        if tokens.dim() == 3:
            tokens = tokens.squeeze(0)
        n_feat = tokens.shape[0]
        feat_dim = tokens.shape[1]
    except Exception as e:
        n_feat = 0
        feat_dim = 0
    
    # Find matching coords
    pid = slide_id[:12]
    matched = 0
    for svs_path, coords in coords_cache.items():
        base = os.path.basename(svs_path)
        if pid in svs_path:
            matched = len(coords)
            break
    
    print('  %d: slide=%s label=%d pt=%s n_feat=%d feat_dim=%d matched_coords=%d' % (
        i, slide_id, label, os.path.basename(pt_path), n_feat, feat_dim, matched))

# Check label distribution
print('\nLabel distribution:')
print(df['label'].value_counts())
print('Total: %d' % len(df))

# Check if all pt files exist
df_valid = df[df['pt_path_norm'].apply(os.path.exists)]
print('\nValid pt files: %d/%d' % (len(df_valid), len(df)))
