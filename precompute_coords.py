# -*- coding: utf-8 -*-
import sys, os, random, time, pickle
sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')

from train_wsi import build_svs_index, find_svs, sample_tiles_quick
import pandas as pd
import numpy as np

csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
wsi_root = r'E:\TCGA-WSI-data\TCGA_WSI_BLCA'
TILE_SIZE = 256
MAX_TILES = 512
TISSUE_THRESH = 0.3

df = pd.read_csv(csv_path)
df['patient_id'] = df['slide_id'].str[:12]
svs_index = build_svs_index(wsi_root)
df['svs_path'] = df['patient_id'].map(lambda p: find_svs(p, svs_index))
df_valid = df[df['svs_path'].notna()].reset_index(drop=True)
print("Valid slides:", len(df_valid))

# Pre-compute tile coords for ALL slides
# This is done once and cached to avoid re-reading level 2 every epoch
coords_cache = {}
t0_total = time.time()
n_done = 0

for idx, row in df_valid.iterrows():
    svs = row['svs_path']
    seed = 42 + idx  # same seed as dataset will use
    coords = sample_tiles_quick(svs, TILE_SIZE, MAX_TILES, seed, TISSUE_THRESH)
    coords_cache[svs] = coords
    n_done += 1
    elapsed = time.time() - t0_total
    eta = elapsed / n_done * (len(df_valid) - n_done) if n_done > 0 else 0
    print("\r  %d/%d slides done. Elapsed: %.0fs. ETA: %.0fs. Last: %s (%d tiles)" % (
        n_done, len(df_valid), elapsed, eta, os.path.basename(svs), len(coords)), end='')

print("\nTotal time: %.0fs for %d slides" % (time.time() - t0_total, len(df_valid)))

# Save
cache_path = os.path.join(os.path.dirname(csv_path), 'tile_coords_cache.pkl')
with open(cache_path, 'wb') as f:
    pickle.dump(coords_cache, f)
print("Saved to:", cache_path)

# Stats
all_lengths = [len(v) for v in coords_cache.values()]
print("Tile count: min=%d, max=%d, mean=%.0f" % (
    min(all_lengths), max(all_lengths), np.mean(all_lengths)))
