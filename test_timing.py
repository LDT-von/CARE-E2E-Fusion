# -*- coding: utf-8 -*-
import sys, os, random, time
sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')

from train_wsi import WSIDataset, sample_tiles_quick, build_svs_index, find_svs
import pandas as pd

# Time: how long does one tile sample + read take?
csv_path = r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv'
wsi_root = r'E:\TCGA-WSI-data\TCGA_WSI_BLCA'

df = pd.read_csv(csv_path)
df['patient_id'] = df['slide_id'].str[:12]
svs_index = build_svs_index(wsi_root)
df['svs_path'] = df['patient_id'].map(lambda p: find_svs(p, svs_index))
df_valid = df[df['svs_path'].notna()].reset_index(drop=True)
print("Valid slides:", len(df_valid))

# Test timing for one slide
import openslide
svs = df_valid.iloc[0]['svs_path']
print("Test slide:", svs)

t0 = time.time()
coords = sample_tiles_quick(svs, tile_size=256, max_tiles=16, seed=42, tissue_thresh=0.3)
t1 = time.time()
print("Sample coords time:", round(t1-t0, 2), "s, got", len(coords), "tiles")

t0 = time.time()
slide = openslide.OpenSlide(svs)
tiles = []
for (y, x) in coords[:4]:
    tile = slide.read_region((x, y), 0, (256, 256)).convert('RGB')
    tiles.append(tile)
slide.close()
t1 = time.time()
print("Read 4 tiles time:", round(t1-t0, 2), "s")

# Estimate time for full dataset
n_train = int(len(df_valid) * 0.8)
print("\nEstimate for", n_train, "train slides x 16 tiles:")
print("  Coord sampling (L2 read ~7s each):", round(n_train * 7 / 60, 1), "min")
print("  Tile reads (256 KB/tile, ~10 tiles/s):", round(n_train * 16 / 10 / 60, 1), "min")
