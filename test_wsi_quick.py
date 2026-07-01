import sys
sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')

# Test: just 2 slides
from train_wsi import WSIDataset, collate_wsi, BATCH_MAX_TILES
from torch.utils.data import DataLoader
import torch

ds = WSIDataset(
    csv_path=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv',
    wsi_root=r'E:\TCGA-WSI-data\TCGA_WSI_BLCA',
    tile_size=256,
    max_tiles=16,  # small for test
    tissue_thresh=0.3,
    seed=42,
    verbose=True,
)
print(f'Dataset: {len(ds)} slides')

# Load 2 samples
import time
t0 = time.time()
sample0 = ds[0]
sample1 = ds[1]
print(f'Loaded 2 samples in {time.time()-t0:.1f}s')
print(f'Sample 0: {sample0["num_tiles"]} tiles, shape={sample0["images"].shape}')

# Collate
batch = collate_wsi([sample0, sample1])
print(f'Batch: images={batch[0].shape}, coords={batch[1].shape}')
print(f'  labels={batch[2]}, slide_ids={batch[3]}, mask={batch[4].shape}')
