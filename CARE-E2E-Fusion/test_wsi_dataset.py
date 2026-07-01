import sys
sys.path.insert(0, r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion')
from wsi_dataset import WSIPatchDataset, collate_patches

ds = WSIPatchDataset(
    csv_path=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv',
    wsi_root=r'E:\TCGA-WSI-data\TCGA_WSI_BLCA',
    tile_size=256,
    max_patches=64,
    tissue_threshold=0.5,
    downsample_for_mask=32,
    seed=42,
)
print(f'Dataset size: {len(ds)}')

import time
t0 = time.time()
sample = ds[0]
print(f'Loaded sample 0 in {time.time()-t0:.2f}s')
print(f'patch_images shape: {sample["patch_images"].shape}')
print(f'coords shape: {sample["coords"].shape}')
print(f'label: {sample["label"]}')
print(f'patch mean: {sample["patch_images"].mean():.3f}, std: {sample["patch_images"].std():.3f}')

from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_patches, num_workers=0)
t0 = time.time()
for i, batch in enumerate(loader):
    if i == 0:
        print(f'\nFirst batch shapes:')
        print(f'  patches: {batch[0].shape}')
        print(f'  coords: {batch[1].shape}')
        print(f'  labels: {batch[2].shape}')
        print(f'  padding_mask: {batch[4].shape}')
    if i == 2:
        print(f'Loaded 3 batches in {time.time()-t0:.2f}s')
        break
