import numpy as np
import os

pt_dir = r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files'
npy_files = [f for f in os.listdir(pt_dir) if f.endswith('.npy')]
pt_files = [f for f in os.listdir(pt_dir) if f.endswith('.pt')]
print(f'.npy files: {len(npy_files)}')
print(f'.pt files: {len(pt_files)}')

for f in npy_files[:3]:
    path = os.path.join(pt_dir, f)
    size = os.path.getsize(path)
    data = np.load(path, allow_pickle=True)
    d = data.item() if hasattr(data, 'item') else data
    print(f'\n{f}: size={size/1024:.1f}KB')
    print(f'  Keys: {list(d.keys())}')
    feat = d['feature']
    if hasattr(feat, 'shape'):
        print(f'  feature: shape={feat.shape}')
    else:
        print(f'  feature: list of len {[len(x) for x in feat[:3]]}')
    if 'coords' in d:
        print(f'  coords: shape={d["coords"].shape}')
    elif 'index' in d:
        idx = d['index']
        print(f'  index: len={len(idx)}, first 3:')
        for i in idx[:3]:
            print(f'    {i}')

# Also check if .pt files have coordinate info embedded
print('\n--- Checking .pt files for coordinate info ---')
import torch
for pt in pt_files[:3]:
    path = os.path.join(pt_dir, pt)
    try:
        data = torch.load(path, map_location='cpu', weights_only=False)
        print(f'{pt}: type={type(data).__name__}')
        if isinstance(data, dict):
            print(f'  keys: {list(data.keys())}')
        elif hasattr(data, 'shape'):
            print(f'  shape: {data.shape}')
    except Exception as e:
        print(f'{pt}: error loading - {e}')
