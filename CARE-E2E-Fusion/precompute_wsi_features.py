# -*- coding: utf-8 -*-
"""
Pre-extract tile features from WSI images using a small CNN.
Saves features to .npy files for later training.
"""
from __future__ import print_function
import os, sys, random, argparse, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TILE_SIZE = 224


class SmallCNN(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(192, out_dim)
        self.out_dim = out_dim

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x


def build_svs_index(wsi_root):
    idx = {}
    for d in os.listdir(wsi_root):
        full = os.path.join(wsi_root, d)
        if not os.path.isdir(full):
            continue
        for f in os.listdir(full):
            if f.endswith('.svs'):
                pid = f[:12]
                idx.setdefault(pid, []).append((f, full))
    return idx


def find_svs(patient_id, svs_index):
    if patient_id not in svs_index:
        return None
    return os.path.join(svs_index[patient_id][0][1], svs_index[patient_id][0][0])


def sample_tiles_quick(svs_path, tile_size, max_tiles, seed, tissue_thresh=0.3):
    import openslide
    try:
        slide = openslide.OpenSlide(svs_path)
        w, h = slide.dimensions
        level = min(2, slide.level_count - 1)
        lw, lh = slide.level_dimensions[level]
        scale = slide.level_downsamples[level]
        tile_l2 = max(2, tile_size // int(scale))
        img = slide.read_region((0, 0), level, (lw, lh)).convert('RGB')
        arr = np.asarray(img, dtype=np.float32)
        gray = arr.mean(axis=-1)
        del img, arr
        mask = gray < (0.85 * 255)
        rng = random.Random(seed)
        coords = []
        for y in range(0, lh - tile_l2, tile_l2):
            for x in range(0, lw - tile_l2, tile_l2):
                if mask[y:y + tile_l2, x:x + tile_l2].mean() > tissue_thresh:
                    coords.append((int(y * scale), int(x * scale)))
                    if len(coords) >= max_tiles:
                        break
            if len(coords) >= max_tiles:
                break
        if not coords:
            n = int(np.sqrt(max_tiles))
            for i in range(n):
                for j in range(n):
                    coords.append((int(h * (i + 0.5) / n), int(w * (j + 0.5) / n)))
        rng.shuffle(coords)
        slide.close()
        return coords[:max_tiles]
    except:
        return []


def load_coords_cache(csv_path):
    cache_path = os.path.join(os.path.dirname(csv_path), 'tile_coords_cache.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    return {}


class WSITilesDataset(Dataset):
    def __init__(self, csv_path, wsi_root, out_dir, tile_size=224, max_tiles=128,
                 tissue_thresh=0.3, seed=42, coords_cache=None, verbose=True):
        import pandas as pd
        self.df = pd.read_csv(csv_path)
        self.wsi_root = wsi_root
        self.out_dir = out_dir
        self.tile_size = tile_size
        self.max_tiles = max_tiles
        self.tissue_thresh = tissue_thresh
        self.seed = seed
        self.svs_index = build_svs_index(wsi_root)
        self.df = self.df.copy()
        self.df['patient_id'] = self.df['slide_id'].str[:12]
        self.df['svs_path'] = self.df['patient_id'].map(
            lambda p: find_svs(p, self.svs_index)
        )
        n_before = len(self.df)
        self.df = self.df[self.df['svs_path'].notna()].reset_index(drop=True)
        # Dedup by svs_path (one WSI per patient)
        self.df = self.df.drop_duplicates(subset=['svs_path']).reset_index(drop=True)
        if verbose:
            print("[WSI] %d unique SVS files" % len(self.df))
        self._coords_cache = coords_cache if coords_cache else {}

    def __len__(self):
        return len(self.df)

    def _get_coords(self, svs):
        if svs not in self._coords_cache:
            self._coords_cache[svs] = sample_tiles_quick(
                svs, self.tile_size, self.max_tiles, self.seed, self.tissue_thresh
            )
        return self._coords_cache[svs]

    def _out_path(self, svs):
        h = abs(hash(svs)) % (10 ** 8)
        return os.path.join(self.out_dir, '%08d.npy' % h)

    def __getitem__(self, idx):
        import openslide
        row = self.df.iloc[idx]
        svs = row['svs_path']
        slide_id = row['slide_id']
        out_path = self._out_path(svs)
        if os.path.exists(out_path):
            return slide_id, out_path, 'cached'
        coords = self._get_coords(svs)
        try:
            slide = openslide.OpenSlide(svs)
            tiles = []
            for (y, x) in coords:
                tile = slide.read_region((x, y), 0, (self.tile_size, self.tile_size)).convert('RGB')
                tiles.append(np.asarray(tile, dtype=np.uint8))
            slide.close()
            n = len(tiles)
            if n == 0:
                tiles = [np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)]
                n = 1
        except:
            tiles = [np.zeros((self.tile_size, self.tile_size, 3), dtype=np.uint8)]
            n = 1

        tile_stack = np.stack(tiles, axis=0)
        return slide_id, out_path, ('save', tile_stack)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str,
        default=r'C:\Users\cwnu\Desktop\CARE-E2E-Fusion\CARE-E2E-Fusion\blca_slides.csv')
    parser.add_argument('--wsi_root', type=str,
        default=r'E:\TCGA-WSI-data\TCGA_WSI_BLCA')
    parser.add_argument('--out_dir', type=str, default='wsi_tile_cache')
    parser.add_argument('--max_tiles', type=int, default=128)
    parser.add_argument('--embed_dim', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_tiles', type=int, default=8)
    parser.add_argument('--tile_size', type=int, default=224)
    parser.add_argument('--max_slides', type=int, default=-1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    coords_cache = load_coords_cache(args.csv_path)

    dataset = WSITilesDataset(
        csv_path=args.csv_path, wsi_root=args.wsi_root, out_dir=args.out_dir,
        tile_size=args.tile_size, max_tiles=args.max_tiles,
        coords_cache=coords_cache, verbose=True,
    )

    if args.max_slides > 0:
        dataset.df = dataset.df.head(args.max_slides)

    # Step 1: Save tiles
    print('\nStep 1: Saving tiles...')
    n_saved = 0
    for idx in range(len(dataset)):
        slide_id, out_path, payload = dataset[idx]
        if payload == 'cached':
            continue
        _, tile_stack = payload  # [N, 224, 224, 3]
        np.save(out_path, tile_stack)
        n_saved += 1
        if (idx + 1) % 20 == 0:
            print('  saved %d slides' % (idx + 1))
    print('  saved %d new slides' % n_saved)

    # Step 2: Extract features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nStep 2: Extracting features on %s...' % device)
    backbone = SmallCNN(out_dim=args.embed_dim).to(device).eval()

    feat_dir = args.out_dir + '_feat'
    os.makedirs(feat_dir, exist_ok=True)

    n_done = 0
    mean_t = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std_t = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

    for idx in tqdm(range(len(dataset)), desc='Extract'):
        slide_id = dataset.df.iloc[idx]['slide_id']
        svs = dataset.df.iloc[idx]['svs_path']
        out_path = dataset._out_path(svs)
        feat_path = os.path.join(feat_dir, os.path.basename(out_path))
        if os.path.exists(feat_path):
            n_done += 1
            continue
        if not os.path.exists(out_path):
            continue
        tile_stack = np.load(out_path)  # [N, 224, 224, 3]
        x = torch.from_numpy(tile_stack).float() / 255.0
        x = x.permute(0, 3, 1, 2)
        x = (x - mean_t) / std_t
        # Process in mini-batches
        feats = []
        for i in range(0, x.shape[0], args.batch_tiles):
            batch = x[i:i + args.batch_tiles].to(device)
            with torch.no_grad():
                f = backbone(batch)
            feats.append(f.cpu().numpy())
        feat = np.concatenate(feats, axis=0)
        np.save(feat_path, feat)
        n_done += 1

    print('\nDone!')
    print('  Total slides: %d' % len(dataset))
    print('  Features saved to: %s' % feat_dir)


if __name__ == '__main__':
    main()
