"""
WSI DataLoader for real .svs images
====================================

使用 openslide 从全切片图像中：
1. 在低倍 (level 2) 生成组织 mask（Otsu 阈值或亮度阈值）
2. 在 level 0 上以指定 tile_size 步长提取 patches
3. 输出 256x256 RGB tensors（ImageNet 标准化）

关键参数：
- target_mpp: 目标每像素微米（默认 0.5，相当于 20x）
- tile_size: tile 像素尺寸 @ target_mpp（默认 256，即 256 @ 20x）
- max_patches: 每张 slide 最多取的 patches（默认 1024）
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import openslide


# ImageNet 标准化
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_svs_index(wsi_root: str) -> dict:
    """建立 patient_id(前12字符) -> [(filename, full_path), ...] 的索引。"""
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


def find_svs_for_patient(patient_id: str, svs_index: dict) -> Optional[str]:
    """找一个 patient 的第一个 .svs 文件。"""
    if patient_id not in svs_index:
        return None
    fname, fdir = svs_index[patient_id][0]
    return os.path.join(fdir, fname)


def otsu_threshold(gray: np.ndarray) -> int:
    """Otsu 阈值算法。"""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size
    if total == 0:
        return 128

    sum_total = np.dot(np.arange(256), hist.astype(np.float64))
    sum_bg = 0.0
    w_bg = 0.0
    max_var = 0.0
    threshold = 128

    for t in range(256):
        w_bg += hist[t]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += t * hist[t]
        m_bg = sum_bg / w_bg
        m_fg = (sum_total - sum_bg) / w_fg
        var_between = w_bg * w_fg * (m_bg - m_fg) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold = t
    return threshold


def get_tissue_mask(slide: openslide.OpenSlide, downsample: int = 32) -> np.ndarray:
    """在低分辨率上生成组织 mask。"""
    w, h = slide.dimensions
    level = slide.get_best_level_for_downsample(downsample)
    level_w, level_h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (level_w, level_h)).convert('RGB')
    arr = np.asarray(img)
    # 转灰度
    gray = arr.mean(axis=-1)
    # Otsu
    thresh = otsu_threshold(gray.astype(np.uint8))
    mask = gray < thresh
    return mask, level


def sample_tiles_for_wsi(
    slide: openslide.OpenSlide,
    mask: np.ndarray,
    mask_level: int,
    tile_size_l0: int,
    max_tiles: int,
    tissue_threshold: float = 0.5,
    seed: int = 42,
) -> List[Tuple[int, int]]:
    """根据组织 mask 采样 tile 坐标 (in level 0)。"""
    mask_h, mask_w = mask.shape
    scale = slide.level_downsamples[mask_level]
    tile_size_mask = max(1, int(round(tile_size_l0 / scale)))

    # 找到所有 (y, x) 满足 tile 区域主要是组织
    coords: List[Tuple[int, int]] = []
    for y in range(0, mask_h - tile_size_mask, tile_size_mask):
        for x in range(0, mask_w - tile_size_mask, tile_size_mask):
            region = mask[y:y + tile_size_mask, x:x + tile_size_mask]
            if region.mean() > tissue_threshold:
                coords.append((int(y * scale), int(x * scale)))

    if not coords:
        # 退而求其次：取中心区域
        w, h = slide.dimensions
        coords = [(h // 2, w // 2)]

    random.Random(seed).shuffle(coords)
    return coords[:max_tiles]


def read_patch(slide: openslide.OpenSlide, y_l0: int, x_l0: int, tile_size: int) -> np.ndarray:
    """在 level 0 读取一个 tile，返回 HWC uint8 numpy。"""
    w, h = slide.dimensions
    # 边界裁剪
    x = max(0, min(x_l0, w - tile_size))
    y = max(0, min(y_l0, h - tile_size))
    img = slide.read_region((x, y), 0, (tile_size, tile_size)).convert('RGB')
    return np.asarray(img, dtype=np.uint8)


class WSIPatchDataset(Dataset):
    """真实 WSI patch 数据集：缓存 tile 坐标 → 每次 __getitem__ 读取 patch。"""

    def __init__(
        self,
        csv_path: str,
        wsi_root: str,
        tile_size: int = 256,
        max_patches: int = 256,
        tissue_threshold: float = 0.5,
        downsample_for_mask: int = 32,
        seed: int = 42,
        normalize: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.wsi_root = wsi_root
        self.svs_index = build_svs_index(wsi_root)
        self.tile_size = tile_size
        self.max_patches = max_patches
        self.tissue_threshold = tissue_threshold
        self.downsample = downsample_for_mask
        self.seed = seed
        self.normalize = normalize

        # 只保留能找到 WSI 的样本
        self.df['patient_id'] = self.df['slide_id'].str[:12]
        self.df['svs_path'] = self.df['patient_id'].map(
            lambda p: find_svs_for_patient(p, self.svs_index)
        )
        n_total = len(self.df)
        self.df = self.df[self.df['svs_path'].notna()].reset_index(drop=True)
        print(f"[WSIPatchDataset] Filtered: {len(self.df)}/{n_total} samples have WSI files")

        # 每个 WSI 预计算 tile 坐标（cache）
        self._tile_cache: dict = {}
        print(f"[WSIPatchDataset] Pre-computing tile coords for {len(self.df)} slides...")
        for idx, row in self.df.iterrows():
            svs = row['svs_path']
            if svs not in self._tile_cache:
                try:
                    slide = openslide.OpenSlide(svs)
                    mask, mask_level = get_tissue_mask(slide, self.downsample)
                    coords = sample_tiles_for_wsi(
                        slide, mask, mask_level, self.tile_size, self.max_patches,
                        tissue_threshold=self.tissue_threshold, seed=self.seed + idx,
                    )
                    slide.close()
                    self._tile_cache[svs] = coords
                except Exception as e:
                    print(f"  [WARN] Failed on {svs}: {e}")
                    self._tile_cache[svs] = []
            if (idx + 1) % 50 == 0:
                print(f"  ... {idx + 1}/{len(self.df)}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        svs = row['svs_path']
        coords = self._tile_cache.get(svs, [])

        # 读取 tile
        slide = openslide.OpenSlide(svs)
        n_tiles = len(coords)
        if n_tiles == 0:
            # WSI 组织区域太小：取中心 tile
            w, h = slide.dimensions
            tile = read_patch(slide, h // 2, w // 2, self.tile_size)
            coords_used = [(h // 2, w // 2)]
        else:
            tiles = []
            for (y, x) in coords:
                tiles.append(read_patch(slide, y, x, self.tile_size))
            tile = np.stack(tiles, axis=0)  # [N, H, W, 3]
            coords_used = coords
        slide.close()

        # 转为 tensor
        if tile.ndim == 3:
            tile = tile[None, ...]  # [1, H, W, 3]
        x = torch.from_numpy(tile).float() / 255.0  # [N, H, W, 3]
        x = x.permute(0, 3, 1, 2)  # [N, 3, H, W]
        if self.normalize:
            mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
            std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
            x = (x - mean) / std

        # 模拟坐标（占位：tile 索引的归一化位置）
        n = x.shape[0]
        coords_norm = torch.linspace(0, 1, n).unsqueeze(1).expand(-1, 2).float()

        # 标签
        if 'label' in self.df.columns:
            label = torch.tensor([float(row['label'])], dtype=torch.float32)
        else:
            label = torch.zeros(1)

        return {
            'patch_images': x,           # [N, 3, H, W] 已标准化
            'coords': coords_norm,        # [N, 2]
            'label': label,               # [1]
            'slide_id': row['slide_id'],
            'num_patches': n,
        }


def collate_patches(batch):
    """Collate: padding to max patches in batch."""
    labels = torch.stack([s['label'] for s in batch])
    n = max(s['num_patches'] for s in batch)
    c = batch[0]['patch_images'].shape[1]
    h = batch[0]['patch_images'].shape[2]
    w = batch[0]['patch_images'].shape[3]

    pad_imgs = torch.zeros(len(batch), n, c, h, w)
    pad_coords = torch.zeros(len(batch), n, 2)
    pad_mask = torch.ones(len(batch), n, dtype=torch.bool)  # True = padding

    for i, s in enumerate(batch):
        nn = s['num_patches']
        pad_imgs[i, :nn] = s['patch_images']
        pad_coords[i, :nn] = s['coords']
        pad_mask[i, :nn] = False  # not padding

    return (
        pad_imgs,       # [B, N, 3, H, W]
        pad_coords,     # [B, N, 2]
        labels,         # [B, 1]
        [s['slide_id'] for s in batch],
        pad_mask,       # [B, N]
    )
