"""CARE feature adapter for the E2E-ViT + CARE fusion project.

The original CARE preprocessing stores CONCH patch features and patch
coordinates in a single ``.npy`` dictionary. This adapter normalizes the common
CARE variants into the format expected by ``train.RealWSIDataset``:

    tile_tokens: [N, C] float tensor
    coords:      [N, 2] float tensor in roughly normalized slide coordinates
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def find_care_feature_file(data_root: str | Path, slide_id: str, tile_size: int = 256) -> Path:
    """Find a CARE feature file for one slide.

    Supported common layouts:
      - data_root/<slide_id>_0_<tile_size>.npy
      - data_root/<slide_id>_0_1024.npy
      - data_root/**/<slide_id>_*.npy
    """
    root = Path(data_root)
    candidates = [
        root / f"{slide_id}_0_{tile_size}.npy",
        root / f"{slide_id}_0_1024.npy",
        root / f"{slide_id}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path

    matches = sorted(root.rglob(f"{slide_id}_*.npy"))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No CARE feature file found for slide_id={slide_id!r} under {root}")


def load_care_feature(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load one CARE ``.npy`` feature file.

    CARE examples use keys named ``feature`` and ``index``. Some local exports
    use ``coords`` instead of ``index``; both are supported.
    """
    raw = np.load(path, allow_pickle=True)
    data = raw.item() if hasattr(raw, "item") else raw[()]

    features = data["feature"]
    if isinstance(features, list):
        features = np.concatenate(features, axis=0)
    features = np.asarray(features, dtype=np.float32)

    if "coords" in data:
        coords = np.asarray(data["coords"], dtype=np.float32)
    elif "index" in data:
        coords = _coords_from_index(data["index"])
    else:
        coords = np.zeros((features.shape[0], 2), dtype=np.float32)

    if coords.shape[0] != features.shape[0]:
        coords = coords[: features.shape[0]]

    coords = _normalize_coords(coords)
    return torch.from_numpy(features).float(), torch.from_numpy(coords).float()


def _coords_from_index(index_values) -> np.ndarray:
    coords = []
    for value in index_values:
        text = str(value)
        parts = text.replace("\\", "/").split("/")[-1].split("_")
        coords.append([float(parts[0]), float(parts[1])])
    return np.asarray(coords, dtype=np.float32)


def _normalize_coords(coords: np.ndarray) -> np.ndarray:
    if coords.size == 0:
        return coords.astype(np.float32)
    coords = coords.astype(np.float32)
    mins = coords.min(axis=0, keepdims=True)
    maxs = coords.max(axis=0, keepdims=True)
    denom = np.maximum(maxs - mins, 1.0)
    return (coords - mins) / denom
