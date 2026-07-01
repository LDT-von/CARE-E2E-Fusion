from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def grid_anchors(num_regions: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    side = int(math.ceil(math.sqrt(num_regions)))
    axis = torch.linspace(0.0, 1.0, side, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)[:num_regions]


def normalize_coords(coords: Tensor, padding_mask: Tensor | None = None) -> Tensor:
    if padding_mask is None:
        mins = coords.amin(dim=-2, keepdim=True)
        maxs = coords.amax(dim=-2, keepdim=True)
    else:
        valid = ~padding_mask.unsqueeze(-1)
        large = torch.finfo(coords.dtype).max / 4
        small = torch.finfo(coords.dtype).min / 4
        mins = coords.masked_fill(~valid, large).amin(dim=-2, keepdim=True)
        maxs = coords.masked_fill(~valid, small).amax(dim=-2, keepdim=True)
        no_valid = ~valid.any(dim=-2, keepdim=True)
        mins = torch.where(no_valid, torch.zeros_like(mins), mins)
        maxs = torch.where(no_valid, torch.ones_like(maxs), maxs)
    return (coords - mins) / (maxs - mins).clamp_min(1.0)


def cluster_topology_prior(coords: Tensor, cluster_ids: Tensor, num_regions: int, num_clusters: int) -> Tensor:
    if coords.ndim == 2:
        coords = coords.unsqueeze(0)
    if cluster_ids.ndim == 1:
        cluster_ids = cluster_ids.unsqueeze(0)
    coords = normalize_coords(coords)
    anchors = grid_anchors(num_regions, coords.device, coords.dtype)
    nearest = torch.cdist(coords, anchors.unsqueeze(0).expand(coords.shape[0], -1, -1)).argmin(dim=-1)
    rows = []
    for b in range(coords.shape[0]):
        reg = []
        for k in range(num_regions):
            mask = nearest[b] == k
            if mask.any():
                hist = F.one_hot(cluster_ids[b, mask].clamp(0, num_clusters - 1).long(), num_clusters).float().mean(0)
                pts = coords[b, mask]
                moments = torch.cat([pts.mean(0), pts.std(0, unbiased=False)], dim=0)
            else:
                hist = torch.zeros(num_clusters, device=coords.device)
                moments = torch.zeros(4, device=coords.device)
            reg.append(torch.cat([hist.to(coords.dtype), moments.to(coords.dtype)], dim=0))
        rows.append(torch.stack(reg, dim=0))
    return torch.stack(rows, dim=0)
