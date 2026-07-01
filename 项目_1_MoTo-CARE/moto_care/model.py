from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .losses import molecular_contrastive_loss, topology_loss
from .topology import grid_anchors, normalize_coords


@dataclass
class MoToCAREConfig:
    input_dim: int = 768
    embed_dim: int = 256
    num_regions: int = 8
    num_heads: int = 4
    num_tasks: int = 1
    topology_dim: int = 12
    molecule_dim: int = 128
    top_k_regions: int = 4
    assignment_temperature: float = 0.35
    topology_weight: float = 0.5
    molecular_weight: float = 0.2
    entropy_weight: float = 0.01
    dropout: float = 0.1
    label_smoothing: float = 0.1


class TopologyAwareAssignment(nn.Module):
    def __init__(self, cfg: MoToCAREConfig):
        super().__init__()
        self.cfg = cfg
        self.region_queries = nn.Parameter(torch.randn(cfg.num_regions, cfg.embed_dim) * 0.02)
        self.topology_proj = nn.Linear(cfg.topology_dim, cfg.embed_dim)
        self.spatial_scale = nn.Parameter(torch.tensor(6.0))
        self.semantic_scale = nn.Parameter(torch.tensor(1.0))
        self.topology_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, tokens: Tensor, coords: Tensor, topology_prior: Optional[Tensor], padding_mask: Optional[Tensor]):
        bsz, _, _ = tokens.shape
        coords = normalize_coords(coords, padding_mask)
        anchors = grid_anchors(self.cfg.num_regions, tokens.device, tokens.dtype)
        spatial = -self.spatial_scale.abs() * torch.cdist(coords, anchors.unsqueeze(0).expand(bsz, -1, -1)).square()
        semantic = self.semantic_scale * torch.einsum(
            "bnd,kd->bnk", F.normalize(tokens, dim=-1), F.normalize(self.region_queries, dim=-1)
        )
        if topology_prior is None:
            topo = torch.zeros_like(semantic)
        else:
            topo_keys = F.normalize(self.topology_proj(topology_prior), dim=-1)
            topo = self.topology_scale * torch.einsum("bnd,bkd->bnk", F.normalize(tokens, dim=-1), topo_keys)
        score = spatial + semantic + topo
        if 0 < self.cfg.top_k_regions < self.cfg.num_regions:
            idx = score.topk(self.cfg.top_k_regions, dim=-1).indices
            keep = torch.zeros_like(score, dtype=torch.bool).scatter_(-1, idx, True)
            score = score.masked_fill(~keep, -1e4)
        if padding_mask is not None:
            score = score.masked_fill(padding_mask.unsqueeze(-1), -1e4)
        assignment = F.softmax(score / self.cfg.assignment_temperature, dim=-1)
        if padding_mask is not None:
            assignment = assignment.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        mass = assignment.sum(dim=1).clamp_min(1e-6)
        regions = torch.einsum("bnk,bnd->bkd", assignment, tokens) / mass.unsqueeze(-1)
        return regions, assignment


class MoToCARE(nn.Module):
    def __init__(self, cfg: MoToCAREConfig):
        super().__init__()
        self.cfg = cfg
        self.feature_proj = nn.Sequential(nn.LayerNorm(cfg.input_dim), nn.Linear(cfg.input_dim, cfg.embed_dim), nn.GELU())
        self.assignment = TopologyAwareAssignment(cfg)
        self.region_attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True, dropout=cfg.dropout)
        self.region_norm = nn.LayerNorm(cfg.embed_dim)
        self.topology_head = nn.Sequential(nn.LayerNorm(cfg.embed_dim), nn.Linear(cfg.embed_dim, cfg.topology_dim))
        self.gate = nn.Linear(cfg.embed_dim, 1)
        self.head = nn.Sequential(nn.LayerNorm(cfg.embed_dim), nn.Linear(cfg.embed_dim, cfg.num_tasks))
        self.mol_proj = nn.Linear(cfg.molecule_dim, cfg.embed_dim)

    def forward(
        self,
        features: Tensor,
        coords: Tensor,
        labels: Optional[Tensor] = None,
        topology_prior: Optional[Tensor] = None,
        topology_target: Optional[Tensor] = None,
        molecule_tokens: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        tokens = self.feature_proj(features)
        regions, assignment = self.assignment(tokens, coords, topology_prior, padding_mask)
        regions = self.region_norm(regions + self.region_attn(regions, regions, regions)[0])
        topology_pred = self.topology_head(regions)
        gates = F.softmax(self.gate(regions).squeeze(-1), dim=-1)
        slide = torch.einsum("bk,bkd->bd", gates, regions)
        logits = self.head(slide)
        losses = {}
        if labels is not None:
            if self.cfg.label_smoothing > 0:
                smooth = labels.float() * (1 - self.cfg.label_smoothing) + 0.5 * self.cfg.label_smoothing
                losses["task"] = F.binary_cross_entropy_with_logits(logits, smooth)
            else:
                losses["task"] = F.binary_cross_entropy_with_logits(logits, labels.float())
        if topology_target is not None:
            losses["topology"] = topology_loss(topology_pred, topology_target)
        if molecule_tokens is not None:
            mol = self.mol_proj(molecule_tokens)
            align = F.softmax(torch.einsum("bld,bkd->blk", mol, regions) / (regions.shape[-1] ** 0.5), dim=-1)
            evidence = torch.einsum("blk,bkd->bld", align, regions)
            losses["molecular"] = molecular_contrastive_loss(evidence, mol)
        token_entropy = -(assignment.clamp_min(1e-8) * assignment.clamp_min(1e-8).log()).sum(-1)
        if padding_mask is not None:
            entropy = token_entropy.masked_fill(padding_mask, 0.0).sum() / (~padding_mask).sum().clamp_min(1)
        else:
            entropy = token_entropy.mean()
        total = losses.get("task", torch.zeros((), device=features.device))
        total = total + self.cfg.entropy_weight * entropy
        if "topology" in losses:
            total = total + self.cfg.topology_weight * losses["topology"]
        if "molecular" in losses:
            total = total + self.cfg.molecular_weight * losses["molecular"]
        return {
            "logits": logits,
            "probs": torch.sigmoid(logits),
            "assignment": assignment,
            "region_tokens": regions,
            "topology_pred": topology_pred,
            "loss": total,
            "losses": losses,
        }
