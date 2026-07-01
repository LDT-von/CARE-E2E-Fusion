from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ot import sinkhorn_unbalanced


@dataclass
class PathwayMorphOTConfig:
    atom_dim: int = 256
    topology_dim: int = 12
    spatial_dim: int = 4
    pathway_dim: int = 64
    hidden_dim: int = 128
    num_tasks: int = 1
    epsilon: float = 0.08
    tau: float = 0.8
    ot_iters: int = 40
    ot_cost_weight: float = 0.05
    entropy_weight: float = 0.005
    label_smoothing: float = 0.1


class PathwayMorphOT(nn.Module):
    def __init__(self, cfg: PathwayMorphOTConfig):
        super().__init__()
        self.cfg = cfg
        self.atom_encoder = nn.Sequential(
            nn.LayerNorm(cfg.atom_dim + cfg.topology_dim + cfg.spatial_dim),
            nn.Linear(cfg.atom_dim + cfg.topology_dim + cfg.spatial_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.pathway_encoder = nn.Sequential(
            nn.LayerNorm(cfg.pathway_dim),
            nn.Linear(cfg.pathway_dim, cfg.hidden_dim),
            nn.GELU(),
        )
        self.topology_to_hidden = nn.Linear(cfg.topology_dim, cfg.hidden_dim)
        self.fuse = nn.Sequential(nn.Linear(cfg.hidden_dim * 2 + 1, cfg.hidden_dim), nn.GELU())
        self.head = nn.Linear(cfg.hidden_dim, cfg.num_tasks)

    def forward(
        self,
        region_embeddings: Tensor,
        topology: Tensor,
        spatial: Tensor,
        pathway_tokens: Tensor,
        labels: Optional[Tensor] = None,
        mu_region: Optional[Tensor] = None,
        mu_pathway: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        atoms = torch.cat([region_embeddings, topology, spatial], dim=-1)
        za = self.atom_encoder(atoms)
        zg = self.pathway_encoder(pathway_tokens)

        base_cost = torch.cdist(za, zg).square()
        topo_summary = F.normalize(self.topology_to_hidden(topology), dim=-1)
        pathway_summary = F.normalize(zg, dim=-1)
        topo_cost = 1.0 - torch.einsum("bkh,blh->bkl", topo_summary, pathway_summary)
        cost = base_cost + 0.2 * topo_cost

        bsz, num_regions, _ = za.shape
        num_pathways = zg.shape[1]
        if mu_region is None:
            mu_region = torch.full((bsz, num_regions), 1.0 / num_regions, device=za.device)
        if mu_pathway is None:
            mu_pathway = torch.full((bsz, num_pathways), 1.0 / num_pathways, device=za.device)

        plan = sinkhorn_unbalanced(cost, mu_region, mu_pathway, self.cfg.epsilon, self.cfg.tau, self.cfg.ot_iters)
        pair_features = torch.cat(
            [
                za.unsqueeze(2).expand(-1, -1, num_pathways, -1),
                zg.unsqueeze(1).expand(-1, num_regions, -1, -1),
                cost.unsqueeze(-1),
            ],
            dim=-1,
        )
        fused_pairs = self.fuse(pair_features)
        patient = (plan.unsqueeze(-1) * fused_pairs).sum(dim=(1, 2))
        logits = self.head(patient)

        losses = {}
        if labels is not None:
            if self.cfg.label_smoothing > 0:
                smooth = labels.float() * (1 - self.cfg.label_smoothing) + 0.5 * self.cfg.label_smoothing
                losses["task"] = F.binary_cross_entropy_with_logits(logits, smooth)
            else:
                losses["task"] = F.binary_cross_entropy_with_logits(logits, labels.float())
        losses["ot_cost"] = (plan * cost).sum(dim=(1, 2)).mean()
        entropy = -(plan.clamp_min(1e-8) * plan.clamp_min(1e-8).log()).sum(dim=(1, 2)).mean()
        losses["transport_entropy"] = entropy
        total = losses.get("task", torch.zeros((), device=logits.device))
        total = total + self.cfg.ot_cost_weight * losses["ot_cost"] - self.cfg.entropy_weight * entropy
        return {"logits": logits, "probs": torch.sigmoid(logits), "transport_plan": plan, "cost": cost, "loss": total, "losses": losses}
