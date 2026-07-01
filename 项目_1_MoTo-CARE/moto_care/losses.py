from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def topology_loss(pred: Tensor, target: Tensor) -> Tensor:
    value = F.smooth_l1_loss(pred, target)
    cdf = F.smooth_l1_loss(torch.cumsum(pred, dim=-1), torch.cumsum(target, dim=-1))
    return value + cdf


def molecular_contrastive_loss(evidence: Tensor, molecular: Tensor, temperature: float = 0.2) -> Tensor:
    bsz, num_tokens, dim = evidence.shape
    left = F.normalize(evidence.reshape(bsz * num_tokens, dim), dim=-1)
    right = F.normalize(molecular.reshape(bsz * num_tokens, dim), dim=-1)
    logits = left @ right.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)
