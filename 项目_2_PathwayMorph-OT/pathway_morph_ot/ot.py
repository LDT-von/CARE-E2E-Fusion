from __future__ import annotations

import torch
from torch import Tensor


def sinkhorn_unbalanced(cost: Tensor, mu_a: Tensor, mu_g: Tensor, epsilon: float = 0.08, tau: float = 0.8, iters: int = 40) -> Tensor:
    """Unbalanced Sinkhorn in scaling form.

    cost: [B, K, L], mu_a: [B, K], mu_g: [B, L].
    """
    cost = cost - cost.amin(dim=(1, 2), keepdim=True)
    kernel = torch.exp((-cost / epsilon).clamp(min=-50.0, max=50.0)).clamp_min(1e-12)
    u = torch.ones_like(mu_a)
    v = torch.ones_like(mu_g)
    power = tau / (tau + epsilon)
    for _ in range(iters):
        kv = torch.einsum("bkl,bl->bk", kernel, v).clamp_min(1e-12)
        u = (mu_a / kv).clamp_min(1e-12).pow(power)
        ktu = torch.einsum("bkl,bk->bl", kernel, u).clamp_min(1e-12)
        v = (mu_g / ktu).clamp_min(1e-12).pow(power)
    return u.unsqueeze(-1) * kernel * v.unsqueeze(1)
