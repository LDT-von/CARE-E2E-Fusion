from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from pathway_morph_ot import PathwayMorphOT, PathwayMorphOTConfig


def main() -> None:
    cfg = PathwayMorphOTConfig(atom_dim=32, topology_dim=8, spatial_dim=4, pathway_dim=16, hidden_dim=24, num_tasks=2, ot_iters=20)
    model = PathwayMorphOT(cfg)
    out = model(
        region_embeddings=torch.randn(2, 5, 32),
        topology=torch.rand(2, 5, 8),
        spatial=torch.rand(2, 5, 4),
        pathway_tokens=torch.randn(2, 4, 16),
        labels=torch.randint(0, 2, (2, 2)).float(),
    )
    out["loss"].backward()
    grad = model.atom_encoder[1].weight.grad.abs().sum().item()
    print("project_2_smoke=ok")
    print(f"logits_shape={tuple(out['logits'].shape)}")
    print(f"transport_plan_shape={tuple(out['transport_plan'].shape)}")
    print(f"grad_l1={grad:.6f}")
    if grad <= 0:
        raise RuntimeError("Expected non-zero OT model gradient")


if __name__ == "__main__":
    main()
