from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from moto_care import MoToCARE, MoToCAREConfig


def main() -> None:
    cfg = MoToCAREConfig(input_dim=64, embed_dim=32, num_regions=6, num_heads=4, num_tasks=2, topology_dim=10, molecule_dim=16)
    model = MoToCARE(cfg)
    batch = {
        "features": torch.randn(2, 40, 64),
        "coords": torch.rand(2, 40, 2),
        "labels": torch.randint(0, 2, (2, 2)).float(),
        "topology_prior": torch.rand(2, 6, 10),
        "topology_target": torch.rand(2, 6, 10),
        "molecule_tokens": torch.randn(2, 3, 16),
    }
    out = model(**batch)
    out["loss"].backward()
    grad = model.assignment.region_queries.grad.abs().sum().item()
    print("project_1_smoke=ok")
    print(f"logits_shape={tuple(out['logits'].shape)}")
    print(f"assignment_shape={tuple(out['assignment'].shape)}")
    print(f"grad_l1={grad:.6f}")
    if grad <= 0:
        raise RuntimeError("Expected non-zero assignment gradient")


if __name__ == "__main__":
    main()
