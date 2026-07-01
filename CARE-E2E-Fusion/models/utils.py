"""工具函数"""
import torch
import torch.nn as nn
from typing import Dict


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total - trainable}


def print_model_summary(model: nn.Module, model_name: str = "E2E-ViT + CARE Fusion"):
    """打印模型摘要"""
    params = count_parameters(model)
    print("=" * 60)
    print(f"  {model_name}")
    print("=" * 60)
    print(f"  Total parameters:      {params['total']:,}")
    print(f"  Trainable parameters: {params['trainable']:,}")
    print(f"  Frozen parameters:    {params['frozen']:,}")
    print("=" * 60)
    print("\n  Module Breakdown:")
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        print(f"    {name:35s}: {mod_params:>12,}")
    print("=" * 60)


def get_default_device() -> torch.device:
    """获取默认设备"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def seed_everything(seed: int = 1):
    """设置所有随机种子"""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
