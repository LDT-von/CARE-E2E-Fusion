"""检查 PT feature 数值质量"""
import torch, numpy as np

# 随机选一个 PT
pt = r'E:\TCGA-data\CPathPatchFeature\blca\chief\pt_files\TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.pt'
d = torch.load(pt, map_location='cpu')
feats = d if isinstance(d, torch.Tensor) else (d.get('features', list(d.values())[0]) if isinstance(d, dict) else d)
feats = feats.float()

print(f"Shape: {feats.shape}")
print(f"dtype: {feats.dtype}")
print(f"Mean: {feats.mean():.4f}")
print(f"Std: {feats.std():.4f}")
print(f"Min: {feats.min():.4f}")
print(f"Max: {feats.max():.4f}")
print(f"Has NaN: {feats.isnan().any()}")
print(f"Has Inf: {feats.isinf().any()}")
print(f"Zeros: {(feats.abs() < 1e-6).sum().item()} / {feats.numel()}")
print(f"\n前5个 token 的 L2 norm:")
norms = feats.norm(dim=-1)
print(norms[:10])
print(f"\nNorm mean: {norms.mean():.4f}, std: {norms.std():.4f}")

# 这些是 CONCH / UNI features，应该有语义信息
# CONCH features: L2-normalized, mean ≈ 0.02, std ≈ 0.08
# UNI features: L2-normalized, mean ≈ 0.05, std ≈ 0.12
# 如果数值范围在 0-10 之间，可能是原始 (non-normalized) features
print(f"\n特征看起来是: ", end="")
if feats.std() < 0.2:
    print("L2-normalized features (CONCH/UNI style) - 合理 ✅")
elif feats.std() < 1.0:
    print("Partially normalized features - 可接受")
else:
    print("Raw unnormalized features - 需检查")
