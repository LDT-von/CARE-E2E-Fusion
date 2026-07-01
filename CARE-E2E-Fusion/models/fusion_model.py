"""
E2E-ViT + CARE Fusion: 融合模型完整实现
=========================================

核心设计：
  - 输入:   E2E-ViT 的固定网格 + 长条图
  - 编码:   Conv2d Patch Embedding (CONCH 预训练) + ALiBi 位置编码
  - 第一层:  可学习 Region Tokens 动态划分区域
  - 第二层:  CARE 的 ARM 区域注意力聚合
  - 预测:   多任务分子标志物预测头
  - 蒸馏:   保留 CONCH 的蒸馏损失

Author: carEtoE
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List


# ============================================================
# 位置编码工具
# ============================================================

def get_alibi_slopes(heads: int) -> Tensor:
    """生成 ALiBi 斜率系数。"""
    def get_slopes_power_of_2(n: int) -> List[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(heads) == int(math.log2(heads)):
        return Tensor(get_slopes_power_of_2(heads))
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return Tensor(
            get_slopes_power_of_2(closest_power_of_2)
            + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]
        )


def build_alibi_bias(
    seq_len: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """构建 ALiBi 注意力偏置矩阵 [num_heads, seq_len, seq_len]。"""
    slopes = get_alibi_slopes(num_heads).to(device)
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    diagonal = positions.unsqueeze(0) - positions.unsqueeze(1)
    alibi_bias = -torch.abs(diagonal.unsqueeze(0))
    return alibi_bias * slopes.view(-1, 1, 1)


# ============================================================
# Patch Embedding (来自 E2E-ViT)
# ============================================================

class PatchEmbed(nn.Module):
    """Conv2d 非重叠分块：将图像划分为 patch tokens。"""
    def __init__(self, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        H_p = x.shape[2]
        W_p = x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, H_p*W_p, embed_dim]
        return x, (H_p, W_p)


# ============================================================
# Patch Merger (来自 E2E-ViT)
# ============================================================

class PatchMerger(nn.Module):
    """将 patch tokens 聚合成 tile tokens（每个 tile 内部 mean pooling）。"""
    def __init__(self, patch_size: int = 16, tile_size: int = 256, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.tile_size = tile_size
        self.patches_per_tile = (tile_size // patch_size) ** 2
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: Tensor, num_tiles: int) -> Tensor:
        B, N_total, C = patch_tokens.shape
        tile_tokens = patch_tokens[:, :num_tiles * self.patches_per_tile, :]
        tile_tokens = tile_tokens.view(B, num_tiles, self.patches_per_tile, C)
        tile_tokens = tile_tokens.mean(dim=2)
        return self.norm(tile_tokens)


# ============================================================
# Transformer Block (来自 E2E-ViT)
# ============================================================

class E2ETransformerBlock(nn.Module):
    """带 ALiBi 的 Transformer Block。"""
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, alibi_bias: Optional[Tensor] = None) -> Tensor:
        # ALiBi shape: [num_heads, seq_len, seq_len]
        # 需要扩展到 [batch_size * num_heads, seq_len, seq_len] 才能正确广播
        if alibi_bias is not None:
            B = x.shape[0]
            alibi_bias = alibi_bias.unsqueeze(0).expand(B, -1, -1, -1)  # [B, num_heads, seq, seq]
            alibi_bias = alibi_bias.reshape(B * self.attn.num_heads, x.shape[1], x.shape[1])
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=alibi_bias)[0]
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 第一层自适应：动态区域划分（你的核心 idea）
# ============================================================

class DynamicRegionPartition(nn.Module):
    """动态区域划分：K 个可学习 Region Tokens 通过 Cross-Attention 决定 tile 归属。

    创新点：让区域划分本身可学习，不再是 CARE 那种固定 64×64 硬切。
    每个 region token 去"问"所有 tile："你属于我吗？"，从而得到大小可变的区域。
    """

    def __init__(self, embed_dim: int = 768, num_region_tokens: int = 8, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_region_tokens = num_region_tokens

        # K 个可学习的 region tokens
        self.region_tokens = nn.Parameter(torch.randn(1, num_region_tokens, embed_dim) * 0.02)

        # Cross-attention: region → tile
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_norm = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        # 可选：预测每个 region 覆盖的 tile 数量
        self.coverage_head = nn.Linear(embed_dim, 1)

    def forward(
        self,
        tile_tokens: Tensor,           # [B, N, C]
        coords: Optional[Tensor] = None,  # [B, N, 2] 归一化坐标
        return_coverage: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        B, N, C = tile_tokens.shape
        K = self.num_region_tokens

        # 扩展 region tokens 到 batch size
        region_tokens = self.region_tokens.expand(B, -1, -1)  # [B, K, C]

        # Cross-Attention: Region Tokens (Q) ← Tile Tokens (K, V)
        region_out, attn_weights = self.cross_attn(
            query=region_tokens,
            key=tile_tokens,
            value=tile_tokens,
        )
        region_tokens = self.cross_norm(region_tokens + region_out)

        # FFN
        region_out = self.ffn(region_tokens)
        region_features = self.ffn_norm(region_tokens + region_out)  # [B, K, C]

        outputs = [region_features, attn_weights]

        if return_coverage:
            coverage_logits = self.coverage_head(region_features).squeeze(-1)
            coverage = torch.sigmoid(coverage_logits) * N
            outputs.append(coverage)
        else:
            outputs.append(None)

        return tuple(outputs)


# ============================================================
# 第二层自适应：ARM 区域注意力聚合（来自 CARE）
# ============================================================

class AdaptiveRegionModeling(nn.Module):
    """CARE 的 ARM 模块：对每个区域内的 tile 做注意力聚合。

    流程：
    1. 每个 region token 做 tile-to-region cross-attention（加权平均 tile 特征）
    2. 区域之间做 self-attention（捕捉区域间关系，如肿瘤区↔间质区）
    3. FFN + 归一化
    4. 输出区域级特征
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Tile → Region cross-attention
        self.tile_to_region_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.tile_norm = nn.LayerNorm(embed_dim)

        # Region self-attention（区域间交互）
        self.region_self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.region_norm = nn.LayerNorm(embed_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)

        self.region_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        region_features: Tensor,   # [B, K, C] 来自第一层
        tile_tokens: Tensor,        # [B, N, C] 所有 tile
        attn_weights: Tensor,       # [B, K, N] region 对 tile 的注意力
        mask_threshold: float = 0.01,
    ) -> Tuple[Tensor, Tensor]:
        B, K, C = region_features.shape
        N = tile_tokens.shape[1]

        # 屏蔽低注意力权重的 tile
        mask = (attn_weights < mask_threshold).float() * (-1e9)

        # 加权聚合 tile 特征
        attn_weighted = F.softmax(attn_weights + mask, dim=-1)  # [B, K, N]
        region_aggregated = torch.bmm(attn_weighted, tile_tokens)  # [B, K, C]

        # Tile-to-region cross-attention
        # Tile-to-region cross-attention
        # mask shape: [B, K, N] -> [B*num_heads, K, N]
        if mask is not None and mask.abs().max() < 1e8:
            B, K, N = mask.shape
            mask_expanded = mask.unsqueeze(1).expand(B, self.tile_to_region_attn.num_heads, K, N)
            mask_expanded = mask_expanded.reshape(B * self.tile_to_region_attn.num_heads, K, N)
        else:
            mask_expanded = None
        tile_out, _ = self.tile_to_region_attn(
            query=region_features,
            key=tile_tokens,
            value=tile_tokens,
            attn_mask=mask_expanded,
        )
        region_features = self.tile_norm(region_features + tile_out)

        # Region self-attention（区域间关系）
        region_out, _ = self.region_self_attn(
            query=region_features,
            key=region_features,
            value=region_features,
        )
        region_features = self.region_norm(region_features + region_out)

        # FFN
        region_out = self.ffn(region_features)
        region_embeddings = self.ffn_norm(region_features + region_out)
        region_embeddings = self.region_proj(region_embeddings)  # [B, K, C]

        # WSI 级别：所有区域平均
        region_pooled = region_embeddings.mean(dim=1)  # [B, C]

        return region_embeddings, region_pooled


# ============================================================
# 蒸馏损失（来自 CARE）
# ============================================================

class DistillationLoss(nn.Module):
    """知识蒸馏：让学生模型的 tile tokens 学习教师模型（CONCH）的行为。"""

    def __init__(self, embed_dim: int = 768, temperature: float = 3.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_tile_tokens: Tensor,   # [B, N, C]
        teacher_tile_tokens: Tensor,   # [B, N, C_tea]
    ) -> Tuple[Tensor, Dict[str, float]]:
        T = self.temperature

        # Tile 级别 KL 散度
        s = student_tile_tokens / T
        t = teacher_tile_tokens / T

        distill_loss = F.kl_div(
            F.log_softmax(s, dim=-1),
            F.softmax(t, dim=-1),
            reduction='batchmean',
        ) * (T ** 2)

        return distill_loss, {'distill_loss': distill_loss.item()}


# ============================================================
# 多任务预测头（来自 CARE）
# ============================================================

class MultiTaskHead(nn.Module):
    """多任务分子标志物预测头，支持阈值化预测（来自 CARE）。"""

    def __init__(
        self,
        embed_dim: int = 768,
        num_tasks: int = 4,
        task_names: Optional[List[str]] = None,
        dropout: float = 0.25,
        use_threshold: bool = True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]
        self.use_threshold = use_threshold

        self.shared = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.task_heads = nn.ModuleList([nn.Linear(embed_dim, 1) for _ in range(num_tasks)])

        if use_threshold:
            self.task_thresholds = nn.Parameter(torch.zeros(num_tasks), requires_grad=True)

    def forward(self, wsi_features: Tensor) -> Dict[str, Tensor]:
        x = self.shared(wsi_features)
        logits = torch.cat([head(x) for head in self.task_heads], dim=-1)  # [B, num_tasks]
        probs = torch.sigmoid(logits)

        if self.use_threshold:
            thresholds = torch.sigmoid(self.task_thresholds) * 0.5
            preds = (probs > thresholds).float()
        else:
            preds = (probs > 0.5).float()

        return {'logits': logits, 'probs': probs, 'preds': preds}


# ============================================================
# 完整融合模型
# ============================================================

class E2EViTCAREFusion(nn.Module):
    """E2E-ViT + CARE 融合模型：双层自适应区域建模。

    流程:
      WSI 图像 [B, 3, H, W*N_tiles]
          ↓
      Patch Embedding (Conv2d) → Patch Tokens [B, N_patches, C]
          ↓
      Patch Merger → Tile Tokens [B, N_tiles, C]
          ↓
      ALiBi 位置编码
          ↓
      Transformer Backbone → Encoded Tokens [B, N_tiles, C]
          ↓
      ┌──────────────────────────────────────────────────────┐
      │ 分支1 (直接): Mean Pool → FC → 直接 logits          │
      │ 分支2 (双层自适应):                                │
      │   第一层: DynamicRegionPartition → K 个区域特征      │
      │   第二层: ARM → 区域聚合特征                        │
      │   → MultiTaskHead → 自适应 logits                  │
      └──────────────────────────────────────────────────────┘
          ↓
      总损失 = 任务损失(直接) + 任务损失(自适应) + 蒸馏损失
    """

    def __init__(
        self,
        # 输入配置
        tile_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        # 编码器配置
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        # 区域建模配置
        num_region_tokens: int = 8,
        # 预测配置
        num_tasks: int = 4,
        task_names: Optional[List[str]] = None,
        # 损失权重
        task_loss_weight: float = 0.1,
        distill_weight: float = 0.5,
        # 其他选项
        use_alibi: bool = True,
        use_distillation: bool = True,
        use_two_branches: bool = True,
        # 如果传入 pretrained_vit_state_dict，就加载预训练权重
        pretrained_state_dict: Optional[Dict] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_region_tokens = num_region_tokens
        self.num_tasks = num_tasks
        self.task_names = task_names
        self.task_loss_weight = task_loss_weight
        self.distill_weight = distill_weight
        self.use_alibi = use_alibi
        self.use_distillation = use_distillation
        self.use_two_branches = use_two_branches

        # Patch Embedding
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)
        self.patch_merger = PatchMerger(patch_size, tile_size, embed_dim)
        self.tile_size = tile_size
        self.patch_size = patch_size
        self.patches_per_tile = (tile_size // patch_size) ** 2

        # Transformer Backbone
        self.blocks = nn.ModuleList([
            E2ETransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        # 分支1: 直接分类
        self.head_direct = nn.Linear(embed_dim, num_tasks)

        # 第一层自适应: 动态区域划分
        self.dynamic_region_partition = DynamicRegionPartition(
            embed_dim, num_region_tokens, num_heads, dropout
        )

        # 第二层自适应: ARM
        self.arm = AdaptiveRegionModeling(embed_dim, num_heads, dropout)

        # 分支2: 自适应分类
        self.head_adaptive = MultiTaskHead(
            embed_dim, num_tasks, task_names, dropout, use_threshold=True
        )

        # 蒸馏
        if use_distillation:
            self.distillation = DistillationLoss(embed_dim, temperature=3.0, alpha=0.5)

        # 特征融合（两个分支）
        if use_two_branches:
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

        # 加载预训练权重（可选）
        if pretrained_state_dict is not None:
            self.load_state_dict(pretrained_state_dict, strict=False)

    def load_pretrained_vit(self, state_dict: Dict):
        """从预训练 ViT 加载权重到 patch_embed 和 transformer blocks。"""
        loaded_keys = []
        for key, value in state_dict.items():
            if 'patch_embed' in key or 'blocks' in key:
                target_key = key.replace('vit.', '').replace('model.', '')
                if hasattr(self, target_key.split('.')[0]):
                    try:
                        target = self
                        for part in target_key.split('.'):
                            target = getattr(target, part)
                        if target.shape == value.shape:
                            target.load_state_dict(value if isinstance(value, dict) else {k: v for k, v in [value]}, strict=False)
                            loaded_keys.append(key)
                    except (AttributeError, TypeError):
                        pass

        if not loaded_keys:
            # 简单策略：直接跳过 embedding 层，从 Transformer block 开始加载
            self._load_transformer_blocks(state_dict)

    def _load_transformer_blocks(self, state_dict: Dict):
        """尝试加载 Transformer block 权重。"""
        loaded = 0
        for key, value in state_dict.items():
            if 'blocks.' in key:
                try:
                    block_idx = int(key.split('blocks.')[1].split('.')[0])
                    if block_idx < len(self.blocks):
                        attr_path = key.replace(f'blocks.{block_idx}.', '')
                        target = self.blocks[block_idx]
                        for part in attr_path.split('.'):
                            target = getattr(target, part)
                        if hasattr(value, 'shape') and target.shape == value.shape:
                            target.data = value.data
                            loaded += 1
                except (IndexError, AttributeError):
                    pass
        print(f"Loaded {loaded} Transformer block parameters from pretrained model.")

    def forward(
        self,
        strip_image: Tensor,                  # [B, 3, H, W_total] 长条图
        coords: Optional[Tensor] = None,       # [B, N, 2] tile 坐标
        labels: Optional[Tensor] = None,       # [B, num_tasks] 标签
        teacher_tile_tokens: Optional[Tensor] = None,  # [B, N, C] 蒸馏教师
    ) -> Dict[str, Tensor]:
        B, C_img, H, W_total = strip_image.shape

        # ---- Patch Embedding ----
        patch_tokens, grid_size = self.patch_embed(strip_image)  # [B, N_patches, C]
        H_p, W_p = grid_size  # H_p = H/patch_size, W_p = W_total/patch_size
        N_patches_per_row = W_p  # 每行的 patch 数量（等于 W_p，因为 H 方向已经固定为 tile_size）

        # 每个 tile 占 W_p 中的多少列
        patches_per_tile_w = self.tile_size // self.patch_size  # = tile_size/patch_size = 16
        N_tiles = W_p // patches_per_tile_w  # 沿 W 方向的 tile 数（H 方向只有 1 个 tile 高）

        # ---- Patch Merger ----
        tile_tokens = self.patch_merger(patch_tokens, N_tiles)  # [B, N_tiles, C]

        # ---- ALiBi 位置编码 ----
        alibi_bias = None
        if self.use_alibi:
            alibi_bias = build_alibi_bias(
                seq_len=tile_tokens.shape[1],
                num_heads=self.blocks[0].attn.num_heads,
                device=tile_tokens.device,
                dtype=tile_tokens.dtype,
            )

        # ---- Transformer Backbone ----
        x = tile_tokens
        for block in self.blocks:
            x = block(x, alibi_bias=alibi_bias)  # [B, N_tiles, C]

        # ---- 分支1: 直接分类 (E2E-ViT style) ----
        x_global = x.mean(dim=1)  # [B, C]
        logits_direct = self.head_direct(x_global)  # [B, num_tasks]

        outputs = {
            'tile_tokens': tile_tokens,
            'transformer_output': x,
            'logits_direct': logits_direct,
        }

        total_loss = 0.0
        loss_dict = {}

        if labels is not None:
            loss_direct = F.binary_cross_entropy_with_logits(logits_direct, labels)
            total_loss = total_loss + loss_direct
            loss_dict['loss_direct'] = loss_direct.item()

        # ---- 分支2: 双层自适应 (CARE style) ----
        if self.use_two_branches:
            # 第一层: 动态区域划分
            region_features, attn_weights, coverage = self.dynamic_region_partition(
                tile_tokens=x,
                coords=coords,
                return_coverage=True,
            )

            # 第二层: ARM 区域聚合
            region_embeddings, region_pooled = self.arm(
                region_features=region_features,
                tile_tokens=x,
                attn_weights=attn_weights,
            )

            # 多任务预测
            adaptive_output = self.head_adaptive(region_pooled)
            logits_adaptive = adaptive_output['logits']  # [B, num_tasks]

            outputs['region_features'] = region_features
            outputs['region_embeddings'] = region_embeddings
            outputs['attn_weights'] = attn_weights
            outputs['coverage'] = coverage
            outputs['logits_adaptive'] = logits_adaptive
            outputs['adaptive_probs'] = adaptive_output['probs']
            outputs['adaptive_preds'] = adaptive_output['preds']

            if labels is not None:
                loss_adaptive = F.binary_cross_entropy_with_logits(logits_adaptive, labels)
                total_loss = total_loss + loss_adaptive
                loss_dict['loss_adaptive'] = loss_adaptive.item()

                # 分支一致性损失（让两个分支的预测接近）
                loss_fusion = F.mse_loss(
                    torch.sigmoid(logits_direct),
                    torch.sigmoid(logits_adaptive),
                )
                total_loss = total_loss + 0.1 * loss_fusion
                loss_dict['loss_fusion'] = loss_fusion.item()

        # ---- 蒸馏损失 ----
        if self.use_distillation and teacher_tile_tokens is not None:
            distill_loss, distill_dict = self.distillation(
                student_tile_tokens=tile_tokens,
                teacher_tile_tokens=teacher_tile_tokens,
            )
            total_loss = total_loss + self.distill_weight * distill_loss
            loss_dict['loss_distill'] = distill_loss.item()

        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict

        return outputs

    def forward_wsi(
        self,
        images: Tensor,           # [B*N, 3, H, W] tiles (N tiles per WSI)
        tile_coords: Tensor,       # [B, N, 2]
        num_tiles: int,           # N
        batch_size: int,          # B
        labels: Optional[Tensor] = None,
        teacher_tile_tokens: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """处理 WSI tiles 的前向传播：将离散的 tiles reshape 成 strip image 后调用 forward。

        Args:
            images: [B*N, 3, H, W] - N tiles per WSI stacked along batch dim
            tile_coords: [B, N, 2] - 归一化坐标
            num_tiles: N - 每张 WSI 的 tile 数
            batch_size: B
            labels: [B, num_tasks]
            teacher_tile_tokens: [B, N, C]
        """
        B, N = batch_size, num_tiles
        C_img, H_tile, W_tile = images.shape[1:]

        # Reshape: [B*N, 3, H, W] -> [B, 3, H, W*N]
        images_per_wsi = images.reshape(B, N, C_img, H_tile, W_tile)
        # Permute: [B, N, 3, H, W] -> [B, 3, H, N*W]
        strip_images = images_per_wsi.permute(0, 2, 3, 1, 4).contiguous()
        strip_images = strip_images.reshape(B, C_img, H_tile, W_tile * N)

        return self.forward(
            strip_image=strip_images,
            coords=tile_coords,
            labels=labels,
            teacher_tile_tokens=teacher_tile_tokens,
        )

    def forward_wsi_direct(
        self,
        images: Tensor,           # [B*N, 3, H, W] tiles (N tiles per WSI)
        tile_coords: Tensor,       # [B, N, 2]
        num_tiles: int,           # N
        batch_size: int,          # B
        labels: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """直接前向传播（绕过 PatchMerger）：每张 tile 直接投影为 token。

        适用于：预先提取的离散 tiles（如 CONCH 特征对应的 tiles）。
        序列长度 = N（不是 N * patches_per_tile）。

        Args:
            images: [B*N, 3, H, W]
            tile_coords: [B, N, 2]
            num_tiles: N
            batch_size: B
            labels: [B, num_tasks]
        """
        B, N = batch_size, num_tiles
        C_img, H_tile, W_tile = images.shape[1:]

        # Patch Embedding: [B*N, 3, H, W] -> [B*N, embed_dim, H/p, W/p] -> [B*N, N_p, embed_dim]
        patch_tokens, grid_size = self.patch_embed(images)  # patch_tokens: [B*N, N_patches_each, C]
        H_p, W_p = grid_size
        N_patches_each = H_p * W_p

        # 每个 WSI 内的 tile 按行拼接：[B*N, N_p, C] -> [B, N*N_p, C]
        patch_tokens = patch_tokens.reshape(B, N * N_patches_each, -1)  # [B, N*N_p, C]


        # 用 tile mean 作为每个 tile 的最终 token（等价于 PatchMerger 但可处理不规则 tiles）
        tile_tokens = patch_tokens.reshape(B, N, N_patches_each, -1).mean(dim=2)  # [B, N, C]

        # ALiBi 位置编码
        alibi_bias = None
        if self.use_alibi:
            alibi_bias = build_alibi_bias(
                seq_len=tile_tokens.shape[1],
                num_heads=self.blocks[0].attn.num_heads,
                device=tile_tokens.device,
                dtype=tile_tokens.dtype,
            )

        # Transformer Backbone
        x = tile_tokens
        for block in self.blocks:
            x = block(x, alibi_bias=alibi_bias)  # [B, N, C]

        # 分支1: 直接分类
        x_global = x.mean(dim=1)  # [B, C]
        logits_direct = self.head_direct(x_global)  # [B, num_tasks]

        outputs = {
            'tile_tokens': tile_tokens,
            'transformer_output': x,
            'logits_direct': logits_direct,
        }

        total_loss = 0.0
        loss_dict = {}

        if labels is not None:
            loss_direct = F.binary_cross_entropy_with_logits(logits_direct, labels)
            total_loss = total_loss + loss_direct
            loss_dict['loss_direct'] = loss_direct.item()

        # 分支2: 双层自适应
        if self.use_two_branches:
            region_features, attn_weights, coverage = self.dynamic_region_partition(
                tile_tokens=x,
                coords=tile_coords,
                return_coverage=True,
            )
            region_embeddings, region_pooled = self.arm(
                region_features=region_features,
                tile_tokens=x,
                attn_weights=attn_weights,
            )
            adaptive_output = self.head_adaptive(region_pooled)
            logits_adaptive = adaptive_output['logits']

            outputs['region_features'] = region_features
            outputs['region_embeddings'] = region_embeddings
            outputs['attn_weights'] = attn_weights
            outputs['coverage'] = coverage
            outputs['logits_adaptive'] = logits_adaptive
            outputs['adaptive_probs'] = adaptive_output['probs']
            outputs['adaptive_preds'] = adaptive_output['preds']

            if labels is not None:
                loss_adaptive = F.binary_cross_entropy_with_logits(logits_adaptive, labels)
                total_loss = total_loss + loss_adaptive
                loss_dict['loss_adaptive'] = loss_adaptive.item()
                loss_fusion = F.mse_loss(
                    torch.sigmoid(logits_direct),
                    torch.sigmoid(logits_adaptive),
                )
                total_loss = total_loss + 0.1 * loss_fusion
                loss_dict['loss_fusion'] = loss_fusion.item()

        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict
        return outputs

