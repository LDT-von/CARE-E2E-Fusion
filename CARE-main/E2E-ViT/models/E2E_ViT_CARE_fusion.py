"""
E2E-ViT + CARE Fusion: Dual-Adaptive Region Modeling for End-to-End WSI Analysis

融合策略：
  - 输入:  E2E-ViT 的固定网格 + 长条图设计
  - 编码:  CONCH Patch Embedding + ALiBi 位置编码
  - 第一层自适应: 可学习 Region Tokens 做动态区域划分
  - 第二层自适应: CARE 的 ARM 区域注意力聚合
  - 预测:  CARE 的多任务分子标志物预测头
  - 蒸馏:  保留 CONCH tile 编码的蒸馏损失

Author: Your Name
Date: 2026
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict, List


# ============================================================
# ALiBi Positional Encoding (from E2E-ViT)
# ============================================================

def get_alibi_slopes(heads: int) -> Tensor:
    """Generate ALiBi slopes for multi-head attention.
    
    Following the original ALiBi paper: 
    https://arxiv.org/abs/2108.12409
    """
    def get_slopes_power_of_2(n: int) -> List[float]:
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if math.log2(heads) == int(math.log2(heads)):
        return Tensor(get_slopes_power_of_2(heads))
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][:heads - closest_power_of_2]
        )


def build_alibi_bias(
    seq_len: int,
    num_heads: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build ALiBi attention bias matrix.
    
    Returns shape [num_heads, seq_len, seq_len].
    """
    alibi slopes = get_alibi_slopes(num_heads).to(device)
    # Create distance matrix: [seq_len, seq_len]
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    diagonal = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
    alibi_bias = -torch.abs(diagonal.unsqueeze(0))  # [1, seq_len, seq_len]
    # Scale by slopes: [num_heads, 1, 1] * [1, seq_len, seq_len]
    alibi_bias = alibi_bias * slopes.view(-1, 1, 1)
    return alibi_bias


# ============================================================
# Dynamic Region Partitioning (第一层自适应)
# ============================================================

class DynamicRegionPartition(nn.Module):
    """动态区域划分：可学习 Region Tokens + Cross-Attention
    
    核心思想：
      - K 个可学习的 region tokens 作为"查询"
      - 每个 tile token 作为"键"和"值"
      - 通过 cross-attention，每个 region token 聚合相关的 tile
      - 最终输出 K 个动态大小的区域特征
    
    这个模块是你 idea 的核心：让模型自己学习如何划分区域，
    而不是像 CARE 那样用固定大小的 64×64 网格硬切。
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_region_tokens: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_region_tokens = num_region_tokens
        
        # 可学习的 Region Tokens
        self.region_tokens = nn.Parameter(
            torch.randn(1, num_region_tokens, embed_dim) * 0.02
        )
        
        # Cross-attention: region tokens 查询 tile tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(embed_dim)
        
        # FFN for region tokens
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
        # 可选：预测每个 region 覆盖了多少 tile（用于可视化/分析）
        self.region_coverage_predictor = nn.Linear(embed_dim, 1)

    def forward(
        self,
        tile_tokens: Tensor,          # [B, N, C] tile tokens from patch merger
        coords: Optional[Tensor] = None,  # [B, N, 2] tile 坐标 (用于空间约束)
        return_coverage: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        Args:
            tile_tokens: [B, N, C] tile tokens
            coords: [B, N, 2] 归一化的 tile 坐标 (x, y)
            return_coverage: 是否返回每个 region 覆盖的 tile 数量
        
        Returns:
            region_features: [B, K, C] K 个动态区域特征
            attention_weights: [B, K, N] 每个 region 对 tile 的注意力权重
            coverage: [B, K] 每个 region 覆盖的 tile 数量 (可选)
        """
        B, N, C = tile_tokens.shape
        K = self.num_region_tokens
        
        # Expand region tokens to batch size
        region_tokens = self.region_tokens.expand(B, -1, -1)  # [B, K, C]
        
        # ---- Cross-Attention: Region Tokens ← Tile Tokens ----
        # Q: region tokens, K/V: tile tokens
        region_out, attn_weights = self.cross_attn(
            query=region_tokens,
            key=tile_tokens,
            value=tile_tokens,
        )
        region_tokens = self.cross_norm(region_tokens + region_out)
        
        # ---- FFN ----
        region_out = self.ffn(region_tokens)
        region_features = self.ffn_norm(region_tokens + region_out)
        
        # ---- 可选：空间约束 ----
        # 如果提供了坐标，可以额外添加一个空间感知的注意力重加权
        # 让空间上相邻的 tile 更容易被同一个 region 关注
        if coords is not None and self.training:
            region_features = self._apply_spatial_constraint(
                region_features, tile_tokens, coords, attn_weights
            )
        
        outputs = [region_features, attn_weights]
        
        if return_coverage:
            coverage_logits = self.region_coverage_predictor(region_features)  # [B, K, 1]
            coverage = coverage_logits.squeeze(-1).softmax(dim=-1) * N  # [B, K]
            outputs.append(coverage)
        else:
            outputs.append(None)
        
        return tuple(outputs)
    
    def _apply_spatial_constraint(
        self,
        region_features: Tensor,
        tile_tokens: Tensor,
        coords: Tensor,
        attn_weights: Tensor,
    ) -> Tensor:
        """空间约束：鼓励空间上相邻的 tile 被同一 region 关注。"""
        # 计算 tile 之间的空间距离
        B, N, _ = coords.shape
        K = self.num_region_tokens
        
        # coords: [B, N, 2] -> [B, N, 1, 2] - [B, 1, N, 2] -> [B, N, N, 2]
        coord_diff = coords.unsqueeze(2) - coords.unsqueeze(1)
        spatial_dist = torch.norm(coord_diff, p=2, dim=-1)  # [B, N, N]
        spatial_dist = spatial_dist / (spatial_dist.max() + 1e-6)  # normalize
        
        # 空间接近度矩阵（越近越容易属于同一 region）
        spatial_proximity = 1.0 - spatial_dist  # [B, N, N]
        
        # 对 attn_weights 做空间感知的重加权
        # 当前的 attn_weights [B, K, N] 是 region 对 tile 的注意力
        # 我们希望：如果两个 tile 在空间上很近，且都被同一个 region 关注，
        # 则加强这种关联
        # 简化处理：直接返回原始 region_features（空间约束作为消融实验选项）
        return region_features


# ============================================================
# Adaptive Region Modeling (来自 CARE, 第二层自适应)
# ============================================================

class AdaptiveRegionModeling(nn.Module):
    """CARE 的 ARM 模块：区域注意力聚合
    
    核心思想：
      - 每个区域内的 tile 通过 cross-attention 聚合
      - region token 作为查询，区域内 tile 作为键值对
      - 输出区域级特征表示
    
    与 DynamicRegionPartition 的区别：
      - DynamicRegionPartition: 决定"哪些 tile 属于哪个区域"（区域划分）
      - ARM: 决定"如何把区域内的 tile 聚合成有意义的特征"（区域聚合）
    
    两层配合：先划分 → 再聚合
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Tile-to-Region Cross-Attention
        self.tile_to_region_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.tile_norm = nn.LayerNorm(embed_dim)
        
        # Region-level Self-Attention（区域之间的交互）
        self.region_self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.region_norm = nn.LayerNorm(embed_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )
        
        # 聚合后的特征变换
        self.region_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        region_features: Tensor,   # [B, K, C] from DynamicRegionPartition
        tile_tokens: Tensor,         # [B, N, C] all tile tokens
        attn_weights: Tensor,        # [B, K, N] region-to-tile attention
        mask_threshold: float = 0.01,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            region_features: [B, K, C] region tokens from DynamicRegionPartition
            tile_tokens: [B, N, C] all tile tokens
            attn_weights: [B, K, N] attention from region to tiles
            mask_threshold: 忽略注意力权重低于此值的 tile
        
        Returns:
            region_embeddings: [B, K, C] 聚合后的区域特征
            region_pooled: [B, C] WSI 级别特征（所有区域平均）
        """
        B, K, C = region_features.shape
        N = tile_tokens.shape[1]
        
        # ---- Step 1: Tile-to-Region Cross-Attention ----
        # 每个 region token 只关注高权重的 tile
        # 创建一个 mask，将低注意力权重的 tile 置为 -inf
        mask = (attn_weights < mask_threshold).float() * (-1e9)
        
        # 加上 mask 后做 softmax（被 mask 的 tile 贡献接近 0）
        attn_weighted = F.softmax(attn_weights + mask, dim=-1)  # [B, K, N]
        
        # 聚合 tile 特征
        # region_out[b, k, :] = sum_n(attn_weighted[b, k, n] * tile_tokens[b, n, :])
        region_aggregated = torch.bmm(attn_weighted, tile_tokens)  # [B, K, C]
        
        # Cross-attention 层
        tile_out, _ = self.tile_to_region_attn(
            query=region_features,
            key=tile_tokens,
            value=tile_tokens,
            attn_mask=mask,
        )
        region_features = self.tile_norm(region_features + tile_out)
        
        # ---- Step 2: Region-level Self-Attention ----
        # 区域之间也可能存在相互作用（如肿瘤区与间质区的关系）
        region_out, region_attn = self.region_self_attn(
            query=region_features,
            key=region_features,
            value=region_features,
        )
        region_features = self.region_norm(region_features + region_out)
        
        # ---- Step 3: FFN ----
        region_out = self.ffn(region_features)
        region_embeddings = self.region_norm(region_features + region_out)
        
        # ---- Step 4: 区域特征变换 ----
        region_embeddings = self.region_proj(region_embeddings)  # [B, K, C]
        
        # ---- Step 5: WSI 级别聚合 ----
        # 加权平均（基于区域自身的注意力）
        region_pooled = region_embeddings.mean(dim=1)  # [B, C]
        
        return region_embeddings, region_pooled


# ============================================================
# 蒸馏损失模块 (来自 CARE)
# ============================================================

class DistillationLoss(nn.Module):
    """知识蒸馏：让学生模型学习教师模型的行为
    
    教师：CONCH tile encoder（冻结）
    学生：整个融合模型
    
    蒸馏信号：
    1. Tile 级别的蒸馏：学生的 tile tokens 接近 CONCH 的原始 tile 编码
    2. Region 级别的蒸馏：学生的区域特征接近教师理解的区域语义
    """

    def __init__(
        self,
        embed_dim: int = 768,
        distill_dim: int = 768,
        alpha: float = 0.5,
        temperature: float = 3.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature
        
        # 投影层：学生的 tile 特征映射到和 CONCH tile 相同的空间
        if embed_dim != distill_dim:
            self.tile_proj = nn.Linear(embed_dim, distill_dim)
            self.region_proj = nn.Linear(embed_dim, distill_dim)
        else:
            self.tile_proj = nn.Identity()
            self.region_proj = nn.Identity()

    def forward(
        self,
        student_tile_tokens: Tensor,     # [B, N, C]
        teacher_tile_tokens: Tensor,       # [B, N, C_tea] from CONCH
        student_region_emb: Tensor,       # [B, K, C]
        teacher_region_emb: Optional[Tensor] = None,  # [B, K, C_tea] if available
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        计算蒸馏损失。
        
        Args:
            student_tile_tokens: 学生的 tile tokens
            teacher_tile_tokens: 教师的 tile tokens（来自 CONCH）
            student_region_emb: 学生的区域特征
            teacher_region_emb: 教师的区域特征（可选）
        
        Returns:
            total_distill_loss: 总蒸馏损失
            loss_dict: 各分项损失的字典
        """
        # Tile 级别蒸馏损失（KL 散度）
        s_t = student_tile_tokens / self.temperature
        t_t = teacher_tile_tokens / self.temperature
        
        distill_tile_loss = F.kl_div(
            F.log_softmax(s_t, dim=-1),
            F.softmax(t_t, dim=-1),
            reduction='batchmean',
        ) * (self.temperature ** 2)
        
        loss_dict = {
            'distill_tile_loss': distill_tile_loss.item(),
        }
        
        # Region 级别蒸馏损失（如果教师区域特征可用）
        if teacher_region_emb is not None:
            s_r = student_region_emb / self.temperature
            t_r = teacher_region_emb / self.temperature
            
            distill_region_loss = F.kl_div(
                F.log_softmax(s_r, dim=-1),
                F.softmax(t_r, dim=-1),
                reduction='batchmean',
            ) * (self.temperature ** 2)
            
            total_distill_loss = (
                (1 - self.alpha) * distill_tile_loss +
                self.alpha * distill_region_loss
            )
            loss_dict['distill_region_loss'] = distill_region_loss.item()
        else:
            total_distill_loss = distill_tile_loss
        
        return total_distill_loss, loss_dict


# ============================================================
# 多任务预测头 (来自 CARE)
# ============================================================

class MultiTaskHead(nn.Module):
    """多任务分子标志物预测头
    
    支持多标签分类（ER/PR/HER2/LDH 等分子标志物）
    每个任务有独立的分类头，支持阈值化预测
    
    来自 CARE 的设计：
    - 每个任务有独立的 sigmoid 分类器
    - 可选：阈值化机制（让模型学习每个标志物的最优阈值）
    - 多任务联合训练，共享特征编码器
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_tasks: int = 4,  # ER, PR, HER2, LDH
        task_names: Optional[List[str]] = None,
        dropout: float = 0.25,
        use_threshold: bool = True,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]
        self.use_threshold = use_threshold
        
        # 共享特征编码
        self.shared = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 每个任务独立的分类头
        self.task_heads = nn.ModuleList([
            nn.Linear(embed_dim, 1) for _ in range(num_tasks)
        ])
        
        # 可学习的阈值（可选，来自 CARE 的阈值化设计）
        if use_threshold:
            self.task_thresholds = nn.Parameter(
                torch.zeros(num_tasks), requires_grad=True
            )

    def forward(
        self,
        wsi_features: Tensor,  # [B, C]
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Args:
            wsi_features: [B, C] WSI 级别特征
        
        Returns:
            outputs: dict of {
                'logits': [num_tasks],
                'probs': [num_tasks],
                'preds': [num_tasks] (二值预测),
            }
        """
        x = self.shared(wsi_features)  # [B, C]
        
        logits = torch.cat([head(x) for head in self.task_heads], dim=-1)  # [B, num_tasks]
        probs = torch.sigmoid(logits)  # [B, num_tasks]
        
        if self.use_threshold:
            thresholds = torch.sigmoid(self.task_thresholds) * 0.5  # [0, 0.5]
            preds = (probs > thresholds).float()
        else:
            preds = (probs > 0.5).float()
        
        return {
            'logits': logits.squeeze(0),      # [num_tasks]
            'probs': probs.squeeze(0),         # [num_tasks]
            'preds': preds.squeeze(0),         # [num_tasks]
        }


# ============================================================
# Patch Merger (来自 E2E-ViT)
# ============================================================

class PatchMerger(nn.Module):
    """Patch Merger: 将 patch tokens 聚合成 tile tokens
    
    来自 E2E-ViT 的设计：
    - 每个 tile 内部的 (H/P)² 个 patch tokens 通过 mean pooling 聚合成 1 个 tile token
    - 这个聚合是可学习的（通过 LayerNorm）
    - 保留了原始 patch token 的特征空间，可以直接使用预训练权重
    """

    def __init__(
        self,
        patch_size: int = 16,  # CONCH 的 patch size
        tile_size: int = 256,  # 每个 tile 的像素大小 (H = W = 256)
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.tile_size = tile_size
        self.patches_per_tile = (tile_size // patch_size) ** 2
        
        # 可学习的缩放和偏移
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, patch_tokens: Tensor) -> Tensor:
        """
        Args:
            patch_tokens: [B, N_patches, C] 其中 N_patches = num_tiles * patches_per_tile
        
        Returns:
            tile_tokens: [B, N_tiles, C]
        """
        B, N_total, C = patch_tokens.shape
        
        # Reshape: [B, N_tiles, patches_per_tile, C]
        N_tiles = N_total // self.patches_per_tile
        tile_tokens = patch_tokens.view(B, N_tiles, self.patches_per_tile, C)
        
        # Mean pooling: [B, N_tiles, C]
        tile_tokens = tile_tokens.mean(dim=2)
        
        # LayerNorm
        tile_tokens = self.norm(tile_tokens)
        
        return tile_tokens


# ============================================================
# Patch Embedding (来自 E2E-ViT / CONCH)
# ============================================================

class PatchEmbed(nn.Module):
    """Patch Embedding: 将图像分割成 patches
    
    来自 E2E-ViT 的设计：
    - 使用非重叠的 Conv2d 层（kernel_size = patch_size）
    - 直接使用 CONCH 预训练的权重
    - 支持任意长宽比的输入（长条图）
    """

    def __init__(
        self,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
        # 注册一个 identity 投影用于初始化（稍后加载 CONCH 权重）
        # 实际使用时会从 HuggingFace 加载预训练权重

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Args:
            x: [B, C, H, W] 图像张量
        
        Returns:
            patch_tokens: [B, N_patches, C] 其中 N_patches = (H*W) / patch_size²
            grid_size: (num_patches_h, num_patches_w)
        """
        B, C, H, W = x.shape
        patch_tokens = self.proj(x)  # [B, C, H/P, W/P]
        
        # Reshape: [B, C, H/P, W/P] -> [B, H/P*W/P, C]
        patch_tokens = patch_tokens.flatten(2).transpose(1, 2)
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        
        return patch_tokens, (num_patches_h, num_patches_w)


# ============================================================
# Transformer Backbone (来自 E2E-ViT)
# ============================================================

class E2ETransformerBlock(nn.Module):
    """E2E-ViT 的 Transformer Block: 带 ALiBi 位置编码
    
    关键特点：
    - 用 ALiBi 替代绝对位置编码
    - ALiBi 让预训练的 ViT 能外推到更长的序列
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        alibi_bias: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: [B, N, C]
            alibi_bias: [num_heads, N, N] ALiBi 偏置
            key_padding_mask: [B, N] True 表示 mask 掉的位置
        """
        # Self-attention with ALiBi
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=alibi_bias,
            key_padding_mask=key_padding_mask,
        )
        x = x + attn_output
        
        # FFN
        x = x + self.mlp(self.norm2(x))
        
        return x


# ============================================================
# 完整融合模型
# ============================================================

class E2EViTCAREFusion(nn.Module):
    """E2E-ViT + CARE 融合模型：双层自适应区域建模
    
    整体流程:
      WSI → 固定网格裁剪 → Patch Embedding → Patch Merger
           → Tile Tokens → ALiBi 编码 → Transformer Backbone
           → ┌──────────────────┐
             │  分支1: 直接分类 (E2E-ViT style)
             │  分支2: 双层自适应区域 → 多任务分类 (CARE style)
             └──────────────────┘
           → 多任务预测头
    
    创新点:
      1. 第一层自适应：DynamicRegionPartition 动态划分区域
      2. 第二层自适应：ARM 区域注意力聚合
      3. 保留 E2E-ViT 的固定网格 + ALiBi 位置编码
      4. 保留 CARE 的多任务预测头和蒸馏损失
    """

    def __init__(
        self,
        # === 输入配置 ===
        tile_size: int = 256,       # 每个 tile 的像素大小
        patch_size: int = 16,       # ViT 的 patch size (CONCH = 16)
        # === 编码器配置 ===
        embed_dim: int = 768,       # CONCH 的 embedding dim
        num_heads: int = 12,        # Transformer 的头数
        num_layers: int = 12,       # Transformer 的层数
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        # === 区域建模配置 ===
        num_region_tokens: int = 8,  # 动态区域的数量（可学习）
        # === 预测头配置 ===
        num_tasks: int = 4,          # 分子标志物数量
        task_names: Optional[List[str]] = None,
        task_loss_weight: float = 0.1,  # 任务损失的权重
        distill_weight: float = 0.5,     # 蒸馏损失的权重
        # === 其他 ===
        use_alibi: bool = True,
        use_distillation: bool = True,
        use_two_branches: bool = True,  # 是否同时使用两个分支
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_region_tokens = num_region_tokens
        self.task_loss_weight = task_loss_weight
        self.distill_weight = distill_weight
        self.use_alibi = use_alibi
        self.use_distillation = use_distillation
        self.use_two_branches = use_two_branches
        
        # === Patch Embedding (Conv2d，来自 CONCH) ===
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )
        
        # === Patch Merger ===
        self.patch_merger = PatchMerger(
            patch_size=patch_size,
            tile_size=tile_size,
            embed_dim=embed_dim,
        )
        
        # === Transformer Backbone ===
        self.blocks = nn.ModuleList([
            E2ETransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # === 第一层自适应: 动态区域划分 ===
        self.dynamic_region_partition = DynamicRegionPartition(
            embed_dim=embed_dim,
            num_region_tokens=num_region_tokens,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # === 第二层自适应: ARM 区域聚合 ===
        self.arm = AdaptiveRegionModeling(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # === 预测头 ===
        self.head_direct = nn.Linear(embed_dim, num_tasks)  # 直接分类分支
        self.head_adaptive = MultiTaskHead(
            embed_dim=embed_dim,
            num_tasks=num_tasks,
            task_names=task_names,
            dropout=dropout,
        )
        
        # === 蒸馏模块 ===
        if use_distillation:
            self.distillation = DistillationLoss(
                embed_dim=embed_dim,
                distill_dim=embed_dim,
                alpha=0.5,
                temperature=3.0,
            )
        
        # === 特征投影（融合两个分支）===
        if use_two_branches:
            self.fusion_proj = nn.Linear(embed_dim * 2, embed_dim)

    def load_conch_weights(self, conch_model: nn.Module):
        """加载 CONCH 的预训练权重到 patch embedding 和 transformer backbone"""
        # Conv2d patch embedding
        if hasattr(conch_model, 'vit') and hasattr(conch_model.vit, 'patch_embed'):
            self.patch_embed.proj.load_state_dict(
                conch_model.vit.patch_embed.proj.state_dict(), strict=False
            )
        
        # Transformer blocks
        if hasattr(conch_model, 'vit') and hasattr(conch_model.vit, 'blocks'):
            for i, block in enumerate(self.blocks):
                if i < len(conch_model.vit.blocks):
                    block.load_state_dict(
                        conch_model.vit.blocks[i].state_dict(), strict=False
                    )
    
    def forward(
        self,
        strip_image: Tensor,              # [B, 3, H, N*W] 长条图
        coords: Optional[Tensor] = None,  # [B, N, 2] tile 坐标
        labels: Optional[Tensor] = None,   # [B, num_tasks] 标签
        teacher_tile_tokens: Optional[Tensor] = None,  # [B, N, C] 来自 CONCH
        return_all: bool = False,         # 是否返回所有中间结果
    ) -> Dict[str, Tensor]:
        """
        Args:
            strip_image: [B, 3, H, N*W] 长条图图像
            coords: [B, N, 2] tile 的归一化坐标 (x, y)
            labels: [B, num_tasks] 多任务标签
            teacher_tile_tokens: [B, N, C] 教师模型的 tile tokens (用于蒸馏)
            return_all: 是否返回所有中间结果
        
        Returns:
            outputs: dict 包含:
                - logits_direct: 直接分支的 logits [B, num_tasks]
                - logits_adaptive: 自适应分支的 logits [B, num_tasks]
                - loss: 总损失
                - loss_dict: 各分项损失
        """
        B = strip_image.shape[0]
        
        # ---- Patch Embedding ----
        patch_tokens, grid_size = self.patch_embed(strip_image)  # [B, N_patches, C]
        
        # ---- Patch Merger ----
        tile_tokens = self.patch_merger(patch_tokens)  # [B, N, C]
        N_tiles = tile_tokens.shape[1]
        
        # ---- ALiBi 位置编码 ----
        alibi_bias = None
        if self.use_alibi:
            alibi_bias = build_alibi_bias(
                seq_len=N_tiles,
                num_heads=self.blocks[0].num_heads,
                device=strip_image.device,
                dtype=strip_image.dtype,
            )
        
        # ---- Transformer Backbone ----
        x = tile_tokens
        for block in self.blocks:
            x = block(x, alibi_bias=alibi_bias)  # [B, N, C]
        
        # ---- 分支1: 直接分类 (E2E-ViT style) ----
        # 全局平均池化 + 分类
        x_global = x.mean(dim=1)  # [B, C]
        logits_direct = self.head_direct(x_global)  # [B, num_tasks]
        
        outputs = {
            'tile_tokens': tile_tokens,
            'transformer_output': x,
            'logits_direct': logits_direct,
        }
        
        # ---- 分支2: 双层自适应区域 (CARE style) ----
        if self.use_two_branches:
            # 第一层: 动态区域划分
            region_features, attn_weights, coverage = self.dynamic_region_partition(
                tile_tokens=x,  # 用 Transformer 输出的 token
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
            logits_adaptive = adaptive_output['logits']  # [num_tasks]
            
            outputs['region_features'] = region_features
            outputs['region_embeddings'] = region_embeddings
            outputs['attn_weights'] = attn_weights
            outputs['coverage'] = coverage
            outputs['logits_adaptive'] = logits_adaptive.unsqueeze(0)
            outputs['adaptive_probs'] = adaptive_output['probs'].unsqueeze(0)
            outputs['adaptive_preds'] = adaptive_output['preds'].unsqueeze(0)
        
        # ---- 损失计算 ----
        loss_dict = {}
        total_loss = 0.0
        
        if labels is not None:
            # 直接分支损失
            loss_direct = F.binary_cross_entropy_with_logits(
                logits_direct, labels
            )
            total_loss = total_loss + loss_direct
            loss_dict['loss_direct'] = loss_direct.item()
            
            if self.use_two_branches:
                # 自适应分支损失
                loss_adaptive = F.binary_cross_entropy_with_logits(
                    logits_adaptive.unsqueeze(0), labels
                )
                total_loss = total_loss + loss_adaptive
                loss_dict['loss_adaptive'] = loss_adaptive.item()
                
                # 分支融合损失（让两个分支互相学习）
                loss_fusion = F.mse_loss(
                    torch.sigmoid(logits_direct),
                    torch.sigmoid(logits_adaptive.unsqueeze(0)),
                )
                total_loss = total_loss + 0.1 * loss_fusion
                loss_dict['loss_fusion'] = loss_fusion.item()
        
        # ---- 蒸馏损失 ----
        if self.use_distillation and teacher_tile_tokens is not None:
            distill_loss, distill_dict = self.distillation(
                student_tile_tokens=tile_tokens,
                teacher_tile_tokens=teacher_tile_tokens,
                student_region_emb=region_embeddings if self.use_two_branches else None,
                teacher_region_emb=None,
            )
            total_loss = total_loss + self.distill_weight * distill_loss
            loss_dict['loss_distill'] = distill_loss.item()
            loss_dict.update(distill_dict)
        
        outputs['loss'] = total_loss
        outputs['loss_dict'] = loss_dict
        
        return outputs


# ============================================================
# 工具函数
# ============================================================

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """统计模型的参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        'total': total,
        'trainable': trainable,
        'frozen': frozen,
    }


def print_model_summary(model: nn.Module):
    """打印模型摘要"""
    params = count_parameters(model)
    print("=" * 60)
    print("E2E-ViT + CARE Fusion Model Summary")
    print("=" * 60)
    print(f"Total parameters:      {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters:    {params['frozen']:,}")
    print("=" * 60)
    
    # 各模块参数量
    print("\nModule Parameters:")
    for name, module in model.named_children():
        mod_params = sum(p.numel() for p in module.parameters())
        print(f"  {name:30s}: {mod_params:,}")
