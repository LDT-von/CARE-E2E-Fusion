# E2E-ViT + CARE Fusion: Dual-Adaptive Region Modeling

> **双层自适应区域建模**：融合 E2E-ViT 的固定网格输入 + CARE 的自适应区域聚合

## 核心思想

```
WSI → 固定网格裁剪 → 长条图 → Conv2d Patch Embedding → Patch Merger
                                                              ↓
                                                    Transformer Backbone (CONCH)
                                                              ↓
                                    ┌──────────────────────────┼──────────────────────────┐
                                    ↓                          ↓                          ↓
                              直接分支                    第一层自适应                   第二层自适应
                              (Mean Pool)              动态区域划分                  CARE 的 ARM
                              ↓                          ↓                          ↓
                              FC Head              可学习 Region Tokens           区域注意力聚合
                                                           ↓
                                                    动态大小的区域
                                                           ↓
                                                    多任务预测头
                                                    ER / PR / HER2 / LDH
```

## 创新点

1. **第一层自适应：动态区域划分**
   - K 个可学习 region tokens 作为查询
   - 通过 cross-attention 让模型自己决定哪些 tile 属于哪个区域
   - 区域大小不再固定，由数据驱动

2. **第二层自适应：CARE 的 ARM 区域聚合**
   - 每个区域内做注意力聚合
   - 区域之间做 self-attention（捕捉区域间关系）
   - 输出区域级特征

3. **保留 E2E-ViT 的优势**
   - 固定网格：保留空间连续性
   - ALiBi 位置编码：支持任意长度的序列外推
   - 零新增参数（backbone 不变）

4. **保留 CARE 的蒸馏机制**
   - 教师：CONCH tile encoder
   - 学生：整个融合模型
   - 蒸馏损失 + 任务损失联合训练

## 文件结构

```
E2E-ViT/
├── models/
│   ├── __init__.py
│   ├── E2E_ViT_CARE_fusion.py   # 核心模型代码
│   └── README_fusion.md          # 本文件
├── train_fusion.py               # 训练封装
├── main_fusion.py                # 主入口
└── configs/
    └── fusion_config.yaml        # 配置文件（可选）
```

## 快速开始

### 1. 模型架构

核心模型 `E2EViTCAREFusion` 包含以下模块：

| 模块 | 来自 | 作用 |
|------|------|------|
| `PatchEmbed` | E2E-ViT | Conv2d patch embedding |
| `PatchMerger` | E2E-ViT | 将 patch tokens 聚合成 tile tokens |
| `ALiBi` | E2E-ViT | 相对位置编码，支持序列外推 |
| `Transformer Blocks` | CONCH | 预训练的 ViT backbone |
| `DynamicRegionPartition` | 新设计 | 第一层自适应：动态区域划分 |
| `AdaptiveRegionModeling` | CARE | 第二层自适应：区域注意力聚合 |
| `MultiTaskHead` | CARE | 多任务分子标志物预测 |
| `DistillationLoss` | CARE | 知识蒸馏 |

### 2. 训练

```python
from models.E2E_ViT_CARE_fusion import E2EViTCAREFusion
from train_fusion import FusionTrainer

model = E2EViTCAREFusion(
    tile_size=256,
    patch_size=16,
    embed_dim=768,
    num_heads=12,
    num_layers=12,
    num_region_tokens=8,  # 动态区域数量
    num_tasks=4,          # ER, PR, HER2, LDH
    task_names=['ER', 'PR', 'HER2', 'LDH'],
    use_alibi=True,
    use_distillation=True,
)

# 加载 CONCH 预训练权重
from transformers import AutoModel
conch = AutoModel.from_pretrained("Zipper-1/CARE", trust_remote_code=True)
model.load_conch_weights(conch)

trainer = FusionTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=2e-5),
    device=torch.device('cuda'),
    num_tasks=4,
    task_names=['ER', 'PR', 'HER2', 'LDH'],
)
```

### 3. 数据格式

模型期望的输入格式：

```python
# 长条图（来自 E2E-ViT）
strip_image = torch.randn(1, 3, 256, 256 * 100)  # [B, 3, H, W*N]
coords = torch.randn(1, 100, 2)  # [B, N, 2] tile 坐标

# 标签（多任务）
labels = torch.tensor([[1.0, 0.0, 1.0, 0.5]])  # [B, num_tasks]

outputs = model(strip_image, coords, labels)
print(outputs.keys())
# ['tile_tokens', 'transformer_output', 'logits_direct',
#  'region_features', 'region_embeddings', 'attn_weights',
#  'coverage', 'logits_adaptive', 'loss', 'loss_dict']
```

## 消融实验设计

| 实验 | 描述 | 预期结论 |
|------|------|----------|
| E2E-ViT only | 只有 E2E-ViT 分支，无 CARE | 验证 CARE 的价值 |
| CARE only | 只有 CARE 分支，无 E2E-ViT 输入 | 验证 E2E-ViT 输入的价值 |
| Fixed Grid + ARM | 固定网格 + CARE ARM（无第一层自适应） | 验证第一层自适应的价值 |
| Random + ARM | 随机采样 + CARE ARM | 验证固定网格的价值 |
| **Full Model** | 双层自适应（完整模型） | 验证两者结合的价值 |
| No Distillation | 去掉蒸馏损失 | 验证蒸馏的价值 |

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_region_tokens` | 8 | 动态区域数量（K）。太大可能过拟合，太小可能欠拟合 |
| `task_loss_weight` | 0.1 | 任务损失的权重 |
| `distill_weight` | 0.5 | 蒸馏损失的权重 |
| `tile_size` | 256 | 每个 tile 的像素大小 |
| `patch_size` | 16 | ViT 的 patch size（CONCH = 16） |
| `use_alibi` | True | 是否使用 ALiBi 位置编码 |
| `use_two_branches` | True | 是否同时使用两个分支 |

## 预期发表点

1. **第一个**将固定网格输入与自适应区域建模结合的工作
2. **第一个**提出双层自适应（动态区域划分 + 区域注意力聚合）的工作
3. 在 **TCGA / MUT 分子标志物数据集**上验证
4. 消融实验覆盖所有关键组件

## 参考

- E2E-ViT: [Turning Pre-Trained Vision Transformers into End-to-End Histopathology WSI Models]()
- CARE: [Adaptive Region Modeling for Pathology WSI Classification]()
- ALiBi: [Train Short, Test Long: Attention with Linear Biases]()
- CONCH: [Curation of Histopathology Images via Pre-training]()
