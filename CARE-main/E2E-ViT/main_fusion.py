"""
主入口：E2E-ViT + CARE 融合模型训练

使用方法:
    python main_fusion.py --config configs/fusion_config.yaml

    或直接指定参数:
    python main_fusion.py \
        --dataset MUT \
        --experiment_target BAP1 \
        --task t1_gene \
        --num_tasks 1 \
        --num_region_tokens 8 \
        --lr 2e-5 \
        --max_epochs 200 \
        --k 5
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Tuple, Dict
from PIL import Image
import pandas as pd


# ============================================================
# 数据预处理：WSI → 固定网格长条图
# ============================================================

class WSIToStripConverter:
    """将 WSI 转换为固定网格长条图

    来自 E2E-ViT 的设计：
    - 使用 OTSU 或 tissue mask 去除背景
    - 用固定大小的滑动窗口裁剪非重叠 tile
    - 按空间顺序将 tile 拼接成一张长条图
    """

    def __init__(
        self,
        tile_size: int = 256,      # tile 像素大小 (H = W)
        target_mag: int = 20,       # 放大倍数
        min_tissue_ratio: float = 0.1,  # tile 中组织占比阈值
        background_threshold: int = 200,  # 背景阈值
    ):
        self.tile_size = tile_size
        self.target_mag = target_mag
        self.min_tissue_ratio = min_tissue_ratio
        self.background_threshold = background_threshold

    def wsi_to_strip(
        self,
        wsi_path: str,
        tissue_mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, np.ndarray]:
        """将 WSI 转换为长条图

        Args:
            wsi_path: WSI 文件路径 (.svs, .tif, .ndpi)
            tissue_mask: 组织掩码（可选，如果不提供则自动计算）

        Returns:
            strip_image: [3, H, W] 长条图张量
            coords: [N, 2] 每个 tile 的 (x, y) 坐标
        """
        # TODO: 实现 WSI 读取和 tile 裁剪
        # 这部分可以复用 CARE 的 wsi_core/WholeSlideImage.py
        raise NotImplementedError("WSI reading not implemented yet")

    def tile_to_strip(
        self,
        tiles: List[np.ndarray],
        coords: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将 tile 列表转换为长条图

        Args:
            tiles: [N, H, W, 3] tile 图像列表
            coords: [N, 2] tile 坐标

        Returns:
            strip_image: [3, H, W*N] 长条图
            coords_out: [N, 2] 归一化坐标
        """
        # 按 x 坐标排序（从左到右，从上到下）
        sort_idx = np.lexsort((coords[:, 1], coords[:, 0]))
        tiles = [tiles[i] for i in sort_idx]
        coords = coords[sort_idx]

        # 拼接成长条图
        strip = np.concatenate(tiles, axis=2)  # [H, W*N, 3]

        # 归一化坐标
        max_x = coords[:, 0].max()
        max_y = coords[:, 1].max()
        coords_norm = coords / np.array([max_x, max_y])

        # 转换为张量 [3, H, W*N]
        strip_tensor = torch.from_numpy(strip).permute(2, 0, 1).float() / 255.0

        return strip_tensor, coords_norm


# ============================================================
# 数据集
# ============================================================

class FusionWSIDataset(Dataset):
    """融合模型的数据集"""

    def __init__(
        self,
        csv_path: str,
        data_root: str,
        tile_size: int = 256,
        transform=None,
        max_tiles: int = 4096,  # 限制最大 tile 数量（内存限制）
        return_coords: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.data_root = data_root
        self.tile_size = tile_size
        self.transform = transform
        self.max_tiles = max_tiles
        self.return_coords = return_coords

        self.converter = WSIToStripConverter(tile_size=tile_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]
        slide_id = row['slide_id']
        label = row.get('label', 0)

        # 加载预处理好的 tile 特征（来自 CARE 的 .h5 或 .npy）
        # 这里假设你已经用 CONCH 提取好了 tile 特征
        feature_path = os.path.join(
            self.data_root,
            f"{slide_id}_0_{self.tile_size}.npy"
        )

        if os.path.exists(feature_path):
            # 加载已有的 tile 特征
            data = np.load(feature_path, allow_pickle=True).item()
            tile_features = torch.from_numpy(data['feature'])  # [N, C]
            coords = torch.from_numpy(data['coords'])          # [N, 2]
        else:
            # 如果没有预处理好的特征，需要从 WSI 重新提取
            # TODO: 实现 WSI 到 strip 的转换
            raise FileNotFoundError(f"Feature file not found: {feature_path}")

        # 限制 tile 数量
        if tile_features.shape[0] > self.max_tiles:
            # 随机采样
            indices = np.random.choice(tile_features.shape[0], self.max_tiles, replace=False)
            indices = np.sort(indices)
            tile_features = tile_features[indices]
            coords = coords[indices]

        return {
            'strip_image': tile_features,  # [N, C] (实际使用时会是 [3, H, W] 图像)
            'coords': coords,               # [N, 2]
            'label': torch.tensor(label, dtype=torch.float32),
            'slide_id': slide_id,
        }


# ============================================================
# 配置
# ============================================================

def get_default_args() -> argparse.Namespace:
    """默认配置"""
    parser = argparse.ArgumentParser(description='E2E-ViT + CARE Fusion Training')

    # 数据配置
    parser.add_argument('--dataset', type=str, default='MUT', help='数据集名称')
    parser.add_argument('--subdataset', type=str, default=None)
    parser.add_argument('--experiment_target', type=str, default='BAP1', help='实验目标基因/标志物')
    parser.add_argument('--task', type=str, default='t1_gene', help='任务类型')
    parser.add_argument('--csv_path', type=str, default='./dataset_csv')
    parser.add_argument('--data_root_dir', type=str, default='./data')
    parser.add_argument('--split_dir', type=str, default=None)

    # 模型配置
    parser.add_argument('--model_name', type=str, default='E2E_ViT_CARE_Fusion')
    parser.add_argument('--embed_dim', type=int, default=768, help='CONCH embedding dim')
    parser.add_argument('--num_heads', type=int, default=12, help='Transformer 头数')
    parser.add_argument('--num_layers', type=int, default=12, help='Transformer 层数')
    parser.add_argument('--num_region_tokens', type=int, default=8, help='动态区域数量')
    parser.add_argument('--tile_size', type=int, default=256, help='Tile 像素大小')
    parser.add_argument('--patch_size', type=int, default=16, help='ViT patch size')
    parser.add_argument('--use_alibi', type=bool, default=True, help='使用 ALiBi 位置编码')
    parser.add_argument('--use_distillation', type=bool, default=True, help='使用蒸馏损失')
    parser.add_argument('--distill_weight', type=float, default=0.5)
    parser.add_argument('--task_loss_weight', type=float, default=0.1)

    # 预测配置
    parser.add_argument('--num_tasks', type=int, default=1, help='分子标志物数量')
    parser.add_argument('--task_names', type=str, default='BAP1', help='任务名称，逗号分隔')

    # 训练配置
    parser.add_argument('--lr', type=float, default=2e-5, help='学习率')
    parser.add_argument('--reg', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--batch_size', type=int, default=1, help='批大小')
    parser.add_argument('--max_epochs', type=int, default=200)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--k', type=int, default=5, help='折数')
    parser.add_argument('--k_start', type=int, default=-1)
    parser.add_argument('--k_end', type=int, default=-1)
    parser.add_argument('--label_frac', type=float, default=1.0)

    # 其他配置
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--results_dir', type=str, default='./results/fusion')
    parser.add_argument('--exp_code', type=str, default=None)
    parser.add_argument('--testing', action='store_true')
    parser.add_argument('--log_data', action='store_true', default=True)
    parser.add_argument('--early_stopping', action='store_true', default=True)

    args = parser.parse_args([])

    # 填充默认值
    args.task_names = args.task_names.split(',') if isinstance(args.task_names, str) else [args.task_names]
    args.num_tasks = len(args.task_names)

    return args


# ============================================================
# 主函数
# ============================================================

def main(args):
    # 设置随机种子
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 路径设置
    if args.exp_code is None:
        args.exp_code = f"{args.dataset}_{args.experiment_target}_{args.model_name}_{args.num_region_tokens}"

    args.results_dir = os.path.join(args.results_dir, args.exp_code + f'_s{args.seed}')
    os.makedirs(args.results_dir, exist_ok=True)

    if args.split_dir is None:
        args.split_dir = os.path.join(
            './splits', args.task,
            f'{args.dataset}_{args.experiment_target}_{args.label_frac*100}'
        )

    # 导入模型
    from models.E2E_ViT_CARE_fusion import E2EViTCAREFusion, print_model_summary

    # 创建模型
    print("Creating model...")
    model = E2EViTCAREFusion(
        tile_size=args.tile_size,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_region_tokens=args.num_region_tokens,
        num_tasks=args.num_tasks,
        task_names=args.task_names,
        task_loss_weight=args.task_loss_weight,
        distill_weight=args.distill_weight,
        use_alibi=args.use_alibi,
        use_distillation=args.use_distillation,
    )

    # 打印模型摘要
    print_model_summary(model)
    model = model.to(device)

    # 加载 CONCH 预训练权重（可选）
    # if args.load_conch:
    #     from transformers import AutoModel
    #     conch = AutoModel.from_pretrained("Zipper-1/CARE", trust_remote_code=True)
    #     model.load_conch_weights(conch)
    #     print("Loaded CONCH pretrained weights")

    # 训练
    from train_fusion import train_fold

    all_val_auc = []
    all_test_auc = []

    for fold in range(args.k):
        if args.testing and fold > 0:
            break

        # 加载数据集
        # 这里需要根据你的数据格式来实现
        # train_loader, val_loader, test_loader = get_dataloaders(args, fold)

        print(f"\nFold {fold}: Dataset loading not implemented yet.")
        print("请根据你的数据格式实现数据集加载逻辑。")

        # 如果数据集已准备好，取消注释下面的代码
        # results = train_fold(
        #     fold=fold,
        #     model=model,
        #     train_loader=train_loader,
        #     val_loader=val_loader,
        #     device=device,
        #     args=args,
        #     results_dir=args.results_dir,
        # )
        # all_val_auc.append(results.get('auc_adaptive', results['auc_direct']))

    # 打印最终结果
    if all_val_auc:
        print(f"\n{'='*60}")
        print(f"Final Results ({args.k} folds)")
        print(f"{'='*60}")
        print(f"Mean Val AUC: {np.mean(all_val_auc):.4f} ± {np.std(all_val_auc):.4f}")


if __name__ == '__main__':
    args = get_default_args()
    main(args)
