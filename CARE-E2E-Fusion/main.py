"""
主入口: E2E-ViT + CARE 融合模型

使用方法:
    # 模拟数据测试（直接运行）
    python main.py

    # 指定参数
    python main.py --num_samples 200 --num_region_tokens 8 --num_layers 4

    # 真实数据训练
    python main.py --dataset real --csv_path ./data/xxx.csv --data_root_dir ./data/
"""

from train import main, get_args

if __name__ == '__main__':
    args = get_args()
    main(args)
