"""
训练封装：E2E-ViT + CARE 融合模型的训练流程

支持:
  - 多任务分子标志物预测
  - 双分支训练（直接分支 + 自适应分支）
  - 知识蒸馏
  - 5折交叉验证
  - 早停机制
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Optional, Dict, List, Tuple, Callable
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score


class FusionTrainer:
    """融合模型的训练器"""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        num_tasks: int = 4,
        task_names: Optional[List[str]] = None,
        task_loss_weight: float = 0.1,
        distill_weight: float = 0.5,
        use_distillation: bool = True,
        log_dir: Optional[str] = None,
        early_stopping_patience: int = 15,
        early_stopping_stop_epoch: int = 20,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.num_tasks = num_tasks
        self.task_names = task_names or [f'task_{i}' for i in range(num_tasks)]
        self.task_loss_weight = task_loss_weight
        self.distill_weight = distill_weight
        self.use_distillation = use_distillation
        self.log_dir = log_dir
        self.scaler = GradScaler('cuda')
        self.best_score = None
        self.counter = 0
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_stop_epoch = early_stopping_stop_epoch

    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        loss_breakdown = {}
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            # 解包数据
            strip_image, coords, labels, slide_ids = self._prepare_batch(batch)

            self.optimizer.zero_grad()

            # 前向传播（混合精度）
            with autocast(device_type='cuda'):
                outputs = self.model(
                    strip_image=strip_image,
                    coords=coords,
                    labels=labels,
                    teacher_tile_tokens=None,  # 蒸馏可选
                )

            # 反向传播
            loss = outputs['loss']
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 累计损失
            total_loss += loss.item()
            for k, v in outputs.get('loss_dict', {}).items():
                loss_breakdown[k] = loss_breakdown.get(k, 0.0) + v
            num_batches += 1

            # 打印每 50 个 batch 的状态
            if batch_idx % 50 == 0:
                self._print_batch_status(epoch, batch_idx, len(train_loader), loss.item(), outputs)

        # 计算平均值
        avg_loss = total_loss / num_batches
        for k in loss_breakdown:
            loss_breakdown[k] /= num_batches

        return {'loss': avg_loss, **loss_breakdown}

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_labels = []
        all_probs_direct = []
        all_probs_adaptive = []
        loss_breakdown = {}

        for batch in tqdm(val_loader, desc='Validating'):
            strip_image, coords, labels, slide_ids = self._prepare_batch(batch)

            with autocast(device_type='cuda'):
                outputs = self.model(
                    strip_image=strip_image,
                    coords=coords,
                    labels=labels,
                    teacher_tile_tokens=None,
                )

            total_loss += outputs['loss'].item()
            num_batches += 1

            for k, v in outputs.get('loss_dict', {}).items():
                loss_breakdown[k] = loss_breakdown.get(k, 0.0) + v

            # 收集预测结果
            all_labels.append(labels.cpu().numpy())
            probs_d = torch.sigmoid(outputs['logits_direct']).cpu().numpy()
            all_probs_direct.append(probs_d)
            if 'logits_adaptive' in outputs:
                probs_a = torch.sigmoid(outputs['logits_adaptive']).cpu().numpy()
                all_probs_adaptive.append(probs_a)

        avg_loss = total_loss / num_batches
        for k in loss_breakdown:
            loss_breakdown[k] /= num_batches

        # 计算 AUC
        all_labels = np.concatenate(all_labels)
        all_probs_direct = np.concatenate(all_probs_direct)

        if self.num_tasks == 1:
            auc_direct = roc_auc_score(all_labels, all_probs_direct)
        else:
            try:
                auc_direct = roc_auc_score(all_labels, all_probs_direct, multi_class='ovr', average='macro')
            except ValueError:
                auc_direct = 0.0

        results = {'loss': avg_loss, 'auc_direct': auc_direct, **loss_breakdown}

        if all_probs_adaptive:
            all_probs_adaptive = np.concatenate(all_probs_adaptive)
            try:
                auc_adaptive = roc_auc_score(all_labels, all_probs_adaptive, multi_class='ovr', average='macro')
            except ValueError:
                auc_adaptive = 0.0
            results['auc_adaptive'] = auc_adaptive

        return results

    def _prepare_batch(
        self,
        batch: Tuple,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor, List[str]]:
        """准备批次数据"""
        if len(batch) == 4:
            strip_image, coords, labels, slide_ids = batch
        elif len(batch) == 3:
            strip_image, labels, coords = batch
            slide_ids = []
        else:
            raise ValueError(f"Unexpected batch format with {len(batch)} elements")

        strip_image = strip_image.to(self.device)
        labels = labels.to(self.device).float()

        if coords is not None:
            coords = coords.to(self.device)
        else:
            coords = None

        return strip_image, coords, labels, slide_ids

    def _print_batch_status(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        loss: float,
        outputs: Dict,
    ):
        """打印批次状态"""
        msg = f"Epoch {epoch} [{batch_idx}/{total_batches}] Loss: {loss:.4f}"
        if 'loss_direct' in outputs.get('loss_dict', {}):
            msg += f" | Direct: {outputs['loss_dict'].get('loss_direct', 0):.4f}"
        if 'loss_adaptive' in outputs.get('loss_dict', {}):
            msg += f" | Adaptive: {outputs['loss_dict'].get('loss_adaptive', 0):.4f}"
        if 'loss_distill' in outputs.get('loss_dict', {}):
            msg += f" | Distill: {outputs['loss_dict'].get('loss_distill', 0):.4f}"
        print(msg)

    def early_stopping_check(
        self,
        epoch: int,
        val_loss: float,
        ckpt_path: str,
    ) -> bool:
        """早停检查"""
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(epoch, ckpt_path)
            return False

        if score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping: {self.counter}/{self.early_stopping_patience}")
            if self.counter >= self.early_stopping_patience and epoch > self.early_stopping_stop_epoch:
                print("Early stopping triggered!")
                return True
        else:
            self.best_score = score
            self._save_checkpoint(epoch, ckpt_path)
            self.counter = 0

        return False

    def _save_checkpoint(self, epoch: int, ckpt_path: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_score': self.best_score,
        }
        torch.save(checkpoint, ckpt_path)


def train_fold(
    fold: int,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    args,
    results_dir: str,
) -> Dict[str, float]:
    """训练单个 fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold}")
    print(f"{'='*60}")

    # 优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.reg,
    )

    # 训练器
    trainer = FusionTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        num_tasks=args.num_tasks,
        task_loss_weight=args.task_loss_weight,
        distill_weight=args.distill_weight,
        use_distillation=args.use_distillation,
    )

    best_val_auc = 0.0
    best_results = {}

    for epoch in range(args.max_epochs):
        t0 = time.time()

        # 训练
        train_results = trainer.train_epoch(train_loader, epoch)
        print(f"\nEpoch {epoch} Train Loss: {train_results['loss']:.4f} | Time: {time.time()-t0:.1f}s")

        # 验证
        val_results = trainer.validate(val_loader)
        print(f"Epoch {epoch} Val Loss: {val_results['loss']:.4f} | AUC Direct: {val_results.get('auc_direct', 0):.4f} | AUC Adaptive: {val_results.get('auc_adaptive', 0):.4f}")

        # 保存最佳模型
        val_auc = val_results.get('auc_adaptive', val_results['auc_direct'])
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_results = val_results
            ckpt_path = os.path.join(results_dir, f'fold_{fold}_best.pt')
            trainer._save_checkpoint(epoch, ckpt_path)
            print(f"  -> New best! AUC: {best_val_auc:.4f}")

        # 早停
        if trainer.early_stopping_check(epoch, val_results['loss'], os.path.join(results_dir, f'fold_{fold}_last.pt')):
            print(f"Early stopping at epoch {epoch}")
            break

    return best_results
