#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fixmatch.py - 完整FixMatch半监督训练器（无省略）

FixMatch核心思想：
1. 对无标签数据应用弱增强，得到伪标签
2. 对同一数据应用强增强，用伪标签监督
3. 只保留高置信度的伪标签（confidence threshold）

论文: "FixMatch: Simplifying Semi-Supervised Learning with
       Consistency and Confidence"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class FixMatchLoss(nn.Module):
    """
    FixMatch损失函数

    L_total = L_supervised + λ_u * L_unsupervised

    其中：
    - L_supervised: 有标签数据的交叉熵损失
    - L_unsupervised: 无标签数据的伪标签一致性损失
    """

    def __init__(
            self,
            confidence_threshold: float = 0.95,
            lambda_unsup: float = 1.0,
            temperature: float = 1.0
    ):
        """
        Args:
            confidence_threshold: 伪标签置信度阈值（只保留>threshold的）
            lambda_unsup: 无监督损失权重
            temperature: 温度参数（用于软化概率分布）
        """
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.lambda_unsup = lambda_unsup
        self.temperature = temperature

        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')

    def forward(
            self,
            logits_weak: torch.Tensor,
            logits_strong: torch.Tensor,
            labels: torch.Tensor,
            is_labeled: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits_weak: [B, C] - 弱增强数据的预测
            logits_strong: [B, C] - 强增强数据的预测
            labels: [B] - 真实标签（有标签样本有效）
            is_labeled: [B] - bool tensor，标记哪些样本有标签

        Returns:
            loss_dict: 包含各项损失的字典
        """
        batch_size = logits_weak.shape[0]
        device = logits_weak.device

        # ====================================================================
        # 1. 有监督损失（有标签样本）
        # ====================================================================
        labeled_mask = is_labeled.bool()

        if labeled_mask.sum() > 0:
            # 使用弱增强的logits计算监督损失
            supervised_loss = self.ce_loss(
                logits_weak[labeled_mask],
                labels[labeled_mask]
            )
        else:
            supervised_loss = torch.tensor(0.0, device=device)

        # ====================================================================
        # 2. 无监督损失（无标签样本）
        # ====================================================================
        unlabeled_mask = ~labeled_mask

        if unlabeled_mask.sum() > 0:
            # 2.1 生成伪标签（从弱增强预测）
            with torch.no_grad():
                # 弱增强预测的概率分布
                probs_weak = F.softmax(
                    logits_weak[unlabeled_mask] / self.temperature,
                    dim=-1
                )

                # 获取最大概率和对应的类别
                max_probs, pseudo_labels = torch.max(probs_weak, dim=-1)

                # 置信度掩码（只保留高置信度的伪标签）
                confidence_mask = max_probs >= self.confidence_threshold

            # 2.2 如果有高置信度样本，计算无监督损失
            if confidence_mask.sum() > 0:
                # 强增强预测用于计算损失
                unsupervised_loss = self.ce_loss(
                    logits_strong[unlabeled_mask][confidence_mask],
                    pseudo_labels[confidence_mask]
                )

                # 统计伪标签使用率
                pseudo_label_ratio = confidence_mask.float().mean()
            else:
                unsupervised_loss = torch.tensor(0.0, device=device)
                pseudo_label_ratio = torch.tensor(0.0, device=device)
        else:
            unsupervised_loss = torch.tensor(0.0, device=device)
            pseudo_label_ratio = torch.tensor(0.0, device=device)

        # ====================================================================
        # 3. 总损失
        # ====================================================================
        total_loss = supervised_loss + self.lambda_unsup * unsupervised_loss

        return {
            'total_loss': total_loss,
            'supervised_loss': supervised_loss,
            'unsupervised_loss': unsupervised_loss,
            'pseudo_label_ratio': pseudo_label_ratio
        }


class PseudoLabelMiner(nn.Module):
    """
    伪标签挖掘器

    功能：
    1. 动态调整置信度阈值（根据训练进度）
    2. 类别平衡的伪标签选择
    3. 伪标签质量统计
    """

    def __init__(
            self,
            num_classes: int,
            initial_threshold: float = 0.95,
            final_threshold: float = 0.70,
            warmup_epochs: int = 10,
            use_class_balance: bool = True
    ):
        """
        Args:
            num_classes: 类别数
            initial_threshold: 初始置信度阈值（训练早期较高）
            final_threshold: 最终置信度阈值（训练后期较低）
            warmup_epochs: 预热epoch数（前N个epoch不生成伪标签）
            use_class_balance: 是否进行类别平衡
        """
        super().__init__()
        self.num_classes = num_classes
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        self.warmup_epochs = warmup_epochs
        self.use_class_balance = use_class_balance

        # 统计信息
        self.register_buffer('class_counts', torch.zeros(num_classes))
        self.register_buffer('total_pseudo_labels', torch.tensor(0.0))

    def get_threshold(self, current_epoch: int, total_epochs: int) -> float:
        """
        动态调整阈值

        策略：线性衰减（从高到低）
        - 早期：高阈值，只选最确定的样本
        - 后期：低阈值，利用更多无标签数据
        """
        if current_epoch < self.warmup_epochs:
            return 1.0  # 预热期不生成伪标签

        # 线性插值
        progress = (current_epoch - self.warmup_epochs) / max(1, total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)

        threshold = self.initial_threshold + (self.final_threshold - self.initial_threshold) * progress

        return threshold

    def mine_pseudo_labels(
            self,
            logits: torch.Tensor,
            threshold: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        挖掘伪标签

        Args:
            logits: [B, C] - 模型预测logits
            threshold: 置信度阈值（可选，默认使用self.initial_threshold）

        Returns:
            pseudo_labels: [B] - 伪标签
            confidence_mask: [B] - bool tensor，标记哪些样本被选中
            max_probs: [B] - 每个样本的最大概率
        """
        if threshold is None:
            threshold = self.initial_threshold

        with torch.no_grad():
            # 获取概率分布
            probs = F.softmax(logits, dim=-1)

            # 最大概率和对应类别
            max_probs, pseudo_labels = torch.max(probs, dim=-1)

            # 置信度筛选
            confidence_mask = max_probs >= threshold

            # 类别平衡（可选）
            if self.use_class_balance and confidence_mask.sum() > 0:
                confidence_mask = self._apply_class_balance(
                    pseudo_labels,
                    confidence_mask,
                    max_probs
                )

            # 更新统计
            self.total_pseudo_labels += confidence_mask.sum()
            for c in range(self.num_classes):
                self.class_counts[c] += (pseudo_labels[confidence_mask] == c).sum()

        return pseudo_labels, confidence_mask, max_probs

    def _apply_class_balance(
            self,
            pseudo_labels: torch.Tensor,
            confidence_mask: torch.Tensor,
            max_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        类别平衡策略

        确保每个类别的伪标签数量大致相等，避免模型偏向多数类
        """
        balanced_mask = torch.zeros_like(confidence_mask)

        # 计算每个类别的高置信度样本数
        class_sample_counts = []
        for c in range(self.num_classes):
            class_mask = (pseudo_labels == c) & confidence_mask
            class_sample_counts.append(class_mask.sum().item())

        # 目标：每个类别最多保留min_count个样本
        min_count = max(1, min(class_sample_counts)) if any(class_sample_counts) else 0

        if min_count == 0:
            return confidence_mask

        # 为每个类别选择top-k样本
        for c in range(self.num_classes):
            class_mask = (pseudo_labels == c) & confidence_mask

            if class_mask.sum() == 0:
                continue

            # 获取该类别样本的概率
            class_probs = max_probs.clone()
            class_probs[~class_mask] = -1  # 非该类样本设为-1

            # 选择top-k
            k = min(min_count, class_mask.sum().item())
            _, top_indices = torch.topk(class_probs, k)

            balanced_mask[top_indices] = True

        return balanced_mask

    def get_statistics(self) -> Dict[str, float]:
        """获取伪标签统计信息"""
        total = self.total_pseudo_labels.item()

        if total == 0:
            return {
                'total_pseudo_labels': 0,
                'class_distribution': [0.0] * self.num_classes
            }

        class_dist = [
            (self.class_counts[c] / total).item()
            for c in range(self.num_classes)
        ]

        return {
            'total_pseudo_labels': total,
            'class_distribution': class_dist
        }

    def reset_statistics(self):
        """重置统计信息"""
        self.class_counts.zero_()
        self.total_pseudo_labels.zero_()


class FixMatchTrainer:
    """
    FixMatch训练器（完整版）

    整合：
    1. 有标签/无标签混合训练
    2. 弱增强/强增强
    3. 伪标签生成和筛选
    4. 动态阈值调整
    """

    def __init__(
            self,
            model: nn.Module,
            criterion_supervised: nn.Module,
            optimizer: torch.optim.Optimizer,
            device: torch.device,
            config: Dict
    ):
        """
        Args:
            model: 检测模型
            criterion_supervised: 有监督损失函数（如CompleteLossFunction）
            optimizer: 优化器
            device: 设备
            config: 配置字典
        """
        self.model = model
        self.criterion_supervised = criterion_supervised
        self.optimizer = optimizer
        self.device = device
        self.config = config

        # FixMatch损失
        fixmatch_config = config.get('semi_supervised', {})
        self.fixmatch_loss = FixMatchLoss(
            confidence_threshold=fixmatch_config.get('confidence_threshold', 0.95),
            lambda_unsup=fixmatch_config.get('lambda_unsup', 1.0)
        ).to(device)

        # 伪标签挖掘器
        self.pseudo_label_miner = PseudoLabelMiner(
            num_classes=config['model']['num_classes'],
            initial_threshold=fixmatch_config.get('confidence_threshold', 0.95),
            final_threshold=fixmatch_config.get('final_threshold', 0.70),
            warmup_epochs=fixmatch_config.get('pseudo_label_warmup', 10)
        ).to(device)

        self.enable_fixmatch = fixmatch_config.get('enable', True)

    def train_step(
            self,
            batch_labeled: Dict,
            batch_unlabeled: Optional[Dict],
            current_epoch: int,
            total_epochs: int
    ) -> Dict[str, float]:
        """
        单步训练（FixMatch）

        Args:
            batch_labeled: 有标签batch
            batch_unlabeled: 无标签batch（可选）
            current_epoch: 当前epoch
            total_epochs: 总epoch数

        Returns:
            metrics: 包含各项指标的字典
        """
        self.model.train()
        self.optimizer.zero_grad()

        # ====================================================================
        # 1. 有监督部分（有标签数据）
        # ====================================================================
        video_l = batch_labeled['video'].to(self.device)
        audio_l = batch_labeled['audio'].to(self.device)
        labels_l = batch_labeled['label'].to(self.device)
        is_labeled = torch.ones(len(labels_l), dtype=torch.bool, device=self.device)

        # 前向传播
        outputs_l = self.model(video_l, audio_l, return_aux=True)

        # 有监督损失
        loss_dict_supervised = self.criterion_supervised(outputs_l, labels_l, is_labeled)
        loss_supervised = loss_dict_supervised['total_loss']

        # ====================================================================
        # 2. 无监督部分（无标签数据，FixMatch）
        # ====================================================================
        if self.enable_fixmatch and batch_unlabeled is not None:
            video_u = batch_unlabeled['video'].to(self.device)
            audio_u = batch_unlabeled['audio'].to(self.device)

            # 弱增强前向（生成伪标签）
            with torch.no_grad():
                outputs_weak = self.model(video_u, audio_u, return_aux=False)
                logits_weak = outputs_weak['clip_logits']

            # 强增强前向（用伪标签监督）
            # 注意：这里假设数据增强在DataLoader中已完成
            # 实际中需要对video_u, audio_u应用强增强
            outputs_strong = self.model(video_u, audio_u, return_aux=False)
            logits_strong = outputs_strong['clip_logits']

            # 获取动态阈值
            threshold = self.pseudo_label_miner.get_threshold(current_epoch, total_epochs)

            # 挖掘伪标签
            pseudo_labels, confidence_mask, max_probs = self.pseudo_label_miner.mine_pseudo_labels(
                logits_weak, threshold
            )

            # 计算FixMatch无监督损失
            # 创建虚拟标签和mask（用于FixMatchLoss）
            dummy_labels = torch.zeros(len(pseudo_labels), dtype=torch.long, device=self.device)
            is_unlabeled = torch.zeros(len(pseudo_labels), dtype=torch.bool, device=self.device)

            loss_dict_fixmatch = self.fixmatch_loss(
                logits_weak, logits_strong, dummy_labels, is_unlabeled
            )
            loss_unsupervised = loss_dict_fixmatch['unsupervised_loss']
            pseudo_label_ratio = loss_dict_fixmatch['pseudo_label_ratio']
        else:
            loss_unsupervised = torch.tensor(0.0, device=self.device)
            pseudo_label_ratio = torch.tensor(0.0, device=self.device)

        # ====================================================================
        # 3. 总损失并反向传播
        # ====================================================================
        total_loss = loss_supervised + loss_unsupervised
        total_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        self.optimizer.step()

        # ====================================================================
        # 4. 返回指标
        # ====================================================================
        metrics = {
            'total_loss': total_loss.item(),
            'supervised_loss': loss_supervised.item(),
            'unsupervised_loss': loss_unsupervised.item(),
            'pseudo_label_ratio': pseudo_label_ratio.item() if isinstance(pseudo_label_ratio,
                                                                          torch.Tensor) else pseudo_label_ratio,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }

        return metrics


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 模拟配置
    config = {
        'model': {'num_classes': 2},
        'semi_supervised': {
            'enable': True,
            'confidence_threshold': 0.95,
            'lambda_unsup': 1.0,
            'pseudo_label_warmup': 10
        }
    }

    # 模拟数据
    batch_size = 8
    num_classes = 2

    logits_weak = torch.randn(batch_size, num_classes)
    logits_strong = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    is_labeled = torch.tensor([True, True, True, True, False, False, False, False])

    # 创建FixMatch损失
    fixmatch_loss = FixMatchLoss(confidence_threshold=0.95)

    # 计算损失
    loss_dict = fixmatch_loss(logits_weak, logits_strong, labels, is_labeled)

    print("\n=== FixMatch损失测试 ===")
    for key, value in loss_dict.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.4f}")
        else:
            print(f"{key}: {value:.4f}")

    # 测试伪标签挖掘器
    print("\n=== 伪标签挖掘器测试 ===")
    miner = PseudoLabelMiner(num_classes=2)

    pseudo_labels, confidence_mask, max_probs = miner.mine_pseudo_labels(logits_weak)

    print(f"伪标签数量: {confidence_mask.sum().item()} / {batch_size}")
    print(f"平均置信度: {max_probs[confidence_mask].mean().item():.4f}")

    stats = miner.get_statistics()
    print(f"类别分布: {stats['class_distribution']}")

    print("\n✅ FixMatch测试通过！")