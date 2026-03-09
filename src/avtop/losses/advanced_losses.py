# src/avtop/losses/advanced_losses.py
"""
高级损失函数：专门针对不平衡数据
- Focal Loss: 聚焦难样本
- Class-Balanced Loss: 基于有效样本数的重加权
- 增强的BCE: 正类加权
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss: 解决类别不平衡和难易样本不平衡
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)

    Loss = -α * (1-p_t)^γ * log(p_t)

    Args:
        alpha: 正类权重，通常设为 n_neg/(n_pos+n_neg)
        gamma: 聚焦参数，γ=2常用，越大越关注难样本
        reduction: 'mean', 'sum', 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B,) 或 (B, num_classes) - 未经过sigmoid/softmax的logits
            targets: (B,) - 0/1标签
        """
        # 确保targets是long类型
        targets = targets.long()

        # 处理二分类
        if logits.dim() == 1:
            # (B,) -> (B, 2)
            logits = torch.stack([1 - logits, logits], dim=-1)

        # 计算概率
        probs = F.softmax(logits, dim=-1)

        # 获取目标类的概率
        targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (probs * targets_one_hot).sum(dim=-1)  # (B,)

        # Focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weight
        alpha_t = self.alpha * targets.float() + (1 - self.alpha) * (1 - targets.float())

        # Cross entropy
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Focal loss
        loss = alpha_t * focal_weight * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ClassBalancedLoss(nn.Module):
    """
    Class-Balanced Loss: 基于有效样本数的损失重加权
    Cui et al. "Class-Balanced Loss Based on Effective Number of Samples" (CVPR 2019)

    有效样本数: E_n = (1 - β^n) / (1 - β)
    权重: w = (1 - β) / (1 - β^n) = 1 / E_n

    Args:
        samples_per_class: 每个类的样本数 [n_neg, n_pos]
        beta: 重采样参数，0.9999 for long-tail, 0.99 for moderate
        loss_type: 'focal' or 'sigmoid' or 'softmax'
    """

    def __init__(self, samples_per_class: list, beta: float = 0.9999,
                 gamma: float = 2.0, loss_type: str = 'focal'):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type

        # 计算每个类的有效样本数
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class).float())
        weights = (1.0 - beta) / effective_num

        # 归一化权重
        weights = weights / weights.sum() * len(weights)
        self.register_buffer('weights', weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) - 未经过激活的logits
            targets: (B,) - 类别标签
        """
        targets = targets.long()

        if self.loss_type == 'focal':
            # 使用focal loss作为基础
            probs = F.softmax(logits, dim=-1)
            targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
            p_t = (probs * targets_one_hot).sum(dim=-1)

            focal_weight = (1 - p_t) ** self.gamma
            ce_loss = F.cross_entropy(logits, targets, weight=self.weights, reduction='none')
            loss = focal_weight * ce_loss
            return loss.mean()

        elif self.loss_type == 'softmax':
            return F.cross_entropy(logits, targets, weight=self.weights)

        elif self.loss_type == 'sigmoid':
            # Binary classification
            targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
            loss = F.binary_cross_entropy_with_logits(
                logits, targets_one_hot,
                weight=self.weights.unsqueeze(0).expand_as(logits),
                reduction='mean'
            )
            return loss

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")


class WeightedBCEWithLogitsLoss(nn.Module):
    """
    增强的BCE损失，带正类权重

    Args:
        pos_weight: 正类权重，通常设为 n_neg / n_pos
        reduction: 'mean', 'sum', 'none'
    """

    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, num_classes) 或 (B,) - logits
            targets: (B,) - 0/1标签
        """
        if logits.dim() == 1:
            # 单输出logit
            targets = targets.float()
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction=self.reduction
            )
        else:
            # 多类logits
            targets_one_hot = F.one_hot(targets.long(), num_classes=logits.size(-1)).float()
            # 只对正类应用pos_weight
            weight = torch.ones_like(logits)
            weight[:, 1] = self.pos_weight  # 假设正类是索引1
            loss = F.binary_cross_entropy_with_logits(
                logits, targets_one_hot, weight=weight, reduction=self.reduction
            )

        return loss


class CombinedLoss(nn.Module):
    """
    组合损失：同时考虑分类损失和额外的正则项

    Example:
        loss = α * focal_loss + β * ranking_loss + γ * regularization
    """

    def __init__(self,
                 cls_loss_type: str = 'focal',
                 cls_weight: float = 1.0,
                 ranking_weight: float = 0.5,
                 reg_weight: float = 0.01,
                 **cls_kwargs):
        super().__init__()

        self.cls_weight = cls_weight
        self.ranking_weight = ranking_weight
        self.reg_weight = reg_weight

        # 分类损失
        if cls_loss_type == 'focal':
            self.cls_loss = FocalLoss(**cls_kwargs)
        elif cls_loss_type == 'cb':
            self.cls_loss = ClassBalancedLoss(**cls_kwargs)
        elif cls_loss_type == 'weighted_bce':
            self.cls_loss = WeightedBCEWithLogitsLoss(**cls_kwargs)
        else:
            self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, targets: torch.Tensor,
                ranking_loss: Optional[torch.Tensor] = None,
                reg_loss: Optional[torch.Tensor] = None) -> dict:
        """
        Args:
            outputs: 模型输出字典，包含 'clip_logits'
            targets: 真实标签
            ranking_loss: 可选的ranking损失
            reg_loss: 可选的正则化损失

        Returns:
            包含各项损失的字典
        """
        # 分类损失
        logits = outputs['clip_logits']
        cls_loss = self.cls_loss(logits, targets)

        total_loss = self.cls_weight * cls_loss
        loss_dict = {'cls': cls_loss.item()}

        # Ranking损失
        if ranking_loss is not None:
            total_loss = total_loss + self.ranking_weight * ranking_loss
            loss_dict['ranking'] = ranking_loss.item()

        # 正则化
        if reg_loss is not None:
            total_loss = total_loss + self.reg_weight * reg_loss
            loss_dict['reg'] = reg_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict


def create_loss_function(loss_type: str, samples_per_class: Optional[list] = None,
                         **kwargs):
    """
    便捷函数：创建损失函数

    Args:
        loss_type: 'focal', 'cb', 'weighted_bce', 'ce'
        samples_per_class: [n_neg, n_pos] 用于cb loss
        **kwargs: 额外参数

    Example:
        # Focal Loss
        loss_fn = create_loss_function('focal', alpha=0.75, gamma=2.0)

        # Class-Balanced Loss
        loss_fn = create_loss_function('cb', samples_per_class=[800, 200], beta=0.9999)

        # Weighted BCE
        loss_fn = create_loss_function('weighted_bce', pos_weight=4.0)
    """
    if loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'cb':
        if samples_per_class is None:
            raise ValueError("samples_per_class required for CB loss")
        return ClassBalancedLoss(samples_per_class, **kwargs)
    elif loss_type == 'weighted_bce':
        return WeightedBCEWithLogitsLoss(**kwargs)
    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")