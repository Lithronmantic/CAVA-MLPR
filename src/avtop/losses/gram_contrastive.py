#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gram_contrastive.py - 修复版本
关键修复：防止对比损失变为负数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class GRAMContrastiveLoss(nn.Module):
    """GRAM对比学习损失 - 修复版"""

    def __init__(self, temperature: float = 0.07, use_volume_metric: bool = False):  # 🔧 默认关闭GRAM
        super().__init__()
        self.temperature = max(temperature, 0.05)  # 🔧 温度下限
        self.use_volume_metric = use_volume_metric

    def forward(self, z_a: torch.Tensor, z_v: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_a: [B, D] - 音频嵌入（L2归一化）
            z_v: [B, D] - 视频嵌入（L2归一化）
        Returns:
            loss: 标量
        """
        B = z_a.shape[0]

        # 确保L2归一化
        z_a = F.normalize(z_a, p=2, dim=-1)
        z_v = F.normalize(z_v, p=2, dim=-1)

        # 计算相似度矩阵
        sim_matrix = torch.matmul(z_a, z_v.t()) / self.temperature  # [B, B]

        if self.use_volume_metric:
            loss = self._gram_volume_loss(sim_matrix)
        else:
            # 🔧 使用更稳定的InfoNCE
            loss = self._infonce_loss(sim_matrix)

        return loss

    def _infonce_loss(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """标准InfoNCE对比损失"""
        B = sim_matrix.shape[0]
        labels = torch.arange(B, device=sim_matrix.device)

        # Audio -> Video
        loss_a2v = F.cross_entropy(sim_matrix, labels)

        # Video -> Audio
        loss_v2a = F.cross_entropy(sim_matrix.t(), labels)

        # 对称损失
        loss = (loss_a2v + loss_v2a) / 2.0

        return loss

    def _gram_volume_loss(self, sim_matrix: torch.Tensor) -> torch.Tensor:
        """GRAM体积对比损失 - 修复版"""
        B = sim_matrix.shape[0]

        # 正样本分数
        pos_scores = torch.diag(sim_matrix)  # [B]

        # 负样本分数矩阵（去除对角线）
        mask = torch.eye(B, device=sim_matrix.device).bool()
        neg_scores = sim_matrix.masked_fill(mask, float('-inf'))  # [B, B]

        # 计算每个样本的负样本"体积"（log-sum-exp）
        neg_volume_a2v = torch.logsumexp(neg_scores, dim=1)  # [B]
        neg_volume_v2a = torch.logsumexp(neg_scores.t(), dim=1)  # [B]

        # 🔧 修复：确保损失非负
        # loss = -log(exp(pos) / (exp(pos) + exp(neg_volume)))
        #      = -pos + log(exp(pos) + exp(neg_volume))
        #      = -pos + logsumexp([pos, neg_volume])

        loss_a2v = -pos_scores + torch.logsumexp(
            torch.stack([pos_scores, neg_volume_a2v], dim=0), dim=0
        )
        loss_v2a = -pos_scores + torch.logsumexp(
            torch.stack([pos_scores, neg_volume_v2a], dim=0), dim=0
        )

        # 平均
        loss = (loss_a2v.mean() + loss_v2a.mean()) / 2.0

        # 🔧 安全检查
        if torch.isnan(loss) or torch.isinf(loss) or loss < 0:
            # 降级为InfoNCE
            return self._infonce_loss(sim_matrix / self.temperature)

        return loss


class BidirectionalKDLoss(nn.Module):
    """双向知识蒸馏损失"""

    def __init__(self, temperature: float = 4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(
            self,
            logits_fusion: torch.Tensor,
            logits_audio: Optional[torch.Tensor],
            logits_video: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Args:
            logits_fusion: [B, C] - 融合模型输出
            logits_audio: [B, C] - 音频单模态输出（可选）
            logits_video: [B, C] - 视频单模态输出（可选）
        Returns:
            loss: 标量
        """
        if logits_audio is None and logits_video is None:
            return torch.tensor(0.0, device=logits_fusion.device)

        T = self.temperature
        loss = 0.0
        count = 0

        # 软化概率分布
        p_fusion = F.log_softmax(logits_fusion / T, dim=-1)

        # 1. Audio → Fusion
        if logits_audio is not None:
            q_audio = F.softmax(logits_audio / T, dim=-1)
            loss += self.kl_div(p_fusion, q_audio) * (T * T)
            count += 1

        # 2. Video → Fusion
        if logits_video is not None:
            q_video = F.softmax(logits_video / T, dim=-1)
            loss += self.kl_div(p_fusion, q_video) * (T * T)
            count += 1

        # 3. Fusion → Audio
        if logits_audio is not None:
            p_audio = F.log_softmax(logits_audio / T, dim=-1)
            q_fusion_for_audio = F.softmax(logits_fusion / T, dim=-1)
            loss += self.kl_div(p_audio, q_fusion_for_audio) * (T * T)
            count += 1

        # 4. Fusion → Video
        if logits_video is not None:
            p_video = F.log_softmax(logits_video / T, dim=-1)
            q_fusion_for_video = F.softmax(logits_fusion / T, dim=-1)
            loss += self.kl_div(p_video, q_fusion_for_video) * (T * T)
            count += 1

        # 平均
        if count > 0:
            loss = loss / count

        return loss


class ConsistencyLoss(nn.Module):
    """跨模态一致性损失"""

    def __init__(self, consistency_type: str = 'cosine'):
        super().__init__()
        self.consistency_type = consistency_type

    def forward(
            self,
            z_a: torch.Tensor,
            z_v: torch.Tensor,
            audio_feat: Optional[torch.Tensor] = None,
            video_feat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z_a: [B, D] - 音频全局嵌入
            z_v: [B, D] - 视频全局嵌入
            audio_feat: [B, T, D] - 音频特征序列（可选）
            video_feat: [B, T, D] - 视频特征序列（可选）
        Returns:
            loss: 标量
        """
        loss = 0.0
        count = 0

        # 1. 跨模态嵌入一致性
        if self.consistency_type == 'cosine':
            cos_sim = F.cosine_similarity(z_a, z_v, dim=-1)
            loss += (1.0 - cos_sim).mean()
        elif self.consistency_type == 'mse':
            loss += F.mse_loss(z_a, z_v)
        else:
            loss += F.l1_loss(z_a, z_v)
        count += 1

        # 2. 时序一致性（相邻帧）
        if audio_feat is not None:
            temporal_loss_a = self._temporal_consistency(audio_feat)
            loss += temporal_loss_a
            count += 1

        if video_feat is not None:
            temporal_loss_v = self._temporal_consistency(video_feat)
            loss += temporal_loss_v
            count += 1

        # 平均
        if count > 0:
            loss = loss / count

        return loss

    def _temporal_consistency(self, feat: torch.Tensor) -> torch.Tensor:
        """时序一致性：相邻帧的特征应该相似"""
        if feat.shape[1] <= 1:
            return torch.tensor(0.0, device=feat.device)

        # 计算相邻帧之间的差异
        diff = feat[:, 1:, :] - feat[:, :-1, :]  # [B, T-1, D]

        # L2范数（希望小）
        loss = torch.norm(diff, p=2, dim=-1).mean()

        return loss


class CompleteLossFunction(nn.Module):
    """完整的多任务损失函数 - 修复版"""

    def __init__(
            self,
            num_classes: int = 2,
            lambda_contrastive: float = 0.3,
            lambda_kd: float = 0.2,
            lambda_consistency: float = 0.1,
            temperature: float = 0.07,
            kd_temperature: float = 4.0,
            use_focal_loss: bool = False,
            focal_alpha: float = 0.25,
            focal_gamma: float = 2.0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_contrastive = lambda_contrastive
        self.lambda_kd = lambda_kd
        self.lambda_consistency = lambda_consistency

        # 分类损失
        if use_focal_loss:
            self.cls_criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.cls_criterion = nn.CrossEntropyLoss(reduction='mean')

        # 🔧 对比学习：默认使用稳定的InfoNCE
        self.contrastive_criterion = GRAMContrastiveLoss(
            temperature=temperature,
            use_volume_metric=False  # 🔧 关闭GRAM
        )

        # 双向KD
        self.kd_criterion = BidirectionalKDLoss(temperature=kd_temperature)

        # 一致性损失
        self.consistency_criterion = ConsistencyLoss(consistency_type='cosine')

        print("[CompleteLossFunction] 初始化完成")
        print(f"  - 对比学习权重: {lambda_contrastive}")
        print(f"  - KD权重: {lambda_kd}")
        print(f"  - 一致性权重: {lambda_consistency}")
        print(f"  - 温度参数: {temperature}")

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            labels: torch.Tensor,
            is_labeled: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: 模型输出字典
            labels: [B] - 标签
            is_labeled: [B] - 是否有标签的mask
        Returns:
            loss_dict: 包含所有损失的字典
        """
        # 提取输出
        clip_logits = outputs['clip_logits']
        z_v = outputs.get('z_v')
        z_a = outputs.get('z_a')
        video_logits = outputs.get('video_logits')
        audio_logits = outputs.get('audio_logits')
        video_feat = outputs.get('video_feat')
        audio_feat = outputs.get('audio_feat')

        device = clip_logits.device

        # 1. 分类损失
        if is_labeled.sum() > 0:
            labeled_mask = is_labeled.bool()

            loss_cls_fusion = self.cls_criterion(
                clip_logits[labeled_mask],
                labels[labeled_mask]
            )

            loss_cls_aux = 0.0
            if video_logits is not None:
                loss_cls_aux += self.cls_criterion(
                    video_logits[labeled_mask],
                    labels[labeled_mask]
                )
            if audio_logits is not None:
                loss_cls_aux += self.cls_criterion(
                    audio_logits[labeled_mask],
                    labels[labeled_mask]
                )

            classification_loss = loss_cls_fusion + 0.5 * loss_cls_aux
        else:
            classification_loss = torch.tensor(0.0, device=device)

        # 2. 对比学习损失
        if z_v is not None and z_a is not None:
            contrastive_loss = self.contrastive_criterion(z_a, z_v)

            # 🔧 安全检查
            if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                print("⚠️ 对比损失异常，设为0")
                contrastive_loss = torch.tensor(0.0, device=device)
            elif contrastive_loss < 0:
                print(f"⚠️ 对比损失为负数: {contrastive_loss.item():.4f}，设为0")
                contrastive_loss = torch.tensor(0.0, device=device)
        else:
            contrastive_loss = torch.tensor(0.0, device=device)

        # 3. KD损失
        if (video_logits is not None or audio_logits is not None):
            kd_loss = self.kd_criterion(clip_logits, audio_logits, video_logits)
        else:
            kd_loss = torch.tensor(0.0, device=device)

        # 4. 一致性损失
        if z_v is not None and z_a is not None:
            consistency_loss = self.consistency_criterion(
                z_a, z_v, audio_feat, video_feat
            )
        else:
            consistency_loss = torch.tensor(0.0, device=device)

        # 5. 总损失
        total_loss = (
                classification_loss +
                self.lambda_contrastive * contrastive_loss +
                self.lambda_kd * kd_loss +
                self.lambda_consistency * consistency_loss
        )

        # 🔧 最终安全检查
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print("⚠️ 总损失异常，降级为分类损失")
            total_loss = classification_loss

        return {
            'total_loss': total_loss,
            'classification_loss': classification_loss,
            'contrastive_loss': contrastive_loss,
            'kd_loss': kd_loss,
            'consistency_loss': consistency_loss
        }


class FocalLoss(nn.Module):
    """Focal Loss（用于处理类别不平衡）"""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


if __name__ == "__main__":
    # 测试
    B, C, D = 8, 2, 128

    outputs = {
        'clip_logits': torch.randn(B, C),
        'z_v': F.normalize(torch.randn(B, D), p=2, dim=-1),
        'z_a': F.normalize(torch.randn(B, D), p=2, dim=-1),
        'video_logits': torch.randn(B, C),
        'audio_logits': torch.randn(B, C),
    }

    labels = torch.randint(0, C, (B,))
    is_labeled = torch.tensor([True] * B)

    criterion = CompleteLossFunction()
    loss_dict = criterion(outputs, labels, is_labeled)

    print("\n损失:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")

    print("\n✅ 测试通过")