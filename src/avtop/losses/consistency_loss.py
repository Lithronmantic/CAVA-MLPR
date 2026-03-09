#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一致性正则化损失 (Consistency Regularization)

目标：对同一未标注样本的不同视图/模态预测保持一致

应用场景：
1. 同一样本在不同数据增强下的输出应该一致
2. 同一样本在不同模态（音频/视频/融合）下的输出应该一致
3. 用于半监督学习，稳定未标注数据的训练

公式：
L_cons = MSE(p1, p2) or KL(p1 || p2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """
    一致性损失
    
    支持多种一致性度量：
    - MSE: 均方误差
    - KL: KL散度
    - JS: JS散度（对称）
    - Cosine: 余弦距离
    """
    def __init__(self, 
                 consistency_type='mse',
                 temperature=1.0,
                 sharpening_temperature=0.5):
        """
        Args:
            consistency_type: 'mse', 'kl', 'js', 'cosine'
            temperature: KL散度的温度
            sharpening_temperature: 锐化预测的温度（<1使预测更sharp）
        """
        super().__init__()
        
        self.consistency_type = consistency_type
        self.temperature = temperature
        self.sharpening_temperature = sharpening_temperature
    
    def sharpen(self, p, T):
        """
        锐化概率分布（提高置信度）
        
        p_sharp = p^(1/T) / Σ p^(1/T)
        """
        p_sharp = p ** (1.0 / T)
        return p_sharp / p_sharp.sum(dim=-1, keepdim=True)
    
    def forward(self, logits1, logits2, sharpen=False, weight=None):
        """
        计算一致性损失
        
        Args:
            logits1: [B, C] or [B, T, C] - 第一个预测
            logits2: [B, C] or [B, T, C] - 第二个预测
            sharpen: bool - 是否锐化预测
            weight: [B] or [B, T] - 样本权重（可选）
        
        Returns:
            loss: scalar
            metrics: dict
        """
        # 1. 转换为概率分布
        p1 = F.softmax(logits1 / self.temperature, dim=-1)
        p2 = F.softmax(logits2 / self.temperature, dim=-1)
        
        # 2. 可选：锐化预测（让模型更confident）
        if sharpen:
            p1 = self.sharpen(p1, self.sharpening_temperature)
            p2 = self.sharpen(p2, self.sharpening_temperature)
        
        # 3. 计算一致性损失
        if self.consistency_type == 'mse':
            # 均方误差
            loss_raw = F.mse_loss(p1, p2, reduction='none').mean(dim=-1)
            
        elif self.consistency_type == 'kl':
            # KL散度（不对称）
            log_p1 = torch.log(p1 + 1e-10)
            loss_raw = F.kl_div(log_p1, p2, reduction='none').sum(dim=-1)
            
        elif self.consistency_type == 'js':
            # JS散度（对称）
            m = (p1 + p2) / 2
            log_m = torch.log(m + 1e-10)
            kl1 = F.kl_div(log_m, p1, reduction='none').sum(dim=-1)
            kl2 = F.kl_div(log_m, p2, reduction='none').sum(dim=-1)
            loss_raw = (kl1 + kl2) / 2
            
        elif self.consistency_type == 'cosine':
            # 余弦距离 = 1 - cosine_similarity
            cos_sim = F.cosine_similarity(p1, p2, dim=-1)
            loss_raw = 1 - cos_sim
            
        else:
            raise ValueError(f"Unknown consistency type: {self.consistency_type}")
        
        # 4. 应用权重
        if weight is not None:
            loss_raw = loss_raw * weight
            loss = loss_raw.sum() / weight.sum()
        else:
            loss = loss_raw.mean()
        
        # 5. 计算指标
        with torch.no_grad():
            # 预测一致性（硬标签）
            pred1 = logits1.argmax(dim=-1)
            pred2 = logits2.argmax(dim=-1)
            agreement = (pred1 == pred2).float().mean()
            
            # 平均置信度
            conf1 = p1.max(dim=-1)[0].mean()
            conf2 = p2.max(dim=-1)[0].mean()
        
        metrics = {
            'consistency_loss': loss.item(),
            'prediction_agreement': agreement.item(),
            'confidence_1': conf1.item(),
            'confidence_2': conf2.item()
        }
        
        return loss, metrics


class MultiViewConsistency(nn.Module):
    """
    多视图一致性：对N个视图的预测施加一致性约束
    
    应用：
    - 同一样本的多种数据增强
    - 同一样本的多个模态（音频、视频、融合）
    """
    def __init__(self, consistency_type='mse', temperature=1.0):
        super().__init__()
        
        self.consistency_type = consistency_type
        self.temperature = temperature
        self.base_loss = ConsistencyLoss(consistency_type, temperature)
    
    def forward(self, logits_list, mode='pairwise'):
        """
        Args:
            logits_list: list of [B, C] - 多个视图的预测
            mode: 'pairwise' or 'mean_teacher'
                - pairwise: 所有pair之间的一致性
                - mean_teacher: 与均值预测的一致性
        
        Returns:
            loss: scalar
            metrics: dict
        """
        n_views = len(logits_list)
        
        if n_views < 2:
            return torch.tensor(0.0, device=logits_list[0].device), {}
        
        if mode == 'pairwise':
            # 所有pair之间
            losses = []
            for i in range(n_views):
                for j in range(i+1, n_views):
                    loss_ij, _ = self.base_loss(logits_list[i], logits_list[j])
                    losses.append(loss_ij)
            
            loss = sum(losses) / len(losses)
            
        elif mode == 'mean_teacher':
            # 与均值预测比较
            # 先计算所有视图的平均概率
            probs = [F.softmax(logits / self.temperature, dim=-1) 
                    for logits in logits_list]
            mean_prob = sum(probs) / len(probs)
            
            # 每个视图与均值的一致性
            losses = []
            for logits in logits_list:
                prob = F.softmax(logits / self.temperature, dim=-1)
                loss_i = F.mse_loss(prob, mean_prob)
                losses.append(loss_i)
            
            loss = sum(losses) / len(losses)
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 计算多视图一致性指标
        with torch.no_grad():
            preds = [logits.argmax(dim=-1) for logits in logits_list]
            
            # 计算所有pair的一致性
            agreements = []
            for i in range(n_views):
                for j in range(i+1, n_views):
                    agree = (preds[i] == preds[j]).float().mean()
                    agreements.append(agree)
            
            avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0
        
        metrics = {
            'multiview_consistency': loss.item(),
            'multiview_agreement': avg_agreement if isinstance(avg_agreement, float) else avg_agreement.item()
        }
        
        return loss, metrics


class TemporalConsistency(nn.Module):
    """
    时序一致性：相邻帧的预测应该平滑
    
    用于segment-level预测的正则化
    """
    def __init__(self, smoothness_weight=1.0):
        super().__init__()
        self.smoothness_weight = smoothness_weight
    
    def forward(self, logits_seq):
        """
        Args:
            logits_seq: [B, T, C] - 序列预测
        
        Returns:
            loss: scalar
        """
        # 计算相邻帧概率的差异
        probs = F.softmax(logits_seq, dim=-1)  # [B, T, C]
        
        # 相邻帧的差异
        diff = probs[:, 1:] - probs[:, :-1]  # [B, T-1, C]
        
        # L2范数
        loss = (diff ** 2).mean()
        
        return loss * self.smoothness_weight


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Consistency Loss 测试")
    print("="*70)
    
    B, C = 8, 2
    
    # 测试1: 基础一致性损失
    print("\n[测试1] 基础一致性损失")
    
    for cons_type in ['mse', 'kl', 'js', 'cosine']:
        criterion = ConsistencyLoss(consistency_type=cons_type)
        
        logits1 = torch.randn(B, C, requires_grad=True)
        logits2 = logits1 + torch.randn(B, C) * 0.1  # 添加小噪声
        
        loss, metrics = criterion(logits1, logits2)
        
        print(f"\n  {cons_type.upper()}:")
        print(f"    Loss: {loss.item():.4f}")
        print(f"    Agreement: {metrics['prediction_agreement']*100:.1f}%")
        
        loss.backward()
    
    print("\n  ✅ 所有一致性类型测试通过")
    
    # 测试2: 多视图一致性
    print(f"\n{'='*70}")
    print("[测试2] 多视图一致性")
    print(f"{'='*70}")
    
    criterion_mv = MultiViewConsistency()
    
    # 3个视图
    base_logits = torch.randn(B, C)
    logits_views = [
        base_logits + torch.randn(B, C) * 0.1 for _ in range(3)
    ]
    
    for mode in ['pairwise', 'mean_teacher']:
        loss_mv, metrics_mv = criterion_mv(logits_views, mode=mode)
        
        print(f"\n  {mode}:")
        print(f"    Loss: {loss_mv.item():.4f}")
        print(f"    Agreement: {metrics_mv['multiview_agreement']*100:.1f}%")
    
    # 测试3: 时序一致性
    print(f"\n{'='*70}")
    print("[测试3] 时序一致性")
    print(f"{'='*70}")
    
    criterion_temp = TemporalConsistency()
    
    T = 16
    logits_seq = torch.randn(B, T, C, requires_grad=True)
    
    loss_temp = criterion_temp(logits_seq)
    
    print(f"\n  Loss: {loss_temp.item():.4f}")
    
    loss_temp.backward()
    
    print(f"\n✅ 所有测试通过!")
