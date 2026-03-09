#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双向知识蒸馏损失 (Bidirectional Knowledge Distillation)

目标：对齐音频和视频模态的输出分布

公式：
L_KD = KL(p_v || p_a) + KL(p_a || p_v)

其中：
- p_v: 视频分支的输出概率分布
- p_a: 音频分支的输出概率分布
- KL: Kullback-Leibler散度

特点：
- 双向：视频↔音频互相指导
- 支持温度缩放（temperature scaling）
- 支持clip级和segment级蒸馏
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalKDLoss(nn.Module):
    """
    双向知识蒸馏损失
    
    用途：
    1. 让视频和音频分支学到一致的判别特征
    2. 一个模态学到的知识可以传递给另一个模态
    3. 提高模态互补性和鲁棒性
    """
    def __init__(self, temperature=2.0, reduction='batchmean'):
        """
        Args:
            temperature: 温度参数（>1使分布更平滑）
            reduction: KL散度的reduce方式
                - 'batchmean': 除以batch size（标准KL）
                - 'mean': 除以所有元素数量
                - 'sum': 求和
        """
        super().__init__()
        
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, logits_a, logits_v, mode='bidirectional'):
        """
        计算双向KD损失
        
        Args:
            logits_a: 音频分支的logits
                - [B, num_classes] for clip-level
                - [B, T, num_classes] for segment-level
            logits_v: 视频分支的logits（形状同上）
            mode: str
                - 'bidirectional': 双向KD
                - 'a2v': 音频→视频（音频是教师）
                - 'v2a': 视频→音频（视频是教师）
        
        Returns:
            loss: scalar
            metrics: dict
        """
        T = self.temperature
        
        # 1. 应用温度缩放并计算软标签（soft targets）
        p_a = F.softmax(logits_a / T, dim=-1)  # 音频分布
        p_v = F.softmax(logits_v / T, dim=-1)  # 视频分布
        
        # log概率（用于KL散度计算）
        log_p_a = F.log_softmax(logits_a / T, dim=-1)
        log_p_v = F.log_softmax(logits_v / T, dim=-1)
        
        # 2. 计算KL散度
        if mode == 'bidirectional' or mode == 'v2a':
            # KL(p_v || p_a): 视频是教师，音频是学生
            kl_v2a = F.kl_div(log_p_a, p_v, reduction=self.reduction)
        else:
            kl_v2a = 0.0
        
        if mode == 'bidirectional' or mode == 'a2v':
            # KL(p_a || p_v): 音频是教师，视频是学生
            kl_a2v = F.kl_div(log_p_v, p_a, reduction=self.reduction)
        else:
            kl_a2v = 0.0
        
        # 3. 组合损失
        if mode == 'bidirectional':
            loss = (kl_v2a + kl_a2v) / 2
        elif mode == 'v2a':
            loss = kl_v2a
        else:  # a2v
            loss = kl_a2v
        
        # 温度平方缩放（标准KD做法）
        loss = loss * (T ** 2)
        
        # 4. 计算指标
        with torch.no_grad():
            # JS散度（对称的距离度量）
            m = (p_a + p_v) / 2
            js_div = (F.kl_div(torch.log(m + 1e-10), p_a, reduction='batchmean') +
                     F.kl_div(torch.log(m + 1e-10), p_v, reduction='batchmean')) / 2
            
            # 预测一致性（hard predictions）
            pred_a = logits_a.argmax(dim=-1)
            pred_v = logits_v.argmax(dim=-1)
            agreement = (pred_a == pred_v).float().mean()
        
        metrics = {
            'kd_loss': loss.item(),
            'kl_v2a': kl_v2a if isinstance(kl_v2a, float) else kl_v2a.item(),
            'kl_a2v': kl_a2v if isinstance(kl_a2v, float) else kl_a2v.item(),
            'js_divergence': js_div.item(),
            'prediction_agreement': agreement.item()
        }
        
        return loss, metrics


class TriModalKDLoss(nn.Module):
    """
    三模态知识蒸馏：Fusion ↔ Video ↔ Audio
    
    用途：
    - 融合模型作为教师，指导单模态
    - 单模态之间也互相学习
    
    损失：
    L = KL(p_v || p_f) + KL(p_a || p_f) + α·[KL(p_v || p_a) + KL(p_a || p_v)]
    """
    def __init__(self, temperature=2.0, bimodal_weight=0.5):
        super().__init__()
        
        self.temperature = temperature
        self.bimodal_weight = bimodal_weight  # α
    
    def forward(self, logits_fusion, logits_video, logits_audio):
        """
        Args:
            logits_fusion: [B, C] - 融合模型输出
            logits_video: [B, C] - 视频分支输出
            logits_audio: [B, C] - 音频分支输出
        
        Returns:
            loss: scalar
            metrics: dict
        """
        T = self.temperature
        
        # 软标签
        p_f = F.softmax(logits_fusion / T, dim=-1)
        p_v = F.softmax(logits_video / T, dim=-1)
        p_a = F.softmax(logits_audio / T, dim=-1)
        
        log_p_v = F.log_softmax(logits_video / T, dim=-1)
        log_p_a = F.log_softmax(logits_audio / T, dim=-1)
        
        # 1. 融合→单模态（主要损失）
        kl_f2v = F.kl_div(log_p_v, p_f, reduction='batchmean')
        kl_f2a = F.kl_div(log_p_a, p_f, reduction='batchmean')
        
        loss_fusion_to_single = (kl_f2v + kl_f2a) / 2
        
        # 2. 单模态之间（辅助损失）
        kl_v2a = F.kl_div(log_p_a, p_v, reduction='batchmean')
        kl_a2v = F.kl_div(log_p_v, p_a, reduction='batchmean')
        
        loss_bimodal = (kl_v2a + kl_a2v) / 2
        
        # 3. 总损失
        loss = loss_fusion_to_single + self.bimodal_weight * loss_bimodal
        loss = loss * (T ** 2)
        
        # 指标
        with torch.no_grad():
            pred_f = logits_fusion.argmax(dim=-1)
            pred_v = logits_video.argmax(dim=-1)
            pred_a = logits_audio.argmax(dim=-1)
            
            agree_fv = (pred_f == pred_v).float().mean()
            agree_fa = (pred_f == pred_a).float().mean()
            agree_va = (pred_v == pred_a).float().mean()
        
        metrics = {
            'kd_fusion_to_single': loss_fusion_to_single.item(),
            'kd_bimodal': loss_bimodal.item(),
            'kd_total': loss.item(),
            'agree_fusion_video': agree_fv.item(),
            'agree_fusion_audio': agree_fa.item(),
            'agree_video_audio': agree_va.item()
        }
        
        return loss, metrics


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("Bidirectional KD Loss 测试")
    print("="*70)
    
    # 参数
    B = 8
    T = 16
    C = 2  # 二分类
    temperature = 2.0
    
    # 创建损失函数
    criterion = BidirectionalKDLoss(temperature=temperature)
    
    # 测试1: Clip-level KD
    print("\n[测试1] Clip-level KD")
    logits_audio = torch.randn(B, C, requires_grad=True)
    logits_video = torch.randn(B, C, requires_grad=True)
    
    loss, metrics = criterion(logits_audio, logits_video, mode='bidirectional')
    
    print(f"  Loss: {loss.item():.4f}")
    print(f"  KL(V||A): {metrics['kl_v2a']:.4f}")
    print(f"  KL(A||V): {metrics['kl_a2v']:.4f}")
    print(f"  JS Divergence: {metrics['js_divergence']:.4f}")
    print(f"  Agreement: {metrics['prediction_agreement']*100:.1f}%")
    
    # 梯度测试
    loss.backward()
    print(f"  ✅ 梯度计算成功")
    
    # 测试2: Segment-level KD
    print(f"\n[测试2] Segment-level KD")
    logits_audio_seg = torch.randn(B, T, C, requires_grad=True)
    logits_video_seg = torch.randn(B, T, C, requires_grad=True)
    
    loss_seg, metrics_seg = criterion(logits_audio_seg, logits_video_seg)
    
    print(f"  Loss: {loss_seg.item():.4f}")
    print(f"  Agreement: {metrics_seg['prediction_agreement']*100:.1f}%")
    
    loss_seg.backward()
    print(f"  ✅ 梯度计算成功")
    
    # 测试3: Tri-Modal KD
    print(f"\n{'='*70}")
    print("Tri-Modal KD Loss 测试")
    print(f"{'='*70}")
    
    criterion_tri = TriModalKDLoss(temperature=temperature, bimodal_weight=0.5)
    
    logits_fusion = torch.randn(B, C, requires_grad=True)
    logits_video = torch.randn(B, C, requires_grad=True)
    logits_audio = torch.randn(B, C, requires_grad=True)
    
    loss_tri, metrics_tri = criterion_tri(logits_fusion, logits_video, logits_audio)
    
    print(f"\n结果:")
    print(f"  Total Loss: {loss_tri.item():.4f}")
    print(f"  Fusion→Single: {metrics_tri['kd_fusion_to_single']:.4f}")
    print(f"  Bimodal: {metrics_tri['kd_bimodal']:.4f}")
    print(f"\n一致性:")
    print(f"  Fusion-Video: {metrics_tri['agree_fusion_video']*100:.1f}%")
    print(f"  Fusion-Audio: {metrics_tri['agree_fusion_audio']*100:.1f}%")
    print(f"  Video-Audio: {metrics_tri['agree_video_audio']*100:.1f}%")
    
    loss_tri.backward()
    print(f"\n✅ 所有测试通过!")
