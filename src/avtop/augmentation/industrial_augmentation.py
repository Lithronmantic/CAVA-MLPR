#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class SpecAugment(nn.Module):
    """
    SpecAugment - 音频频谱增强
    
    来自论文：SpecAugment: A Simple Data Augmentation Method for ASR (Google Brain)
    
    操作：
    1. Time Masking - 遮蔽时间轴的连续片段
    2. Frequency Masking - 遮蔽频率轴的连续片段
    3. Time Warping - 时间轴的轻微扭曲（可选）
    
    Args:
        freq_mask_param: 频率遮蔽的最大宽度
        time_mask_param: 时间遮蔽的最大宽度
        num_freq_masks: 频率遮蔽的数量
        num_time_masks: 时间遮蔽的数量
    """
    def __init__(
        self,
        freq_mask_param=15,
        time_mask_param=20,
        num_freq_masks=2,
        num_time_masks=2,
        mask_value=0.0
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
        
    def frequency_masking(self, spec):
        """
        频率遮蔽
        
        Args:
            spec: [B, T, F] - 频谱图
        """
        B, T, F = spec.shape
        spec_aug = spec.clone()
        
        for _ in range(self.num_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, F - f)
            spec_aug[:, :, f0:f0+f] = self.mask_value
        
        return spec_aug
    
    def time_masking(self, spec):
        """
        时间遮蔽
        
        Args:
            spec: [B, T, F] - 频谱图
        """
        B, T, F = spec.shape
        spec_aug = spec.clone()
        
        for _ in range(self.num_time_masks):
            t = random.randint(0, min(self.time_mask_param, T - 1))
            t0 = random.randint(0, T - t)
            spec_aug[:, t0:t0+t, :] = self.mask_value
        
        return spec_aug
    
    def forward(self, spec):
        """
        Args:
            spec: [B, T, F] - 音频频谱
        """
        # 频率遮蔽
        spec = self.frequency_masking(spec)
        
        # 时间遮蔽
        spec = self.time_masking(spec)
        
        return spec


class MixUp(nn.Module):
    """
    MixUp增强 - 混合两个样本
    
    公式：
        x_mixed = λ * x_i + (1 - λ) * x_j
        y_mixed = λ * y_i + (1 - λ) * y_j
    
    其中 λ ~ Beta(α, α)
    
    Args:
        alpha: Beta分布参数（推荐0.2-0.4）
    """
    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        
    def forward(self, x, y):
        """
        Args:
            x: [B, ...] - 输入数据
            y: [B, C] - one-hot标签
        """
        batch_size = x.shape[0]
        
        # 采样混合系数
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0
        
        # 随机排列索引
        index = torch.randperm(batch_size, device=x.device)
        
        # 混合数据
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # 混合标签
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return mixed_x, mixed_y, lam


class CutMix(nn.Module):
    """
    CutMix增强 - 剪切并混合
    
    操作：
    1. 从一个样本中裁剪出一个区域
    2. 粘贴到另一个样本的相同位置
    3. 标签按面积比例混合
    
    Args:
        alpha: Beta分布参数
    """
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
        
    def rand_bbox(self, size, lam):
        """生成随机边界框"""
        W = size[2]
        H = size[3]
        
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # 随机中心点
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # 计算边界
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def forward(self, x, y):
        """
        Args:
            x: [B, T, C, H, W] - 视频数据
            y: [B, C] - one-hot标签
        """
        batch_size = x.shape[0]
        
        # 采样混合系数
        lam = np.random.beta(self.alpha, self.alpha)
        
        # 随机排列索引
        index = torch.randperm(batch_size, device=x.device)
        
        # 生成边界框
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.shape, lam)
        
        # CutMix
        x[:, :, :, bbx1:bbx2, bby1:bby2] = x[index, :, :, bbx1:bbx2, bby1:bby2]
        
        # 调整λ（根据实际裁剪面积）
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.shape[-1] * x.shape[-2]))
        
        # 混合标签
        mixed_y = lam * y + (1 - lam) * y[index]
        
        return x, mixed_y, lam


class TemporalConsistencyAugment(nn.Module):
    """
    时序一致性增强 - 保持时序信息的同时进行增强
    
    策略：
    1. 时序平滑 - 避免相邻帧差异过大
    2. 时序抖动 - 轻微的时序扰动
    3. 帧采样 - 随机丢弃/重复帧
    """
    def __init__(
        self,
        temporal_jitter_prob=0.3,
        temporal_jitter_range=2,
        frame_drop_prob=0.1
    ):
        super().__init__()
        self.temporal_jitter_prob = temporal_jitter_prob
        self.temporal_jitter_range = temporal_jitter_range
        self.frame_drop_prob = frame_drop_prob
        
    def temporal_jitter(self, x):
        """
        时序抖动 - 轻微打乱帧顺序
        
        Args:
            x: [B, T, ...] - 时序数据
        """
        if random.random() > self.temporal_jitter_prob:
            return x
        
        B, T = x.shape[:2]
        
        # 生成抖动索引
        indices = torch.arange(T)
        for i in range(T):
            if random.random() < 0.3:  # 30%的帧抖动
                offset = random.randint(-self.temporal_jitter_range, 
                                       self.temporal_jitter_range)
                new_idx = max(0, min(T-1, i + offset))
                indices[i] = new_idx
        
        # 应用抖动
        x_jittered = x[:, indices]
        
        return x_jittered
    
    def frame_dropout(self, x):
        """
        帧丢弃 - 随机丢弃某些帧
        
        Args:
            x: [B, T, ...] - 时序数据
        """
        if random.random() > self.frame_drop_prob:
            return x
        
        B, T = x.shape[:2]
        
        # 选择要保留的帧（至少保留80%）
        num_keep = int(T * 0.8)
        keep_indices = sorted(random.sample(range(T), num_keep))
        
        # 保留选中的帧
        x_dropped = x[:, keep_indices]
        
        # 插值回原始长度
        x_dropped = F.interpolate(
            x_dropped.permute(0, 2, 1),  # [B, D, T']
            size=T,
            mode='linear',
            align_corners=False
        ).permute(0, 2, 1)  # [B, T, D]
        
        return x_dropped
    
    def forward(self, x):
        """
        Args:
            x: [B, T, ...] - 时序数据（视频或音频）
        """
        # 时序抖动
        x = self.temporal_jitter(x)
        
        # 帧丢弃
        x = self.frame_dropout(x)
        
        return x


class AdaptiveAugmentation(nn.Module):
    """
    自适应增强 - 根据训练阶段动态调整增强强度
    
    策略：
    1. 早期训练：强增强（提升泛化）
    2. 后期训练：弱增强（保持稳定）
    
    Args:
        initial_strength: 初始增强强度
        final_strength: 最终增强强度
        warmup_epochs: 预热epoch数
    """
    def __init__(
        self,
        spec_augment,
        mixup,
        cutmix,
        temporal_augment,
        initial_strength=1.0,
        final_strength=0.3,
        total_epochs=100
    ):
        super().__init__()
        self.spec_augment = spec_augment
        self.mixup = mixup
        self.cutmix = cutmix
        self.temporal_augment = temporal_augment
        
        self.initial_strength = initial_strength
        self.final_strength = final_strength
        self.total_epochs = total_epochs
        
        self.current_epoch = 0
        self.current_strength = initial_strength
        
    def update_strength(self, epoch):
        """更新增强强度"""
        self.current_epoch = epoch
        progress = epoch / self.total_epochs
        
        # 线性衰减
        self.current_strength = (
            self.initial_strength - 
            (self.initial_strength - self.final_strength) * progress
        )
        
    def forward(self, video, audio, labels, mode='train'):
        """
        Args:
            video: [B, T, C, H, W]
            audio: [B, T, F]
            labels: [B] or [B, C]
            mode: 'train' or 'val'
        """
        if mode != 'train':
            return video, audio, labels
        
        # 根据当前强度决定是否应用增强
        apply_prob = self.current_strength
        
        # 1. SpecAugment（音频）
        if random.random() < apply_prob:
            audio = self.spec_augment(audio)
        
        # 2. 时序一致性增强
        if random.random() < apply_prob * 0.5:
            video = self.temporal_augment(video)
            audio = self.temporal_augment(audio)
        
        # 3. MixUp 或 CutMix（二选一）
        if random.random() < apply_prob * 0.5:
            # 转换标签为one-hot
            if labels.dim() == 1:
                num_classes = 2  # 根据您的任务调整
                labels_onehot = F.one_hot(labels, num_classes).float()
            else:
                labels_onehot = labels
            
            if random.random() < 0.5:
                # MixUp
                video, labels_mixed, _ = self.mixup(video, labels_onehot)
                audio, _, _ = self.mixup(audio, labels_onehot)
                labels = labels_mixed
            else:
                # CutMix（仅视频）
                video, labels, _ = self.cutmix(video, labels_onehot)
        
        return video, audio, labels


class IndustrialAugmentationPipeline(nn.Module):
    """
    完整的工业级增强流水线
    
    整合所有增强方法，提供统一接口
    """
    def __init__(
        self,
        num_classes=2,
        total_epochs=100,
        # SpecAugment参数
        freq_mask_param=15,
        time_mask_param=20,
        # MixUp/CutMix参数
        mixup_alpha=0.3,
        cutmix_alpha=1.0,
        # 时序增强参数
        temporal_jitter_prob=0.3,
        # 自适应参数
        initial_strength=1.0,
        final_strength=0.3
    ):
        super().__init__()
        
        # 创建各个增强模块
        self.spec_augment = SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param
        )
        
        self.mixup = MixUp(alpha=mixup_alpha)
        self.cutmix = CutMix(alpha=cutmix_alpha)
        
        self.temporal_augment = TemporalConsistencyAugment(
            temporal_jitter_prob=temporal_jitter_prob
        )
        
        # 自适应增强包装器
        self.adaptive_aug = AdaptiveAugmentation(
            spec_augment=self.spec_augment,
            mixup=self.mixup,
            cutmix=self.cutmix,
            temporal_augment=self.temporal_augment,
            initial_strength=initial_strength,
            final_strength=final_strength,
            total_epochs=total_epochs
        )
        
        self.num_classes = num_classes
        
    def update_epoch(self, epoch):
        """更新当前epoch（用于自适应调整）"""
        self.adaptive_aug.update_strength(epoch)
        
    def forward(self, video, audio, labels, mode='train'):
        """
        统一接口
        
        Args:
            video: [B, T, C, H, W] or [B, T, D]
            audio: [B, T, F] or [B, T, D]
            labels: [B] or [B, C]
            mode: 'train' or 'val'
        """
        return self.adaptive_aug(video, audio, labels, mode)


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 创建增强流水线
    augmentation = IndustrialAugmentationPipeline(
        num_classes=2,
        total_epochs=100,
        freq_mask_param=15,
        time_mask_param=20,
        mixup_alpha=0.3,
        cutmix_alpha=1.0,
        initial_strength=1.0,
        final_strength=0.3
    )
    
    # 模拟数据
    batch_size = 8
    seq_len = 100
    video = torch.randn(batch_size, seq_len, 3, 64, 64)
    audio = torch.randn(batch_size, seq_len, 128)
    labels = torch.randint(0, 2, (batch_size,))
    
    # 训练模式增强
    for epoch in range(5):
        augmentation.update_epoch(epoch)
        
        video_aug, audio_aug, labels_aug = augmentation(
            video, audio, labels, mode='train'
        )
        
        print(f"\nEpoch {epoch}:")
        print(f"  Video shape: {video_aug.shape}")
        print(f"  Audio shape: {audio_aug.shape}")
        print(f"  Labels shape: {labels_aug.shape}")
        print(f"  Current strength: {augmentation.adaptive_aug.current_strength:.2f}")
    
    # 验证模式（无增强）
    video_val, audio_val, labels_val = augmentation(
        video, audio, labels, mode='val'
    )
    print(f"\nValidation (no augmentation):")
    print(f"  Video unchanged: {torch.allclose(video, video_val)}")
    print(f"  Audio unchanged: {torch.allclose(audio, audio_val)}")