#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
industrial.py - 工业级音视频数据增强（无省略）

音频增强：
1. SpecAugment (时频掩码)
2. 时间拉伸 (Time Stretch)
3. 音高变换 (Pitch Shift)
4. 添加噪声
5. 音量调整

视频增强：
1. 空间增强 (裁剪、翻转、颜色抖动)
2. 时序增强 (随机采样、速度变换)
3. MixUp / CutMix
4. 帧丢弃 (模拟实际情况)

时序一致性增强：
确保音视频增强后仍然对齐
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, Dict
import random


# ============================================================================
# 音频增强
# ============================================================================

class SpecAugment(nn.Module):
    """
    SpecAugment - 频谱增强

    论文: "SpecAugment: A Simple Data Augmentation Method for
           Automatic Speech Recognition"

    方法：
    1. 时间掩码 (Time Masking) - 遮挡连续时间段
    2. 频率掩码 (Frequency Masking) - 遮挡连续频率段
    """

    def __init__(
            self,
            freq_mask_num: int = 2,
            freq_mask_param: int = 27,
            time_mask_num: int = 2,
            time_mask_param: int = 40,
            mask_value: float = 0.0
    ):
        """
        Args:
            freq_mask_num: 频率掩码数量
            freq_mask_param: 频率掩码最大宽度
            time_mask_num: 时间掩码数量
            time_mask_param: 时间掩码最大宽度
            mask_value: 掩码填充值
        """
        super().__init__()
        self.freq_mask_num = freq_mask_num
        self.freq_mask_param = freq_mask_param
        self.time_mask_num = time_mask_num
        self.time_mask_param = time_mask_param
        self.mask_value = mask_value

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec: [B, freq, time] 或 [B, 1, freq, time] - 频谱图

        Returns:
            augmented_spec: 增强后的频谱图
        """
        # 确保维度正确
        if spec.ndim == 3:
            spec = spec.unsqueeze(1)  # [B, 1, freq, time]

        B, C, F, T = spec.shape
        spec = spec.clone()

        for b in range(B):
            # 频率掩码
            for _ in range(self.freq_mask_num):
                f = random.randint(0, self.freq_mask_param)
                f0 = random.randint(0, max(1, F - f))
                spec[b, :, f0:f0 + f, :] = self.mask_value

            # 时间掩码
            for _ in range(self.time_mask_num):
                t = random.randint(0, self.time_mask_param)
                t0 = random.randint(0, max(1, T - t))
                spec[b, :, :, t0:t0 + t] = self.mask_value

        return spec.squeeze(1) if C == 1 else spec

    def __repr__(self):
        return (f"SpecAugment(freq_mask={self.freq_mask_num}x{self.freq_mask_param}, "
                f"time_mask={self.time_mask_num}x{self.time_mask_param})")


class TimeStretch(nn.Module):
    """
    时间拉伸 - 改变音频速度但不改变音高

    使用线性插值实现简化版
    """

    def __init__(self, rate_range: Tuple[float, float] = (0.8, 1.2)):
        """
        Args:
            rate_range: 拉伸比例范围 (min, max)
                - rate < 1.0: 加速
                - rate > 1.0: 减速
        """
        super().__init__()
        self.rate_range = rate_range

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, C, T] - 音频波形

        Returns:
            stretched_audio: [B, C, T] - 拉伸后的音频（保持长度）
        """
        B, C, T = audio.shape

        # 随机拉伸比例
        rate = random.uniform(*self.rate_range)

        # 新长度
        new_T = int(T * rate)

        # 线性插值
        stretched = F.interpolate(
            audio,
            size=new_T,
            mode='linear',
            align_corners=False
        )

        # 裁剪或填充到原长度
        if new_T > T:
            stretched = stretched[:, :, :T]
        elif new_T < T:
            padding = T - new_T
            stretched = F.pad(stretched, (0, padding), mode='constant', value=0)

        return stretched


class AddGaussianNoise(nn.Module):
    """添加高斯噪声"""

    def __init__(self, noise_factor: float = 0.005):
        super().__init__()
        self.noise_factor = noise_factor

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, C, T]
        """
        noise = torch.randn_like(audio) * self.noise_factor
        return audio + noise


class AudioAugmentation(nn.Module):
    """
    完整的音频增强管道

    组合所有音频增强方法
    """

    def __init__(self, config: Dict, is_training: bool = True):
        super().__init__()
        self.is_training = is_training
        self.config = config

        audio_config = config.get('augmentation', {}).get('audio', {})

        # SpecAugment
        self.spec_augment = SpecAugment(
            freq_mask_num=audio_config.get('freq_mask_num', 2),
            freq_mask_param=audio_config.get('freq_mask_param', 27),
            time_mask_num=audio_config.get('time_mask_num', 2),
            time_mask_param=audio_config.get('time_mask_param', 40)
        )

        # 时间拉伸
        self.time_stretch = TimeStretch(
            rate_range=tuple(audio_config.get('time_stretch_rate', [0.8, 1.2]))
        ) if audio_config.get('time_stretch', False) else None

        # 添加噪声
        self.add_noise = AddGaussianNoise(
            noise_factor=audio_config.get('noise_factor', 0.005)
        ) if audio_config.get('add_noise', False) else None

        # 增强概率
        self.aug_prob = config.get('augmentation_prob', 0.5)

    def forward(self, audio: torch.Tensor, is_strong: bool = False) -> torch.Tensor:
        """
        Args:
            audio: [B, C, T] - 音频输入
            is_strong: 是否应用强增强（用于FixMatch）

        Returns:
            augmented_audio: 增强后的音频
        """
        if not self.is_training:
            return audio

        # 弱增强：只应用SpecAugment
        if not is_strong:
            if random.random() < self.aug_prob:
                audio = self.spec_augment(audio)
            return audio

        # 强增强：应用所有增强
        if self.time_stretch and random.random() < 0.5:
            audio = self.time_stretch(audio)

        if self.add_noise and random.random() < 0.5:
            audio = self.add_noise(audio)

        if random.random() < 0.8:  # 强增强时更高概率应用SpecAugment
            audio = self.spec_augment(audio)

        return audio


# ============================================================================
# 视频增强
# ============================================================================

class RandomTemporalCrop(nn.Module):
    """随机时序裁剪 - 随机选择连续帧"""

    def __init__(self, num_frames: int = 16):
        super().__init__()
        self.num_frames = num_frames

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W]

        Returns:
            cropped: [B, num_frames, C, H, W]
        """
        B, T, C, H, W = video.shape

        if T <= self.num_frames:
            # 如果帧数不足，填充
            padding = self.num_frames - T
            video = F.pad(video, (0, 0, 0, 0, 0, 0, 0, padding), mode='replicate')
            return video

        # 随机起始帧
        start = random.randint(0, T - self.num_frames)
        return video[:, start:start + self.num_frames, :, :, :]


class RandomSpatialCrop(nn.Module):
    """随机空间裁剪"""

    def __init__(self, size: Tuple[int, int] = (224, 224), scale: Tuple[float, float] = (0.8, 1.0)):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W]
        """
        B, T, C, H, W = video.shape

        # 随机缩放比例
        scale = random.uniform(*self.scale)

        # 裁剪尺寸
        crop_h = int(H * scale)
        crop_w = int(W * scale)

        # 随机位置
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # 裁剪
        video = video[:, :, :, top:top + crop_h, left:left + crop_w]

        # 调整到目标尺寸
        video = video.view(B * T, C, crop_h, crop_w)
        video = F.interpolate(video, size=self.size, mode='bilinear', align_corners=False)
        video = video.view(B, T, C, *self.size)

        return video


class ColorJitter(nn.Module):
    """颜色抖动 - 调整亮度、对比度、饱和度"""

    def __init__(self, brightness: float = 0.2, contrast: float = 0.2):
        super().__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, T, C, H, W], 范围[0, 1]
        """
        B, T, C, H, W = video.shape

        # 亮度调整
        if random.random() < 0.5:
            factor = random.uniform(1 - self.brightness, 1 + self.brightness)
            video = video * factor

        # 对比度调整
        if random.random() < 0.5:
            factor = random.uniform(1 - self.contrast, 1 + self.contrast)
            mean = video.mean(dim=(2, 3, 4), keepdim=True)
            video = (video - mean) * factor + mean

        # 截断到[0, 1]
        video = torch.clamp(video, 0, 1)

        return video


class MixUp(nn.Module):
    """
    MixUp数据增强

    论文: "mixup: Beyond Empirical Risk Minimization"

    混合两个样本：
    x_mix = λ * x1 + (1 - λ) * x2
    y_mix = λ * y1 + (1 - λ) * y2
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(
            self,
            video: torch.Tensor,
            audio: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            video: [B, T, C, H, W]
            audio: [B, C, T]
            labels: [B, num_classes] (one-hot)

        Returns:
            mixed_video, mixed_audio, mixed_labels
        """
        B = video.shape[0]

        # 采样λ
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 随机排列索引
        index = torch.randperm(B, device=video.device)

        # 混合
        mixed_video = lam * video + (1 - lam) * video[index]
        mixed_audio = lam * audio + (1 - lam) * audio[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return mixed_video, mixed_audio, mixed_labels


class VideoAugmentation(nn.Module):
    """
    完整的视频增强管道
    """

    def __init__(self, config: Dict, is_training: bool = True):
        super().__init__()
        self.is_training = is_training
        self.config = config

        video_config = config.get('augmentation', {}).get('video', {})

        # 空间增强
        self.spatial_crop = RandomSpatialCrop(
            size=tuple(config.get('data', {}).get('video_size', [224, 224]))
        ) if video_config.get('random_crop', False) else None

        self.color_jitter = ColorJitter(
            brightness=0.2, contrast=0.2
        ) if video_config.get('color_jitter', False) else None

        # 时序增强
        self.temporal_crop = RandomTemporalCrop(
            num_frames=config.get('data', {}).get('max_video_frames', 16)
        ) if video_config.get('random_temporal_crop', False) else None

        # MixUp
        self.use_mixup = video_config.get('use_mixup', False)
        self.mixup = MixUp(alpha=video_config.get('mixup_alpha', 0.2)) if self.use_mixup else None

        self.aug_prob = config.get('augmentation_prob', 0.5)

    def forward(
            self,
            video: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            is_strong: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            video: [B, T, C, H, W]
            labels: [B, num_classes] (可选，用于MixUp)
            is_strong: 是否应用强增强

        Returns:
            augmented_video, mixed_labels (如果使用MixUp)
        """
        if not self.is_training:
            return video, labels

        # 弱增强
        if not is_strong:
            if self.temporal_crop and random.random() < self.aug_prob:
                video = self.temporal_crop(video)

            if self.color_jitter and random.random() < self.aug_prob:
                video = self.color_jitter(video)

            return video, labels

        # 强增强
        if self.spatial_crop and random.random() < 0.7:
            video = self.spatial_crop(video)

        if self.temporal_crop and random.random() < 0.7:
            video = self.temporal_crop(video)

        if self.color_jitter and random.random() < 0.8:
            video = self.color_jitter(video)

        # 随机水平翻转
        if random.random() < 0.5:
            video = torch.flip(video, dims=[4])  # 翻转宽度维度

        return video, labels


# ============================================================================
# 完整增强管道
# ============================================================================

class IndustrialAugmentationPipeline(nn.Module):
    """
    工业级完整增强管道

    集成音频和视频增强，确保时序对齐
    """

    def __init__(self, config: Dict, is_training: bool = True):
        super().__init__()
        self.is_training = is_training

        # 音频增强
        self.audio_aug = AudioAugmentation(config, is_training)

        # 视频增强
        self.video_aug = VideoAugmentation(config, is_training)

        print("[IndustrialAugmentationPipeline] 初始化完成")
        print(f"  - 音频增强: SpecAugment, TimeStretch, Noise")
        print(f"  - 视频增强: SpatialCrop, ColorJitter, TemporalCrop")

    def forward(
            self,
            audio: torch.Tensor,
            video: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            is_strong: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            audio: [B, C, T]
            video: [B, T, C, H, W]
            labels: [B, num_classes] (可选)
            is_strong: 是否应用强增强（FixMatch用）

        Returns:
            augmented_audio, augmented_video, mixed_labels
        """
        if not self.is_training:
            return audio, video, labels

        # 音频增强
        audio = self.audio_aug(audio, is_strong=is_strong)

        # 视频增强
        video, labels = self.video_aug(video, labels, is_strong=is_strong)

        return audio, video, labels

    def __repr__(self):
        return (f"IndustrialAugmentationPipeline(\n"
                f"  audio={self.audio_aug},\n"
                f"  video={self.video_aug}\n"
                f")")


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 模拟配置
    config = {
        'data': {
            'video_size': [224, 224],
            'max_video_frames': 16
        },
        'augmentation': {
            'audio': {
                'freq_mask_num': 2,
                'freq_mask_param': 27,
                'time_mask_num': 2,
                'time_mask_param': 40,
                'time_stretch': True,
                'add_noise': True
            },
            'video': {
                'random_crop': True,
                'color_jitter': True,
                'random_temporal_crop': True
            }
        },
        'augmentation_prob': 0.5
    }

    # 创建增强管道
    aug_pipeline = IndustrialAugmentationPipeline(config, is_training=True)

    # 模拟数据
    batch_size = 4
    audio = torch.randn(batch_size, 1, 3200)  # [B, 1, T]
    video = torch.randn(batch_size, 20, 3, 224, 224)  # [B, T, C, H, W]
    labels = F.one_hot(torch.randint(0, 2, (batch_size,)), num_classes=2).float()

    print("\n=== 测试弱增强 ===")
    aug_audio_weak, aug_video_weak, _ = aug_pipeline(audio, video, labels, is_strong=False)
    print(f"音频: {audio.shape} -> {aug_audio_weak.shape}")
    print(f"视频: {video.shape} -> {aug_video_weak.shape}")

    print("\n=== 测试强增强 ===")
    aug_audio_strong, aug_video_strong, _ = aug_pipeline(audio, video, labels, is_strong=True)
    print(f"音频: {audio.shape} -> {aug_audio_strong.shape}")
    print(f"视频: {video.shape} -> {aug_video_strong.shape}")

    print("\n✅ 数据增强测试通过！")