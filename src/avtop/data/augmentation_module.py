#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态数据增强模块
支持音频和视频的强弱增强，用于半监督学习
"""
import torch
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional


# ============================================================================
# 音频增强
# ============================================================================
class AudioAugmentation:
    """音频增强集合"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
    
    def time_stretch(self, audio: torch.Tensor, rate: float = 1.0) -> torch.Tensor:
        """
        时间拉伸（不改变音高）
        
        Args:
            audio: [T] 音频波形
            rate: 拉伸率 (>1加速, <1减速)
        """
        if rate == 1.0:
            return audio
        
        # 简单的线性插值实现
        original_length = audio.shape[0]
        new_length = int(original_length / rate)
        
        # 使用插值
        audio_2d = audio.unsqueeze(0).unsqueeze(0)  # [1, 1, T]
        stretched = F.interpolate(
            audio_2d, 
            size=new_length, 
            mode='linear', 
            align_corners=False
        )
        return stretched.squeeze(0).squeeze(0)
    
    def pitch_shift(self, audio: torch.Tensor, n_steps: int = 0) -> torch.Tensor:
        """
        音高变换（改变音高但不改变时长）
        
        Args:
            audio: [T] 音频波形
            n_steps: 半音步数 (正数升高，负数降低)
        """
        if n_steps == 0:
            return audio
        
        # 音高变换 = 时间拉伸 + 重采样
        rate = 2 ** (n_steps / 12.0)
        stretched = self.time_stretch(audio, rate)
        
        # 重采样回原始长度
        original_length = audio.shape[0]
        audio_2d = stretched.unsqueeze(0).unsqueeze(0)
        resampled = F.interpolate(
            audio_2d,
            size=original_length,
            mode='linear',
            align_corners=False
        )
        return resampled.squeeze(0).squeeze(0)
    
    def add_noise(self, audio: torch.Tensor, noise_factor: float = 0.005) -> torch.Tensor:
        """
        添加高斯白噪声
        
        Args:
            audio: [T] 音频波形
            noise_factor: 噪声强度
        """
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise
    
    def time_mask(self, audio: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """
        时间掩码（类似SpecAugment）
        
        Args:
            audio: [T] 音频波形
            mask_ratio: 掩码比例
        """
        T = audio.shape[0]
        mask_length = int(T * mask_ratio)
        
        if mask_length > 0:
            start = random.randint(0, T - mask_length)
            audio = audio.clone()
            audio[start:start + mask_length] = 0
        
        return audio
    
    def random_gain(self, audio: torch.Tensor, gain_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """
        随机增益调整
        
        Args:
            audio: [T] 音频波形
            gain_range: 增益范围
        """
        gain = random.uniform(*gain_range)
        return audio * gain


class WeakAudioAugment:
    """弱音频增强（用于半监督学习的一致性）"""
    
    def __init__(self, sr: int = 16000):
        self.aug = AudioAugmentation(sr)
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """随机应用1-2种轻度增强"""
        augmentations = [
            lambda x: self.aug.add_noise(x, noise_factor=0.002),
            lambda x: self.aug.random_gain(x, gain_range=(0.9, 1.1)),
        ]
        
        # 随机选择1-2种
        n_aug = random.randint(1, 2)
        selected = random.sample(augmentations, n_aug)
        
        for aug_fn in selected:
            audio = aug_fn(audio)
        
        return audio


class StrongAudioAugment:
    """强音频增强（用于半监督学习）"""
    
    def __init__(self, sr: int = 16000):
        self.aug = AudioAugmentation(sr)
    
    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """随机应用2-4种强增强"""
        augmentations = [
            lambda x: self.aug.time_stretch(x, rate=random.uniform(0.9, 1.1)),
            lambda x: self.aug.pitch_shift(x, n_steps=random.randint(-2, 2)),
            lambda x: self.aug.add_noise(x, noise_factor=random.uniform(0.005, 0.01)),
            lambda x: self.aug.time_mask(x, mask_ratio=random.uniform(0.05, 0.15)),
            lambda x: self.aug.random_gain(x, gain_range=(0.7, 1.3)),
        ]
        
        # 随机选择2-4种
        n_aug = random.randint(2, 4)
        selected = random.sample(augmentations, n_aug)
        
        for aug_fn in selected:
            audio = aug_fn(audio)
        
        return audio


# ============================================================================
# 视频增强
# ============================================================================
class VideoAugmentation:
    """视频增强集合"""
    
    def random_crop(self, video: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
        """
        随机裁剪
        
        Args:
            video: [T, C, H, W]
            crop_size: (crop_h, crop_w)
        """
        T, C, H, W = video.shape
        crop_h, crop_w = crop_size
        
        if H == crop_h and W == crop_w:
            return video
        
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)
        
        return video[:, :, top:top+crop_h, left:left+crop_w]
    
    def random_horizontal_flip(self, video: torch.Tensor, p: float = 0.5) -> torch.Tensor:
        """
        随机水平翻转
        
        Args:
            video: [T, C, H, W]
            p: 翻转概率
        """
        if random.random() < p:
            return video.flip(-1)  # 翻转W维度
        return video
    
    def color_jitter(self, 
                     video: torch.Tensor,
                     brightness: float = 0.2,
                     contrast: float = 0.2,
                     saturation: float = 0.2,
                     hue: float = 0.1) -> torch.Tensor:
        """
        颜色抖动
        
        Args:
            video: [T, C, H, W]
            brightness: 亮度变化范围
            contrast: 对比度变化范围
            saturation: 饱和度变化范围
            hue: 色调变化范围
        """
        # 亮度
        if brightness > 0:
            brightness_factor = random.uniform(1 - brightness, 1 + brightness)
            video = video * brightness_factor
        
        # 对比度
        if contrast > 0:
            contrast_factor = random.uniform(1 - contrast, 1 + contrast)
            mean = video.mean(dim=(-2, -1), keepdim=True)
            video = (video - mean) * contrast_factor + mean
        
        # 饱和度（简化版，只对RGB通道）
        if saturation > 0 and video.shape[1] == 3:
            saturation_factor = random.uniform(1 - saturation, 1 + saturation)
            gray = video.mean(dim=1, keepdim=True)  # [T, 1, H, W]
            video = gray + (video - gray) * saturation_factor
        
        # 裁剪到[0, 1]
        video = torch.clamp(video, 0, 1)
        
        return video
    
    def temporal_crop(self, video: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        时序裁剪
        
        Args:
            video: [T, C, H, W]
            num_frames: 目标帧数
        """
        T = video.shape[0]
        
        if T == num_frames:
            return video
        elif T < num_frames:
            # 如果不够，重复最后一帧
            padding = num_frames - T
            last_frame = video[-1:].repeat(padding, 1, 1, 1)
            return torch.cat([video, last_frame], dim=0)
        else:
            # 随机起始位置
            start = random.randint(0, T - num_frames)
            return video[start:start + num_frames]
    
    def temporal_sampling(self, video: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        时序采样（均匀或随机）
        
        Args:
            video: [T, C, H, W]
            num_frames: 目标帧数
        """
        T = video.shape[0]
        
        if T == num_frames:
            return video
        
        # 均匀采样 + 小抖动
        indices = np.linspace(0, T - 1, num_frames)
        # 添加随机抖动
        jitter = np.random.uniform(-0.5, 0.5, size=num_frames)
        indices = np.clip(indices + jitter, 0, T - 1).astype(int)
        
        return video[indices]
    
    def gaussian_blur(self, video: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
        """
        高斯模糊
        
        Args:
            video: [T, C, H, W]
            kernel_size: 核大小
        """
        # 简化实现：使用平均池化近似
        padding = kernel_size // 2
        blurred = F.avg_pool2d(
            video.flatten(0, 1),  # [T*C, H, W]
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        )
        return blurred.view_as(video)


class WeakVideoAugment:
    """弱视频增强"""
    
    def __init__(self, crop_size: Tuple[int, int] = (224, 224), num_frames: int = 16):
        self.aug = VideoAugmentation()
        self.crop_size = crop_size
        self.num_frames = num_frames
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """应用轻度增强"""
        # 时序采样
        video = self.aug.temporal_sampling(video, self.num_frames)
        
        # 随机水平翻转
        video = self.aug.random_horizontal_flip(video, p=0.5)
        
        # 轻微颜色抖动
        video = self.aug.color_jitter(
            video,
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05
        )
        
        return video


class StrongVideoAugment:
    """强视频增强"""
    
    def __init__(self, crop_size: Tuple[int, int] = (224, 224), num_frames: int = 16):
        self.aug = VideoAugmentation()
        self.crop_size = crop_size
        self.num_frames = num_frames
    
    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        """应用强增强"""
        # 时序裁剪（随机起点）
        video = self.aug.temporal_crop(video, self.num_frames)
        
        # 随机裁剪
        if video.shape[-2:] != self.crop_size:
            video = self.aug.random_crop(video, self.crop_size)
        
        # 随机水平翻转
        video = self.aug.random_horizontal_flip(video, p=0.5)
        
        # 强颜色抖动
        video = self.aug.color_jitter(
            video,
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
        
        # 可能添加模糊
        if random.random() < 0.3:
            video = self.aug.gaussian_blur(video, kernel_size=5)
        
        return video


# ============================================================================
# 组合增强器
# ============================================================================
class MultiModalAugmentation:
    """多模态数据增强（音频+视频）"""
    
    def __init__(self,
                 mode: str = 'weak',
                 audio_sr: int = 16000,
                 video_size: Tuple[int, int] = (224, 224),
                 num_frames: int = 16):
        """
        Args:
            mode: 'weak' 或 'strong'
            audio_sr: 音频采样率
            video_size: 视频尺寸
            num_frames: 视频帧数
        """
        self.mode = mode
        
        if mode == 'weak':
            self.audio_aug = WeakAudioAugment(audio_sr)
            self.video_aug = WeakVideoAugment(video_size, num_frames)
        elif mode == 'strong':
            self.audio_aug = StrongAudioAugment(audio_sr)
            self.video_aug = StrongVideoAugment(video_size, num_frames)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def __call__(self, audio: torch.Tensor, video: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        增强音频和视频
        
        Args:
            audio: [T_audio] 音频波形
            video: [T_video, C, H, W] 视频帧
        
        Returns:
            增强后的 (audio, video)
        """
        audio_aug = self.audio_aug(audio)
        video_aug = self.video_aug(video)
        
        return audio_aug, video_aug


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("数据增强模块测试")
    print("="*70)
    
    # 创建模拟数据
    audio = torch.randn(16000)  # 1秒音频
    video = torch.rand(32, 3, 256, 256)  # 32帧视频
    
    print(f"\n原始数据:")
    print(f"  Audio: {audio.shape}, mean={audio.mean():.3f}, std={audio.std():.3f}")
    print(f"  Video: {video.shape}, mean={video.mean():.3f}, std={video.std():.3f}")
    
    # 测试弱增强
    print(f"\n测试弱增强:")
    weak_aug = MultiModalAugmentation(mode='weak')
    audio_weak, video_weak = weak_aug(audio, video)
    print(f"  Audio: {audio_weak.shape}, mean={audio_weak.mean():.3f}, std={audio_weak.std():.3f}")
    print(f"  Video: {video_weak.shape}, mean={video_weak.mean():.3f}, std={video_weak.std():.3f}")
    
    # 测试强增强
    print(f"\n测试强增强:")
    strong_aug = MultiModalAugmentation(mode='strong')
    audio_strong, video_strong = strong_aug(audio, video)
    print(f"  Audio: {audio_strong.shape}, mean={audio_strong.mean():.3f}, std={audio_strong.std():.3f}")
    print(f"  Video: {video_strong.shape}, mean={video_strong.mean():.3f}, std={video_strong.std():.3f}")
    
    # 测试多次增强的变化
    print(f"\n测试一致性（同一输入的多次增强应不同）:")
    audio1, video1 = strong_aug(audio, video)
    audio2, video2 = strong_aug(audio, video)
    
    audio_diff = (audio1 - audio2).abs().mean()
    video_diff = (video1 - video2).abs().mean()
    
    print(f"  Audio差异: {audio_diff:.6f}")
    print(f"  Video差异: {video_diff:.6f}")
    
    if audio_diff > 0.001 and video_diff > 0.001:
        print(f"  ✅ 增强正确：每次结果不同")
    else:
        print(f"  ⚠️  警告：增强可能未生效")
    
    print(f"\n✅ 数据增强模块测试完成!")
