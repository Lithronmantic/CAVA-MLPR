#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
window_dataset.py
窗口级别的音视频数据集（用于训练）

特性：
1. 读取generate_training_dataset.py生成的CSV
2. 按照窗口范围加载音视频片段
3. 支持is_labeled字段（半监督学习）
4. 缓存机制（可选）
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import soundfile as sf
import cv2
from pathlib import Path
from typing import Optional, Tuple, Callable
import warnings


class WindowDataset(Dataset):
    """
    窗口级音视频数据集

    每个样本是一对对齐的音视频窗口：
    - 音频窗口: [audio_start_s, audio_end_s]
    - 视频窗口: [video_start_frame, video_end_frame]
    """

    def __init__(
            self,
            csv_path: str,
            audio_transform: Optional[Callable] = None,
            video_transform: Optional[Callable] = None,
            target_sr: int = 16000,
            target_video_size: Tuple[int, int] = (224, 224),
            max_audio_length: float = 0.3,  # 最大音频长度（秒）
            max_video_frames: int = 64,  # 最大视频帧数
            cache_mode: str = 'none'  # 'none', 'memory', 'disk'
    ):
        """
        Args:
            csv_path: CSV文件路径（train.csv / val.csv / unlabeled.csv）
            audio_transform: 音频增强函数
            video_transform: 视频增强函数
            target_sr: 目标采样率
            target_video_size: 目标视频尺寸
            max_audio_length: 最大音频长度（超过则截断）
            max_video_frames: 最大视频帧数（超过则采样）
            cache_mode: 缓存模式
        """
        self.csv_path = Path(csv_path)
        self.audio_transform = audio_transform
        self.video_transform = video_transform

        self.target_sr = target_sr
        self.target_video_size = target_video_size
        self.max_audio_length = max_audio_length
        self.max_video_frames = max_video_frames
        self.cache_mode = cache_mode

        # 加载CSV
        self.data = pd.read_csv(csv_path)

        # 确保必需字段存在
        required_fields = [
            'video_path', 'audio_path', 'label', 'is_labeled',
            'video_start_frame', 'video_end_frame',
            'audio_start_s', 'audio_end_s'
        ]

        missing = [f for f in required_fields if f not in self.data.columns]
        if missing:
            raise ValueError(f"CSV缺少字段: {missing}")

        # 过滤无效记录
        self.data = self.data.dropna(subset=['video_path', 'audio_path'])
        self.data = self.data.reset_index(drop=True)

        # 缓存（如果启用）
        self.cache = {} if cache_mode == 'memory' else None

        print(f"✅ Dataset加载完成: {len(self.data)} 窗口对")
        print(f"   - 有标签: {self.data['is_labeled'].sum()}")
        print(f"   - 无标签: {(~self.data['is_labeled'].astype(bool)).sum()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回：
        {
            'audio': Tensor [C, T] - 音频波形
            'video': Tensor [T, C, H, W] - 视频帧序列
            'label': int - 标签 (0/1/-1)
            'is_labeled': bool - 是否有标签
            'metadata': dict - 额外信息
        }
        """
        # 检查缓存
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]

        row = self.data.iloc[idx]

        # 1. 加载音频窗口
        audio = self._load_audio_window(
            audio_path=row['audio_path'],
            start_s=float(row['audio_start_s']),
            end_s=float(row['audio_end_s'])
        )

        # 2. 加载视频窗口
        video = self._load_video_window(
            video_path=row['video_path'],
            start_frame=int(row['video_start_frame']),
            end_frame=int(row['video_end_frame'])
        )

        # 3. 数据增强
        if self.audio_transform is not None:
            audio = self.audio_transform(audio)

        if self.video_transform is not None:
            video = self.video_transform(video)

        # 4. 构造样本
        sample = {
            'audio': audio,
            'video': video,
            'label': int(row['label']),
            'is_labeled': bool(row['is_labeled']),
            'metadata': {
                'sample': row.get('sample', ''),
                'window_score': float(row.get('window_score', 0.0)),
                'sample_quality': float(row.get('sample_quality', 0.0))
            }
        }

        # 缓存
        if self.cache is not None:
            self.cache[idx] = sample

        return sample

    def _load_audio_window(self, audio_path: str, start_s: float, end_s: float) -> torch.Tensor:
        """
        加载音频窗口 [start_s, end_s]

        Returns:
            audio: [1, num_samples] - 单声道音频
        """
        try:
            # 读取完整音频
            y, sr = sf.read(audio_path, dtype='float32', always_2d=False)

            # 转单声道
            if y.ndim == 2:
                y = y.mean(axis=1)

            # 重采样（如果需要）
            if sr != self.target_sr:
                duration = len(y) / sr
                num_samples_new = int(duration * self.target_sr)
                t_old = np.linspace(0, duration, len(y), endpoint=False)
                t_new = np.linspace(0, duration, num_samples_new, endpoint=False)
                y = np.interp(t_new, t_old, y).astype(np.float32)
                sr = self.target_sr

            # 提取窗口
            start_sample = int(start_s * sr)
            end_sample = int(end_s * sr)

            start_sample = max(0, min(start_sample, len(y)))
            end_sample = max(start_sample, min(end_sample, len(y)))

            window = y[start_sample:end_sample]

            # 限制最大长度
            max_samples = int(self.max_audio_length * sr)
            if len(window) > max_samples:
                window = window[:max_samples]

            # 填充到固定长度（可选）
            # if len(window) < max_samples:
            #     window = np.pad(window, (0, max_samples - len(window)))

            # 归一化
            max_val = np.abs(window).max()
            if max_val > 0:
                window = window / max_val

            # 转为Tensor [1, T]
            audio_tensor = torch.from_numpy(window).float().unsqueeze(0)

            return audio_tensor

        except Exception as e:
            warnings.warn(f"音频加载失败 {audio_path}: {e}")
            # 返回静音
            return torch.zeros(1, int(self.max_audio_length * self.target_sr))

    def _load_video_window(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """
        加载视频窗口 [start_frame, end_frame]

        Returns:
            video: [T, 3, H, W] - RGB帧序列
        """
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {video_path}")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 确保范围有效
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame, min(end_frame, total_frames - 1))

            # 跳到起始帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            frames = []
            for _ in range(end_frame - start_frame + 1):
                ret, frame = cap.read()
                if not ret:
                    break

                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 调整大小
                frame = cv2.resize(frame, self.target_video_size)

                frames.append(frame)

            cap.release()

            if len(frames) == 0:
                raise RuntimeError(f"无法读取帧: {video_path}")

            # 帧采样（如果超过最大数量）
            if len(frames) > self.max_video_frames:
                indices = np.linspace(0, len(frames) - 1, self.max_video_frames, dtype=int)
                frames = [frames[i] for i in indices]

            # 转为Tensor [T, H, W, C] -> [T, C, H, W]
            frames = np.stack(frames, axis=0)  # [T, H, W, 3]
            frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2)  # [T, 3, H, W]

            # 归一化到[0, 1]
            frames = frames / 255.0

            return frames

        except Exception as e:
            warnings.warn(f"视频加载失败 {video_path}: {e}")
            # 返回黑屏
            return torch.zeros(
                self.max_video_frames, 3,
                self.target_video_size[0],
                self.target_video_size[1]
            )

    def get_label_distribution(self):
        """获取标签分布（仅有标签样本）"""
        labeled = self.data[self.data['is_labeled'] == 1]
        if len(labeled) == 0:
            return {}
        return labeled['label'].value_counts().to_dict()

    def get_quality_stats(self):
        """获取质量统计"""
        if 'sample_quality' not in self.data.columns:
            return None

        return {
            'mean': self.data['sample_quality'].mean(),
            'median': self.data['sample_quality'].median(),
            'min': self.data['sample_quality'].min(),
            'max': self.data['sample_quality'].max()
        }


def collate_fn(batch):
    """
    自定义collate函数（处理变长序列）

    策略：
    - 音频：填充到batch内最大长度
    - 视频：填充到batch内最大帧数

    🔧 修复：将音频从 [B, 1, T] squeeze 为 [B, T]
    """
    # 分离各个字段
    audios = [item['audio'] for item in batch]
    videos = [item['video'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    is_labeled = torch.tensor([item['is_labeled'] for item in batch], dtype=torch.bool)

    # 填充音频
    max_audio_len = max(a.shape[1] for a in audios)
    audios_padded = []
    for audio in audios:
        if audio.shape[1] < max_audio_len:
            padding = torch.zeros(1, max_audio_len - audio.shape[1])
            audio = torch.cat([audio, padding], dim=1)
        audios_padded.append(audio)
    audios = torch.stack(audios_padded, dim=0)  # [B, 1, T]

    # === 🔧 关键修复：去除通道维度 ===
    # 从 [B, 1, T] -> [B, T]
    audios = audios.squeeze(1)
    # ================================

    # 填充视频
    max_video_frames = max(v.shape[0] for v in videos)
    videos_padded = []
    for video in videos:
        if video.shape[0] < max_video_frames:
            padding = torch.zeros(
                max_video_frames - video.shape[0],
                video.shape[1], video.shape[2], video.shape[3]
            )
            video = torch.cat([video, padding], dim=0)
        videos_padded.append(video)
    videos = torch.stack(videos_padded, dim=0)  # [B, T, C, H, W]

    return {
        'audio': audios,  # 现在是 [B, T] ✅
        'video': videos,  # [B, T, C, H, W]
        'label': labels,  # [B]
        'is_labeled': is_labeled  # [B]
    }


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # 创建数据集
    train_dataset = WindowDataset(
        csv_path="data/train.csv",
        target_sr=16000,
        target_video_size=(224, 224),
        max_audio_length=0.3,
        max_video_frames=64
    )

    val_dataset = WindowDataset(
        csv_path="data/val.csv",
        target_sr=16000,
        target_video_size=(224, 224)
    )

    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 查看样本
    sample = train_dataset[0]
    print(f"\n样本形状:")
    print(f"  音频: {sample['audio'].shape}")
    print(f"  视频: {sample['video'].shape}")
    print(f"  标签: {sample['label']}")
    print(f"  有标签: {sample['is_labeled']}")

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # 测试一个batch
    batch = next(iter(train_loader))
    print(f"\nBatch形状:")
    print(f"  音频: {batch['audio'].shape}")  # 应该是 [4, T] ✅
    print(f"  视频: {batch['video'].shape}")  # [4, T, 3, 224, 224]
    print(f"  标签: {batch['label'].shape}")  # [4]
    print(f"  有标签: {batch['is_labeled'].shape}")  # [4]

    # === 额外验证：确保音频形状正确 ===
    assert batch['audio'].dim() == 2, f"音频应该是2维 [B, T]，实际是 {batch['audio'].dim()}维"
    print(f"\n✅ 音频形状验证通过: {batch['audio'].shape} (期望 [B, T])")

    print("\n✅ Dataset测试通过！")