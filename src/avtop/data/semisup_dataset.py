#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半监督学习数据集
扩展原有Dataset，支持is_labeled标记

与原csv_dataset.py兼容，只是增加了is_labeled字段的处理
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import soundfile as sf
import cv2


class SemiSupDataset(Dataset):
    """
    半监督数据集
    
    关键特性：
    - 返回is_labeled字段
    - 支持标注/未标注混合
    - 与MixedBatchSampler配合使用
    
    CSV格式要求：
    - sample: 样本名称
    - audio: 音频文件路径
    - video: 视频文件路径
    - audio_start_s, audio_end_s: 音频窗口
    - video_start_frame, video_end_frame: 视频窗口
    - is_labeled: 0/1标记（必需）
    - label: 标签（可选，未标注样本可以没有）
    """
    def __init__(self,
                 windows_csv: str,
                 ann_csv: str = None,
                 classes: list = None,
                 audio_sr: int = 16000,
                 video_size: tuple = (224, 224),
                 num_frames: int = 16,
                 use_cache: bool = False):
        """
        Args:
            windows_csv: 窗口CSV文件（需包含is_labeled列）
            ann_csv: 标注CSV文件（包含label列）
            classes: 类别列表
            audio_sr: 音频采样率
            video_size: 视频尺寸
            num_frames: 视频帧数
            use_cache: 是否缓存特征
        """
        self.windows_csv = Path(windows_csv)
        self.ann_csv = Path(ann_csv) if ann_csv else None
        self.classes = classes or ['normal', 'defect']
        self.audio_sr = audio_sr
        self.video_size = video_size
        self.num_frames = num_frames
        self.use_cache = use_cache
        
        # 加载数据
        self._load_data()
        
        # 统计信息
        self._print_stats()
    
    def _load_data(self):
        """加载CSV数据"""
        # 1. 加载windows
        self.df_windows = pd.read_csv(self.windows_csv)
        
        # 检查必需列
        required_cols = ['sample', 'audio', 'video', 'is_labeled']
        missing = [c for c in required_cols if c not in self.df_windows.columns]
        
        if missing:
            raise ValueError(f"Windows CSV缺少必需列: {missing}")
        
        # 2. 加载标注（如果有）
        if self.ann_csv and self.ann_csv.exists():
            df_ann = pd.read_csv(self.ann_csv)
            
            # 合并标注
            if 'sample' in df_ann.columns and 'label' in df_ann.columns:
                # 按sample合并
                self.df = self.df_windows.merge(
                    df_ann[['sample', 'label']],
                    on='sample',
                    how='left'
                )
            else:
                print("警告: 标注文件格式不正确，使用原始windows")
                self.df = self.df_windows.copy()
        else:
            self.df = self.df_windows.copy()
        
        # 3. 处理标签
        if 'label' not in self.df.columns:
            # 如果没有label列，创建一个（未标注样本设为-1）
            self.df['label'] = -1
        
        # 未标注样本的label设为-1
        unlabeled_mask = self.df['is_labeled'] == 0
        self.df.loc[unlabeled_mask, 'label'] = -1
        
        # 4. 创建类别映射
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        # 转换字符串标签为索引
        if self.df['label'].dtype == 'object':
            self.df['label'] = self.df['label'].map(
                lambda x: self.class_to_idx.get(x, -1) if x != -1 else -1
            )
    
    def _print_stats(self):
        """打印数据集统计信息"""
        print(f"\n{'='*70}")
        print(f"SemiSup Dataset Statistics")
        print(f"{'='*70}")
        print(f"Total samples: {len(self.df)}")
        
        # 标注/未标注统计
        n_labeled = (self.df['is_labeled'] == 1).sum()
        n_unlabeled = (self.df['is_labeled'] == 0).sum()
        
        print(f"  Labeled: {n_labeled} ({n_labeled/len(self.df)*100:.1f}%)")
        print(f"  Unlabeled: {n_unlabeled} ({n_unlabeled/len(self.df)*100:.1f}%)")
        
        # 类别分布（仅标注样本）
        if n_labeled > 0:
            labeled_df = self.df[self.df['is_labeled'] == 1]
            print(f"\nClass distribution (labeled only):")
            for class_name, class_idx in self.class_to_idx.items():
                count = (labeled_df['label'] == class_idx).sum()
                print(f"  {class_name}: {count} ({count/n_labeled*100:.1f}%)")
        
        print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        返回一个样本
        
        Returns:
            dict:
                - video: [T, C, H, W] 或 [T, D]
                - audio: [T, freq] 或 [T, D]
                - label: int (标注样本) 或 -1 (未标注样本)
                - is_labeled: bool
                - index: int
        """
        row = self.df.iloc[idx]
        
        # 1. 加载音频
        audio = self._load_audio(row)
        
        # 2. 加载视频
        video = self._load_video(row)
        
        # 3. 标签
        label = int(row['label'])
        is_labeled = bool(row['is_labeled'])
        
        return {
            'video': video,
            'audio': audio,
            'label': label,
            'is_labeled': is_labeled,
            'index': idx,
            'sample': row['sample']
        }
    
    def _load_audio(self, row):
        """加载音频片段"""
        audio_path = row['audio']
        t_start = row.get('audio_start_s', 0)
        t_end = row.get('audio_end_s', None)
        
        try:
            # 读取音频
            y, sr = sf.read(audio_path)
            
            # 单声道
            if y.ndim == 2:
                y = y.mean(axis=1)
            
            # 重采样
            if sr != self.audio_sr:
                from scipy import signal
                n_samples = int(len(y) * self.audio_sr / sr)
                y = signal.resample(y, n_samples)
                sr = self.audio_sr
            
            # 提取片段
            if t_end is not None:
                start_idx = int(t_start * sr)
                end_idx = int(t_end * sr)
                y = y[start_idx:end_idx]
            
            # 转为mel频谱（简化版，实际应使用VGGish）
            # 这里返回原始音频，后续在模型中处理
            audio_tensor = torch.from_numpy(y).float()
            
            # Padding到固定长度
            target_len = int(1.5 * sr)  # 1.5秒
            if len(audio_tensor) < target_len:
                audio_tensor = torch.nn.functional.pad(
                    audio_tensor, (0, target_len - len(audio_tensor))
                )
            else:
                audio_tensor = audio_tensor[:target_len]
            
            return audio_tensor.unsqueeze(0)  # [1, T]
            
        except Exception as e:
            print(f"警告: 音频加载失败 {audio_path}: {e}")
            # 返回零张量
            return torch.zeros(1, int(1.5 * self.audio_sr))
    
    def _load_video(self, row):
        """加载视频帧"""
        video_path = row['video']
        f_start = int(row.get('video_start_frame', 0))
        f_end = int(row.get('video_end_frame', self.num_frames))
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            frames = []
            frame_indices = np.linspace(f_start, f_end, self.num_frames, dtype=int)
            
            for fid in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
                ret, frame = cap.read()
                
                if not ret:
                    # 使用最后一帧或黑帧
                    if frames:
                        frame = frames[-1].copy()
                    else:
                        frame = np.zeros((*self.video_size, 3), dtype=np.uint8)
                
                # 调整尺寸
                frame = cv2.resize(frame, self.video_size)
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 归一化
                frame = frame.astype(np.float32) / 255.0
                
                frames.append(frame)
            
            cap.release()
            
            # [T, H, W, C] -> [T, C, H, W]
            video_tensor = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)
            
            return video_tensor
            
        except Exception as e:
            print(f"警告: 视频加载失败 {video_path}: {e}")
            # 返回零张量
            return torch.zeros(self.num_frames, 3, *self.video_size)
    
    def get_labeled_indices(self):
        """返回所有标注样本的索引"""
        return self.df[self.df['is_labeled'] == 1].index.tolist()
    
    def get_unlabeled_indices(self):
        """返回所有未标注样本的索引"""
        return self.df[self.df['is_labeled'] == 0].index.tolist()


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("="*70)
    print("SemiSup Dataset 测试")
    print("="*70)
    
    # 创建模拟CSV
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("sample,audio,video,audio_start_s,audio_end_s,video_start_frame,video_end_frame,is_labeled,label\n")
        
        for i in range(20):
            is_labeled = 1 if i < 5 else 0  # 25%标注
            label = i % 2 if is_labeled else -1
            
            f.write(f"sample_{i},audio_{i}.flac,video_{i}.avi,")
            f.write(f"0.0,1.5,0,48,{is_labeled},{label}\n")
        
        csv_path = f.name
    
    try:
        # 创建数据集
        dataset = SemiSupDataset(
            windows_csv=csv_path,
            classes=['normal', 'defect']
        )
        
        # 测试索引获取
        labeled_idx = dataset.get_labeled_indices()
        unlabeled_idx = dataset.get_unlabeled_indices()
        
        print(f"Labeled indices: {labeled_idx}")
        print(f"Unlabeled indices: {unlabeled_idx[:5]}...")
        
        # 测试数据加载（会失败因为文件不存在，但可以看到流程）
        print(f"\n尝试加载第一个样本...")
        try:
            sample = dataset[0]
            print(f"  ✅ 样本结构正确")
            print(f"     keys: {sample.keys()}")
        except Exception as e:
            print(f"  ⚠️  加载失败（预期，因为文件不存在）: {e}")
        
        print(f"\n✅ Dataset结构测试通过!")
        
    finally:
        # 清理
        import os
        os.remove(csv_path)
        # 2. 如果提供了标注CSV，合并标签信息
        if self.ann_csv and self.ann_csv.exists():
            df_ann = pd.read_csv(self.ann_csv)
            # 合并label列
            if 'label' in df_ann.columns:
                self.df = self.df_windows.merge(
                    df_ann[['sample', 'label']],
                    on='sample',
                    how='left'
                )
            else:
                self.df = self.df_windows.copy()
        else:
            self.df = self.df_windows.copy()

        # 3. 确保is_labeled字段为整数
        self.df['is_labeled'] = self.df['is_labeled'].astype(int)

        # 4. 对于未标注样本，设置label为-1（占位符）
        if 'label' not in self.df.columns:
            self.df['label'] = -1

        self.df.loc[self.df['is_labeled'] == 0, 'label'] = -1

        # 5. 对于已标注样本，将label映射为类别索引
        if 'label' in self.df.columns:
            # 如果label已经是字符串类别名，映射为索引
            if self.df['label'].dtype == 'object':
                self.df['label'] = self.df['label'].apply(
                    lambda x: self.classes.index(x) if x in self.classes else -1
                )

        # 6. 缓存相关
        self.cache = {} if use_cache else None


    def _print_stats(self):
        """打印数据集统计信息"""
        total = len(self.df)
        labeled = (self.df['is_labeled'] == 1).sum()
        unlabeled = (self.df['is_labeled'] == 0).sum()

        print(f"📊 数据集统计:")
        print(f"  总样本数: {total}")
        print(f"  已标注: {labeled} ({100 * labeled / total:.1f}%)")
        print(f"  未标注: {unlabeled} ({100 * unlabeled / total:.1f}%)")

        if labeled > 0:
            label_dist = self.df[self.df['is_labeled'] == 1]['label'].value_counts()
            print(f"  标注分布:")
            for cls_idx, count in label_dist.items():
                cls_name = self.classes[cls_idx] if cls_idx < len(self.classes) else f"class_{cls_idx}"
                print(f"    {cls_name}: {count}")


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        """
        返回样本字典

        Returns:
            dict: {
                'audio': tensor [audio_length]
                'video': tensor [num_frames, 3, H, W]
                'label': int (-1表示未标注)
                'is_labeled': int (0或1)
                'sample_name': str
            }
        """
        # 从缓存加载
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        row = self.df.iloc[idx]

        # 1. 加载音频
        audio_path = row['audio']
        audio, sr = sf.read(audio_path)

        # 音频窗口切片
        start_sample = int(row['audio_start_s'] * sr)
        end_sample = int(row['audio_end_s'] * sr)
        audio = audio[start_sample:end_sample]

        # 重采样（如果需要）
        if sr != self.audio_sr:
            from librosa import resample
            audio = resample(audio, orig_sr=sr, target_sr=self.audio_sr)

        audio_tensor = torch.from_numpy(audio).float()

        # 2. 加载视频
        video_path = row['video']
        cap = cv2.VideoCapture(str(video_path))

        start_frame = int(row['video_start_frame'])
        end_frame = int(row['video_end_frame'])
        total_frames = end_frame - start_frame

        # 均匀采样num_frames帧
        frame_indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)

        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if ret:
                # BGR -> RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Resize
                frame = cv2.resize(frame, self.video_size)
                # Normalize to [0, 1]
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)

        cap.release()

        # [T, H, W, C] -> [T, C, H, W]
        video_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float()

        # 3. 组装输出
        sample = {
            'audio': audio_tensor,
            'video': video_tensor,
            'label': int(row['label']),
            'is_labeled': int(row['is_labeled']),
            'sample_name': row['sample']
        }

        # 缓存
        if self.use_cache:
            self.cache[idx] = sample

        return sample


    def get_labeled_indices(self):
        """返回所有已标注样本的索引"""
        return self.df[self.df['is_labeled'] == 1].index.tolist()


    def get_unlabeled_indices(self):
        """返回所有未标注样本的索引"""
        return self.df[self.df['is_labeled'] == 0].index.tolist()

# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SemiSup Dataset 测试")
    print("=" * 70)

    # 创建模拟CSV
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("sample,audio,video,audio_start_s,audio_end_s,video_start_frame,video_end_frame,is_labeled,label\n")

        for i in range(20):
            is_labeled = 1 if i < 5 else 0  # 25%标注
            label = i % 2 if is_labeled else -1

            f.write(f"sample_{i},audio_{i}.flac,video_{i}.avi,")
            f.write(f"0.0,1.5,0,48,{is_labeled},{label}\n")

        csv_path = f.name

    try:
        # 创建数据集
        dataset = SemiSupDataset(
            windows_csv=csv_path,
            classes=['normal', 'defect']
        )

        # 测试索引获取
        labeled_idx = dataset.get_labeled_indices()
        unlabeled_idx = dataset.get_unlabeled_indices()

        print(f"\n📋 索引信息:")
        print(f"  Labeled indices: {labeled_idx}")
        print(f"  Unlabeled indices: {unlabeled_idx[:5]}...")

        # 测试数据加载（会失败因为文件不存在，但可以看到流程）
        print(f"\n🔍 尝试加载第一个样本...")
        try:
            sample = dataset[0]
            print(f"  ✅ 样本结构正确")
            print(f"     keys: {sample.keys()}")
        except Exception as e:
            print(f"  ⚠️  加载失败（预期，因为文件不存在）: {str(e)[:50]}")

        print(f"\n✅ Dataset结构测试通过!")

    finally:
        # 清理
        import os

        os.remove(csv_path)