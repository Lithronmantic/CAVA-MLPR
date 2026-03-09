# src/avtop/data/grouped_split.py
"""
分组数据划分工具：防止同一视频的不同片段分散到train/val
"""
import csv
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from collections import defaultdict
from typing import List, Tuple, Dict


class GroupedDataSplitter:
    """按视频ID分组的数据划分器"""

    def __init__(self, csv_path: str, group_column: str = "clip_id",
                 label_column: str = "label"):
        """
        Args:
            csv_path: CSV文件路径
            group_column: 分组列名（通常是视频ID或场景ID）
            label_column: 标签列名
        """
        self.csv_path = Path(csv_path)
        self.group_column = group_column
        self.label_column = label_column

        # 读取数据
        self.rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.rows.append(row)

        # 提取视频ID（从clip_id中提取基础视频ID）
        self.video_ids = self._extract_video_ids()
        self.labels = np.array([int(row[label_column]) for row in self.rows])

    def _extract_video_ids(self) -> List[str]:
        """
        从clip_id提取视频ID
        例如: "video_001_clip_03" -> "video_001"
        """
        video_ids = []
        for row in self.rows:
            clip_id = row[self.group_column]

            # 尝试各种模式提取视频ID
            if '_clip_' in clip_id:
                video_id = clip_id.split('_clip_')[0]
            elif '_seg_' in clip_id:
                video_id = clip_id.split('_seg_')[0]
            elif '_frame_' in clip_id:
                video_id = clip_id.split('_frame_')[0]
            else:
                # 如果没有明确的分隔符，使用整个clip_id
                # （假设每个clip_id对应不同视频）
                video_id = clip_id

            video_ids.append(video_id)

        return video_ids

    def split_stratified_group(self, n_splits: int = 5, val_fold: int = 0,
                               random_state: int = 42) -> Tuple[List[int], List[int]]:
        """
        使用StratifiedGroupKFold划分
        保证：1) 同一视频的所有片段在同一个fold  2) 各fold的类别比例近似

        Args:
            n_splits: 折数
            val_fold: 用作验证集的fold索引
            random_state: 随机种子

        Returns:
            (train_indices, val_indices)
        """
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # 准备数据
        X = np.arange(len(self.rows))
        y = self.labels
        groups = np.array(self.video_ids)

        # 划分
        splits = list(sgkf.split(X, y, groups))
        train_idx, val_idx = splits[val_fold]

        return train_idx.tolist(), val_idx.tolist()

    def split_group(self, n_splits: int = 5, val_fold: int = 0,
                    random_state: int = 42) -> Tuple[List[int], List[int]]:
        """
        使用GroupKFold划分（不考虑类别平衡）
        仅保证同一视频的所有片段在同一个fold
        """
        gkf = GroupKFold(n_splits=n_splits)

        X = np.arange(len(self.rows))
        groups = np.array(self.video_ids)

        splits = list(gkf.split(X, groups=groups))
        train_idx, val_idx = splits[val_fold]

        return train_idx.tolist(), val_idx.tolist()

    def analyze_split(self, train_idx: List[int], val_idx: List[int]) -> Dict:
        """分析划分质量"""
        train_videos = set(self.video_ids[i] for i in train_idx)
        val_videos = set(self.video_ids[i] for i in val_idx)

        # 检查泄漏
        leaked = train_videos & val_videos

        # 统计标签分布
        train_labels = self.labels[train_idx]
        val_labels = self.labels[val_idx]

        train_pos = np.sum(train_labels == 1)
        train_neg = np.sum(train_labels == 0)
        val_pos = np.sum(val_labels == 1)
        val_neg = np.sum(val_labels == 0)

        return {
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_train_videos': len(train_videos),
            'n_val_videos': len(val_videos),
            'leaked_videos': len(leaked),
            'train_pos': train_pos,
            'train_neg': train_neg,
            'train_pos_ratio': train_pos / (train_pos + train_neg),
            'val_pos': val_pos,
            'val_neg': val_neg,
            'val_pos_ratio': val_pos / (val_pos + val_neg)
        }

    def save_split(self, train_idx: List[int], val_idx: List[int],
                   output_dir: str = "./splits"):
        """保存划分后的CSV文件"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存训练集
        train_rows = [self.rows[i] for i in train_idx]
        train_path = output_dir / f"{self.csv_path.stem}_train.csv"
        with open(train_path, 'w', newline='', encoding='utf-8') as f:
            if train_rows:
                writer = csv.DictWriter(f, fieldnames=train_rows[0].keys())
                writer.writeheader()
                writer.writerows(train_rows)

        # 保存验证集
        val_rows = [self.rows[i] for i in val_idx]
        val_path = output_dir / f"{self.csv_path.stem}_val.csv"
        with open(val_path, 'w', newline='', encoding='utf-8') as f:
            if val_rows:
                writer = csv.DictWriter(f, fieldnames=val_rows[0].keys())
                writer.writeheader()
                writer.writerows(val_rows)

        print(f"✅ Saved train set: {train_path} ({len(train_idx)} samples)")
        print(f"✅ Saved val set:   {val_path} ({len(val_idx)} samples)")

        return train_path, val_path


def prepare_grouped_splits(csv_path: str, output_dir: str = "./splits",
                           n_splits: int = 5, val_fold: int = 0,
                           use_stratified: bool = True):
    """
    便捷函数：准备分组划分

    Args:
        csv_path: 原始CSV路径
        output_dir: 输出目录
        n_splits: K折数
        val_fold: 验证集fold
        use_stratified: 是否使用分层分组（推荐）
    """
    splitter = GroupedDataSplitter(csv_path)

    # 执行划分
    if use_stratified:
        train_idx, val_idx = splitter.split_stratified_group(n_splits, val_fold)
        method = "StratifiedGroupKFold"
    else:
        train_idx, val_idx = splitter.split_group(n_splits, val_fold)
        method = "GroupKFold"

    # 分析
    stats = splitter.analyze_split(train_idx, val_idx)

    print(f"\n{'=' * 60}")
    print(f"Split Analysis ({method})")
    print(f"{'=' * 60}")
    print(f"Train: {stats['n_train']} samples from {stats['n_train_videos']} videos")
    print(f"  Pos: {stats['train_pos']} ({stats['train_pos_ratio']:.2%})")
    print(f"  Neg: {stats['train_neg']} ({1 - stats['train_pos_ratio']:.2%})")
    print(f"\nVal:   {stats['n_val']} samples from {stats['n_val_videos']} videos")
    print(f"  Pos: {stats['val_pos']} ({stats['val_pos_ratio']:.2%})")
    print(f"  Neg: {stats['val_neg']} ({1 - stats['val_pos_ratio']:.2%})")
    print(f"\n⚠️  Leaked videos: {stats['leaked_videos']}")
    if stats['leaked_videos'] == 0:
        print("✅ No data leakage detected!")
    else:
        print("❌ Warning: Some videos appear in both train and val!")
    print(f"{'=' * 60}\n")

    # 保存
    train_path, val_path = splitter.save_split(train_idx, val_idx, output_dir)

    return train_path, val_path, stats


if __name__ == '__main__':
    # 示例用法
    import argparse

    parser = argparse.ArgumentParser(description="Group-based data splitting")
    parser.add_argument('--csv', type=str, required=True, help='Input CSV path')
    parser.add_argument('--output', type=str, default='./splits', help='Output directory')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of folds')
    parser.add_argument('--val_fold', type=int, default=0, help='Validation fold index')
    parser.add_argument('--no_stratify', action='store_true', help='Disable stratified split')

    args = parser.parse_args()

    prepare_grouped_splits(
        args.csv,
        args.output,
        args.n_splits,
        args.val_fold,
        use_stratified=not args.no_stratify
    )