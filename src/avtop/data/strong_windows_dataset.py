# -*- coding: utf-8 -*-
import os, random, math
import torch
import pandas as pd
from torch.utils.data import Dataset

class StrongAVWindowDataset(Dataset):
    """
    强监督：整段类别已知；在有效时段内随机采样固定时长窗口（音/视频同起止），返回一条训练样本。
    需要 CSV 列：
      clip_id, video_path, audio_path, label, a_start_s, a_end_s
    """
    def __init__(self, csv_path, root=".", split="train",
                 win_s=3.0,  # 窗口秒数（音频）
                 v_frames=45,  # 视频帧数（30fps→1.5s；若想3s就90）
                 fps_default=30.0,
                 sample_rate=16000,
                 video_reader=None,  # 你现有的视频解码函数，形如 fn(path, start_s, dur_s, frames)->(T,C,H,W)
                 audio_reader=None,  # 你现有的音频解码函数，形如 fn(path, start_s, dur_s, sr)->wave(1,L)
                 transforms=None):
        self.df = pd.read_csv(csv_path)
        self.root = root
        self.split = split
        self.win_s = float(win_s)
        self.v_frames = int(v_frames)
        self.fps_default = float(fps_default)
        self.sr = int(sample_rate)
        self.video_reader = video_reader
        self.audio_reader = audio_reader
        self.transforms = transforms

        # 基本检查
        need = {"clip_id","video_path","audio_path","label","a_start_s","a_end_s"}
        miss = need - set(self.df.columns)
        if miss:
            raise ValueError(f"CSV 缺少列: {sorted(miss)}")

    def __len__(self): return len(self.df)

    def _rand_start(self, a0, a1):
        # 在 [a0, a1 - win_s] 均匀采样；如有效时段过短，则取中点并允许越界裁剪/补零
        span = max(0.0, float(a1) - float(a0) - self.win_s)
        if span > 1e-6:
            return float(a0) + random.random() * span
        else:
            return max(float(a0), float(a1) - self.win_s)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        vpath = os.path.join(self.root, str(r["video_path"]))
        apath = os.path.join(self.root, str(r["audio_path"]))
        y = int(r["label"])
        a0, a1 = float(r["a_start_s"]), float(r["a_end_s"])
        start_s = self._rand_start(a0, a1)

        # ——读音频：返回 shape (1, L)
        wave = self.audio_reader(apath, start_s, self.win_s, self.sr)  # 你已有的读取函数
        # ——读视频：在 [start_s, start_s+win_s) 均匀采样 v_frames 帧
        video = self.video_reader(vpath, start_s, self.win_s, self.v_frames)  # 你已有的读取函数

        if self.transforms:
            video, wave = self.transforms(video, wave, split=self.split)

        return {"video": video, "audio": wave, "label": y, "clip_id": r["clip_id"], "start_s": start_s}

    @staticmethod
    def collate_fn(batch):
        # 视频统一 (B,T,C,H,W)，音频统一 (B,L)；你现有的 collate 也可复用
        videos = torch.stack([b["video"] for b in batch], 0)
        audios = torch.stack([b["audio"] for b in batch], 0)
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        meta = {k: [b[k] for b in batch] for k in batch[0] if k not in ("video","audio","label")}
        return {"video": videos, "audio": audios, "label": labels, "meta": meta}
