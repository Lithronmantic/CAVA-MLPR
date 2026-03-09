# -*- coding: utf-8 -*-
import os, csv, math
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

# 可选：使用你给的多模态增强模块（弱/强都支持）
try:
    from augmentation_module import MultiModalAugmentation
except Exception:
    MultiModalAugmentation = None


class MultiClassAVCSVDataset(torch.utils.data.Dataset):
    """
    适配你当前 CSV 的多类单标签数据集（与 CrossEntropyLoss 对齐）
    支持的 CSV 列（至少包含以下其一满足标签解析）：
      - label: 整数索引 [0..C-1]
      - class_name: 类名 (必须能在 class_names 里找到)
      - class_0..class_{C-1}: one-hot（恰有一个 1）

    建议/可选列（用于时域对齐）：
      - video_start_frame, video_end_frame
      - audio_start_s, audio_end_s

    返回：
      video: [T_v, 3, H, W], float32 in [0,1]
      audio: [T_a, mel_bins, frames_per_slice]  (默认 T_v==T_a)
      label_idx: torch.long 标量（单标签多类）
    """
    def __init__(
        self,
        csv_path: str,
        root: str = "",
        class_names: Optional[List[str]] = None,  # 若不给，则按 label 直接使用索引
        video_num_frames: int = 8,                # 你日志里是 8 帧
        video_size: Tuple[int,int] = (224,224),
        audio_sr: int = 16000,
        mel_bins: int = 128,
        mel_frames_per_slice: int = 64,          # 你日志里是 128x64
        augmentation_mode: Optional[str] = None, # None/'weak'/'strong'
    ):
        super().__init__()
        self.root = root
        self.class_names = list(class_names) if class_names is not None else None
        self.C = len(self.class_names) if self.class_names is not None else None

        self.T_v = int(video_num_frames)
        self.T_a = int(video_num_frames)   # 对齐 slices 数
        self.size = tuple(video_size)
        self.sr = int(audio_sr)
        self.mel_bins = int(mel_bins)
        self.mel_frames = int(mel_frames_per_slice)

        # 增强（可选）
        if augmentation_mode and MultiModalAugmentation is not None:
            self.augment = MultiModalAugmentation(
                mode=augmentation_mode,
                audio_sr=self.sr,
                video_size=self.size,
                num_frames=self.T_v
            )
        else:
            self.augment = None

        # 读取 CSV
        self.rows: List[Dict] = []
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = [c.strip() for c in reader.fieldnames]
            for row in reader:
                # 路径处理
                vpath = row.get("video_path", row.get("video", ""))
                apath = row.get("audio_path", row.get("audio", ""))
                if self.root and vpath and not os.path.isabs(vpath):
                    vpath = os.path.join(self.root, vpath)
                if self.root and apath and not os.path.isabs(apath):
                    apath = os.path.join(self.root, apath)

                # 对齐字段（可空）
                s_v = _safe_int(row.get("video_start_frame"))
                e_v = _safe_int(row.get("video_end_frame"))
                s_a = _safe_float(row.get("audio_start_s"))
                e_a = _safe_float(row.get("audio_end_s"))

                # 解析标签为 0..C-1
                y_idx = self._parse_label(row)

                self.rows.append({
                    "clip_id": row.get("sample", os.path.basename(vpath) or os.path.basename(apath) or str(len(self.rows))),
                    "video_path": vpath,
                    "audio_path": apath,
                    "label_idx": y_idx,
                    "video_start_frame": s_v,
                    "video_end_frame": e_v,
                    "audio_start_s": s_a,
                    "audio_end_s": e_a,
                })

    # ------------------- 标签解析：锁死为 Long 索引 -------------------
    def _parse_label(self, row: Dict) -> int:
        # 1) 直接数字索引
        v = str(row.get("label", "")).strip()
        if v != "" and v.lstrip("-").isdigit():
            idx = int(v)
            if self.C is not None and not (0 <= idx < self.C):
                raise ValueError(f"label 超出范围: {idx} / C={self.C}")
            return idx

        # 2) 通过类名映射
        cname = str(row.get("class_name", "")).strip()
        if cname and self.class_names is not None:
            if cname in self.class_names:
                return self.class_names.index(cname)
            raise ValueError(f"class_name 不在 class_names 列表中: {cname}")

        # 3) one-hot（class_0..class_{C-1}）
        if self.C is not None:
            onehot = []
            for i in range(self.C):
                k = f"class_{i}"
                onehot.append(float(row.get(k, 0)))
            if sum(x > 0.5 for x in onehot) == 1:
                return int(np.argmax(onehot))

        raise ValueError(f"无法解析标签（需要 label 索引 / class_name / one-hot）: {row}")

    def __len__(self): return len(self.rows)

    # ------------------- 视频：按起止帧均匀采样 T_v 帧 -------------------
    def _load_video_frames(self, path: str, size: Tuple[int,int], start_f: Optional[int], end_f: Optional[int], T: int):
        import cv2
        import numpy as np

        if not path or (not os.path.exists(path)):
            # 兜底：纯噪声
            frames = np.random.rand(T, size[1], size[0], 3).astype("float32")
            frames = torch.from_numpy(frames).permute(0,3,1,2)  # [T,3,H,W]
            return frames

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            # 兜底噪声
            frames = np.random.rand(T, size[1], size[0], 3).astype("float32")
            frames = torch.from_numpy(frames).permute(0,3,1,2)
            return frames

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        s = 0 if (start_f is None or start_f < 0) else min(start_f, total-1)
        e = total-1 if (end_f is None or end_f < 0) else min(end_f, total-1)
        if e < s: e = s

        idx = np.linspace(s, e, num=T).astype(int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = (np.random.rand(size[1], size[0], 3)*255).astype("uint8")
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame = frame[:, :, ::-1]  # BGR -> RGB
            frames.append(frame)
        cap.release()

        frames = np.stack(frames, axis=0).astype("float32") / 255.0  # [T,H,W,3]
        frames = torch.from_numpy(frames).permute(0,3,1,2).contiguous()  # [T,3,H,W]
        return frames

    # ------------------- 音频：裁成 T_a 片，每片 mel_bins x mel_frames -------------------
    def _load_audio_slices(self, path: str, sr: int, start_s: Optional[float], end_s: Optional[float],
                           T: int, mel_bins: int, frames_per_slice: int) -> torch.Tensor:
        """
        先截取 [start_s, end_s] 段；转 log-Mel；线性重采样到 T*frames_per_slice，然后 reshape 成 [T, mel, frames_per_slice]
        """
        import numpy as np
        mel = None
        try:
            import torchaudio
            wav, s = torchaudio.load(path)  # [C, N]
            if s != sr:
                wav = torchaudio.functional.resample(wav, s, sr)
            mono = wav.mean(dim=0, keepdim=True)  # [1, N]
            # 裁段
            if start_s is not None and end_s is not None and end_s > start_s:
                s_i = int(max(0, start_s) * sr)
                e_i = int(end_s * sr)
                e_i = max(e_i, s_i+1)
                mono = mono[:, s_i:e_i]
            mel_extractor = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=mel_bins)
            mel = mel_extractor(mono)  # [1, mel, Tspec]
            mel = torch.log(mel + 1e-6)
        except Exception:
            # 回退到 librosa
            import librosa
            if path and os.path.exists(path):
                y, s = librosa.load(path, sr=sr, mono=True)
            else:
                y = np.random.randn(sr)  # 1s noise
                s = sr
            if start_s is not None and end_s is not None and end_s > start_s:
                s_i = int(max(0, start_s) * s)
                e_i = int(end_s * s)
                e_i = max(e_i, s_i+1)
                y = y[s_i:e_i]
            S = librosa.feature.melspectrogram(y=y, sr=s, n_mels=mel_bins)
            S_db = librosa.power_to_db(S, ref=np.max)
            mel = torch.from_numpy(S_db).unsqueeze(0).float()  # [1, mel, Tspec]

        Tspec = mel.size(-1)
        target_T = T * frames_per_slice
        if Tspec == 0:
            mel = mel.new_zeros(1, mel_bins, target_T)
        elif Tspec >= target_T:
            idx = torch.linspace(0, Tspec - 1, steps=target_T).long()
            mel = mel.index_select(-1, idx)
        else:
            pad = target_T - Tspec
            mel = torch.cat([mel, mel.new_zeros(1, mel_bins, pad)], dim=-1)

        mel = mel.view(1, mel_bins, T, frames_per_slice).permute(2,1,3,0).squeeze(-1)  # [T, mel, frames]
        return mel.contiguous()

    # ------------------- getitem -------------------
    def __getitem__(self, idx: int):
        row = self.rows[idx]
        video = self._load_video_frames(
            row["video_path"], self.size,
            row["video_start_frame"], row["video_end_frame"],
            self.T_v
        )  # [T_v,3,H,W]

        audio = self._load_audio_slices(
            row["audio_path"], self.sr,
            row["audio_start_s"], row["audio_end_s"],
            self.T_a, self.mel_bins, self.mel_frames
        )  # [T_a,mel,frames]

        if self.augment is not None:
            # 你的增强模块签名是 (audio, video) → (audio, video)
            try:
                a_aug, v_aug = self.augment(audio.flatten(), video)  # 模块原型是 1D audio / [T,3,H,W]
                # 这里我们不改变切片结构，所以仅对 video 使用模块带来的颜色/翻转等增强
                video = v_aug
            except Exception:
                pass

        y = torch.tensor(int(row["label_idx"]), dtype=torch.long)  # 单标签多类（CE 要求 long 索引）:contentReference[oaicite:3]{index=3}
        return video, audio, y


# ------------------- collate：返回三元组，直接喂给你的 Trainer -------------------
def collate_av(batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    v = torch.stack([b[0] for b in batch], dim=0)  # [B, T, 3, H, W]
    a = torch.stack([b[1] for b in batch], dim=0)  # [B, T, mel, frames]
    y = torch.stack([b[2] for b in batch], dim=0).long()  # [B]
    return v, a, y


# ------------------- utils -------------------
def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None
