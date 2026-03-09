import os, csv, torch, numpy as np
from typing import List, Dict, Tuple

class BinaryAVCSVDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: str, root: str = "", T_v: int = 16, T_a: int = 32, mel_bins: int = 64,
                 sample_rate: int = 16000, size: Tuple[int,int]=(112,112)):
        super().__init__(); self.rows=[]; self.root=root; self.Tv, self.Ta, self.mel=T_v, T_a, mel_bins; self.sample_rate=sample_rate; self.size=size
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r: self.rows.append(row)
    def __len__(self): return len(self.rows)
    def _load_video_uniform(self, path: str, T: int, size: Tuple[int,int]):
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            frames = [(255*np.random.rand(size[1], size[0], 3)).astype("uint8") for _ in range(T)]
        else:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or T
            idxs = [int(i) for i in torch.linspace(0, max(total-1,0), steps=T)]
            frames=[]
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, f = cap.read()
                if not ok:
                    f = frames[-1] if len(frames)>0 else (255*np.random.rand(size[1], size[0], 3)).astype("uint8")
                f = cv2.resize(f, size, interpolation=cv2.INTER_AREA); f = f[:, :, ::-1]; frames.append(f)
            cap.release()
        v = torch.from_numpy(np.stack(frames, axis=0)).permute(0,3,1,2).float()/255.0; return v

    def _load_audio_raw(self, path: str, duration: float = None, sr: int = 16000):
        """加载原始波形，可选指定时长"""
        try:
            import torchaudio
            wav, orig_sr = torchaudio.load(path)
            if orig_sr != sr:
                wav = torchaudio.functional.resample(wav, orig_sr, sr)

            # 转单声道
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)

            # 可选：裁剪/填充到指定时长
            if duration is not None:
                target_len = int(duration * sr)
                if wav.shape[1] > target_len:
                    wav = wav[:, :target_len]
                elif wav.shape[1] < target_len:
                    pad = target_len - wav.shape[1]
                    wav = torch.cat([wav, torch.zeros(1, pad)], dim=1)

            return wav.squeeze(0)  # 返回 (T,)
        except Exception:
            return torch.zeros(int(duration * sr) if duration else sr)

    def __getitem__(self, idx: int):
        r=self.rows[idx]; vid=os.path.join(self.root, r["video_path"]) if self.root else r["video_path"]
        aud=os.path.join(self.root, r["audio_path"]) if self.root else r["audio_path"]; label=int(r["label"])
        video=self._load_video_uniform(vid, self.Tv, self.size); audio = self._load_audio_raw(aud, duration=3.0, sr=self.sample_rate)
        segs=[]; 
        if r.get("seg_starts") and r.get("seg_ends"):
            ss=[s for s in r["seg_starts"].split(",") if s.strip()!=""]; ee=[e for e in r["seg_ends"].split(",") if e.strip()!=""]
            for s,e in zip(ss,ee):
                try: segs.append((int(s), int(e), label))
                except: pass
        return {"video": video, "audio": audio, "label_idx": label, "clip_id": r["clip_id"], "gt_segments": segs}

def collate(batch: List[Dict]):
    video = torch.stack([b["video"] for b in batch], dim=0); audio = torch.stack([b["audio"] for b in batch], dim=0)
    labels = torch.tensor([b["label_idx"] for b in batch], dtype=torch.long); clip_id = [b["clip_id"] for b in batch]
    gt_segments = [b["gt_segments"] for b in batch]
    return {"video": video, "audio": audio, "label_idx": labels, "clip_id": clip_id, "gt_segments": gt_segments}

AVTopDataset = BinaryAVCSVDataset
