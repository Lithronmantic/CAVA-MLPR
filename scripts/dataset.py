# -*- coding: utf-8 -*-
import os, csv
from typing import List, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# -------------------- Small helpers --------------------
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


def _to_none_like(v):
    """Normalize empty-like values to None."""
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("", "none", "null"):
        return None
    return v


def resolve_path(rel: str, data_root: Optional[str]) -> Path:
    """Resolve CSV path robustly across Windows/Linux separators."""
    s = (rel or "").strip().strip('"').strip("'")
    s = os.path.expandvars(os.path.expanduser(s))
    # Linux cannot interpret "\" as path separator from Windows-exported CSV.
    s_norm = s.replace("\\", "/")

    p = Path(s_norm)
    if p.is_absolute() and p.exists():
        return p

    root = _to_none_like(data_root)
    if root is not None:
        base = Path(os.path.expandvars(os.path.expanduser(str(root))))
        cand = base / s_norm
        if cand.exists():
            return cand
        for prefix in ("data/", "data\\", "./", ".\\"):
            pfx = prefix.replace("\\", "/")
            if s_norm.startswith(pfx):
                cand2 = base / s_norm[len(pfx):]
                if cand2.exists():
                    return cand2

    return Path(s_norm)

# -------------------- 鏁版嵁闆?--------------------
class AVFromCSV(Dataset):
    """
    鍗曟爣绛惧绫伙紙涓?CrossEntropy/FocalCE 瀵归綈锛夌殑闊宠棰戞暟鎹泦銆?

    鏀寔 CSV 鍒楋紙鑷冲皯婊¤冻鍏朵竴锛夛細
      - label:       鏁存暟绱㈠紩 [0..C-1]
      - class_name:  绫诲悕锛堝繀椤昏兘鍦?class_names 涓壘鍒帮級
      - class_0..class_{C-1}: one-hot锛堟伆鏈変竴涓?1锛?

    寤鸿/鍙€夊垪锛堢敤浜庢椂鍩熷榻愶級锛?
      - video_start_frame, video_end_frame
      - audio_start_s,    audio_end_s

    杩斿洖锛?
      video: [T, 3, H, W], float32 in [0,1]
      audio: [T, mel_bins, frames_per_slice]
      y:     torch.long锛堝崟鏍囩绫荤储寮曪級
    """
    def __init__(
        self,
        csv_path: str,
        data_root: Optional[str],
        num_classes: int,
        class_names: List[str],
        video_cfg: Dict,
        audio_cfg: Dict,
        is_unlabeled: bool = False,
        augmentation_mode: Optional[str] = None,  # 淇濈暀鎺ュ彛锛?weak'/'strong'/None
    ):
        super().__init__()
        # root 涓?None 琛ㄧず涓嶅姞浠讳綍鍓嶇紑锛堜笌鑷鑴氭湰涓€鑷达級
        self.root: Optional[str] = _to_none_like(data_root)
        self.C = int(num_classes)
        self.class_names = list(class_names)

        # video
        self.T_v = int(video_cfg.get("num_frames", video_cfg.get("frames", 8)))
        v_size = int(video_cfg.get("size", 224))
        self.v_size = (v_size, v_size)

        # audio 鍩烘湰鍙傛暟
        self.sr = int(audio_cfg.get("sample_rate", audio_cfg.get("sr", 16000)))
        self.mel_bins = int(audio_cfg.get("n_mels", audio_cfg.get("mel_bins", 128)))
        self.frames_per_slice = int(audio_cfg.get("segment_frames", audio_cfg.get("target_len", 64)))

        # STFT 鍏抽敭鍙傛暟锛堜笌棰勫鐞嗕竴鑷达紝闃叉涓嶅悓搴撻粯璁や笉涓€鑷达級
        self.n_fft = int(audio_cfg.get("n_fft", 2048))
        self.hop_length = int(audio_cfg.get("hop_length", 512))
        # 鎴戜滑鍦ㄥ唴閮ㄥ仛浜嗛暱搴﹀厹搴曪紝鍥犳榛樿 center=False 鏇寸ǔ瀹?
        self.center = bool(audio_cfg.get("center", False))
        self.pad_mode = str(audio_cfg.get("pad_mode", "reflect"))

        self.is_unlabeled = bool(is_unlabeled)
        self.rows: List[Dict] = []

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_v = row.get("video_path", row.get("video", "")) or ""
                raw_a = row.get("audio_path", row.get("audio", "")) or ""

                vpath = resolve_path(raw_v, self.root)
                apath = resolve_path(raw_a, self.root)

                y_idx = None
                if not self.is_unlabeled:
                    y_idx = self._parse_label(row)

                self.rows.append({
                    "clip_id": row.get("sample", os.path.basename(str(vpath)) or os.path.basename(str(apath)) or str(len(self.rows))),
                    "video_path": str(vpath),
                    "audio_path": str(apath),
                    "label_idx": y_idx,
                    "video_start_frame": _safe_int(row.get("video_start_frame")),
                    "video_end_frame":   _safe_int(row.get("video_end_frame")),
                    "audio_start_s":     _safe_float(row.get("audio_start_s")),
                    "audio_end_s":       _safe_float(row.get("audio_end_s")),
                })

    # -------- 鏍囩瑙ｆ瀽锛氶攣姝讳负 Long 绱㈠紩 --------
    def _parse_label(self, row: Dict) -> int:
        v = str(row.get("label", "")).strip()
        if v != "" and v.lstrip("-").isdigit():
            idx = int(v)
            if not (0 <= idx < self.C):
                raise ValueError(f"label 瓒呰寖鍥? {idx}/{self.C}")
            return idx

        cname = (row.get("class_name") or "").strip()
        if cname:
            if cname in self.class_names:
                return self.class_names.index(cname)
            else:
                raise ValueError(f"class_name 涓嶅湪 class_names: {cname}")

        # one-hot
        oh = []
        for i in range(self.C):
            k = f"class_{i}"
            try:
                oh.append(float(row.get(k, 0)))
            except Exception:
                oh.append(0.0)
        if sum(x > 0.5 for x in oh) == 1:
            return int(np.argmax(oh))

        raise ValueError(f"鏃犳硶瑙ｆ瀽鏍囩锛堥渶瑕?label / class_name / one-hot锛夛細{row}")

    def __len__(self):
        return len(self.rows)

    # -------- 瑙嗛鍔犺浇锛氭寜璧锋甯у潎鍖€閲囨牱 T_v 甯?--------
    def _load_video_frames(self, path: str, start_f: Optional[int], end_f: Optional[int]) -> torch.Tensor:
        import cv2
        H, W = self.v_size[1], self.v_size[0]
        T = self.T_v

        if not path or (not os.path.exists(path)):
            # 瑙嗛缂哄け鏃惰繑鍥為殢鏈哄抚锛堜繚鎸佽缁冧笉涓柇锛涢煶棰戞槸鍏抽敭妯℃€侊級
            frames = np.random.rand(T, H, W, 3).astype("float32")
            return torch.from_numpy(frames).permute(0,3,1,2).contiguous()

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            frames = np.random.rand(T, H, W, 3).astype("float32")
            return torch.from_numpy(frames).permute(0,3,1,2).contiguous()

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        s = 0 if (start_f is None or start_f < 0) else min(start_f, total-1)
        e = total-1 if (end_f   is None or end_f   < 0) else min(end_f,   total-1)
        if e < s:
            e = s

        idx = np.linspace(s, e, num=T).astype(int)
        frames = []
        for i in idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if not ok or frame is None:
                frame = (np.random.rand(H, W, 3)*255).astype("uint8")
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
            frame = frame[:, :, ::-1]  # BGR -> RGB
            frames.append(frame)
        cap.release()

        frames = np.stack(frames, axis=0).astype("float32") / 255.0  # [T,H,W,3]
        frames = torch.from_numpy(frames).permute(0,3,1,2).contiguous()  # [T,3,H,W]
        return frames

    # -------- 鍐呴儴锛氬皢 numpy 娉㈠舰 -> log-mel锛堥暱搴﹀厹搴曪紝闃插憡璀︼級 --------
    def _compute_mel_from_numpy(self, y: np.ndarray, sr: int) -> torch.Tensor:
        """
        - 绌烘暟缁勭洿鎺ユ姤閿?
        - 闀垮害 < n_fft 鍒欎袱渚ф垨灏鹃儴琛ラ綈鍒?n_fft
        - 缁熶竴 center=False锛岄伩鍏?librosa 鍦ㄧ煭搴忓垪鏃剁殑 STFT 鍛婅
        """
        if y is None or (hasattr(y, "size") and y.size == 0):
            raise ValueError("Empty waveform before STFT/mel.")

        if y.ndim > 1:
            y = np.mean(y, axis=-1)  # 杞崟閫氶亾

        if len(y) < self.n_fft:
            pad = self.n_fft - len(y)
            # 灏鹃儴琛ラ浂锛堜篃鍙敤 'reflect'锛屼絾闆跺～鍏呭湪宸ヤ笟鍣０涓嬫洿绋筹級
            y = np.pad(y, (0, pad), mode="constant", constant_values=0.0)

        try:
            import librosa
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=self.mel_bins, center=False, power=2.0
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            mel = torch.from_numpy(S_db).unsqueeze(0).float()  # [1, mel, Tspec]
            return mel
        except Exception as e:
            # 鏋佺鎯呭喌涓嬶紝缁欏嚭鍙缁冪殑闆跺紶閲忥紙骞堕檮鍔犳渶灏戝櫔澹帮紝閬垮厤 -inf锛?
            Tspec_min = max(1, (len(y) - self.n_fft) // max(1, self.hop_length) + 1)
            mel = torch.zeros(1, self.mel_bins, Tspec_min, dtype=torch.float32)
            return mel

    # -------- 闊抽鍔犺浇锛氳娈碘啋logMel鈫掗噸閲囨牱鍒?T*frames_per_slice 鈫?reshape --------
    def _load_audio_slices(self, path: str, start_s: Optional[float], end_s: Optional[float]) -> torch.Tensor:
        T = self.T_v
        mel_bins = self.mel_bins
        fps = self.frames_per_slice

        p = Path(path)
        if not p.exists():
            # 璺緞閿欒涓嶅啀鈥滈潤榛橀殢鏈衡€濓紝鐩存帴鎶ラ敊浠ヤ究鏃╁彂鐜?
            raise FileNotFoundError(f"Audio not found: {p}")

        # 鍏堝皾璇?torchaudio 璇诲彇锛堥€熷害蹇紝瑙ｇ爜绋冲畾锛?
        mel = None
        try:
            import torchaudio
            wav, s = torchaudio.load(str(p))  # [C, N]
            if s != self.sr:
                wav = torchaudio.functional.resample(wav, s, self.sr)
                s = self.sr
            mono = wav.mean(dim=0)  # [N]

            # 鍒囩墖锛堝崟浣嶏細绉掞級
            if start_s is not None and end_s is not None and end_s > start_s:
                s_i = int(max(0.0, start_s) * s)
                e_i = int(end_s * s)
                e_i = max(e_i, s_i + 1)
                mono = mono[s_i:e_i]

            # 闃叉闀垮害杩囩煭瀵艰嚧 STFT 鎶ヨ锛氳ˉ鍒?n_fft
            if mono.numel() < self.n_fft:
                pad = self.n_fft - mono.numel()
                mono = torch.nn.functional.pad(mono, (0, pad), value=0.0)

            # 鏋勯€犱笌 librosa 瀵归綈鐨?Mel 鍙傛暟
            mel_extractor = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sr, n_fft=self.n_fft, win_length=self.n_fft,
                hop_length=self.hop_length, n_mels=mel_bins, center=False,
                pad_mode=self.pad_mode, power=2.0
            )
            mel = mel_extractor(mono.unsqueeze(0))  # [1, mel, Tspec]
            mel = torch.log(mel + 1e-6)
        except Exception:
            # 鍥為€€鍒?librosa锛堝苟缁熶竴鍙傛暟锛?
            try:
                import librosa
                y, s = librosa.load(str(p), sr=self.sr, mono=True)
                if start_s is not None and end_s is not None and end_s > start_s:
                    s_i = int(max(0.0, start_s) * s)
                    e_i = int(end_s * s)
                    e_i = max(e_i, s_i + 1)
                    y = y[s_i:e_i]
                mel = self._compute_mel_from_numpy(y, s)  # [1, mel, Tspec]
            except Exception as e:
                # 鏈€鍚庣殑鍏滃簳锛氱粰鍑哄悎鐞嗗舰鐘剁殑闆跺紶閲忥紝閬垮厤璁粌涓柇
                mel = torch.zeros(1, mel_bins, 1, dtype=torch.float32)

        # 缁熶竴瀵归綈鍒?T * fps
        Tspec = int(mel.size(-1))
        target_T = int(T * fps)
        if Tspec == 0:
            mel = mel.new_zeros(1, mel_bins, target_T)
        elif Tspec >= target_T:
            idx = torch.linspace(0, Tspec - 1, steps=target_T).long()
            mel = mel.index_select(-1, idx)
        else:
            pad = target_T - Tspec
            mel = torch.cat([mel, mel.new_zeros(1, mel_bins, pad)], dim=-1)

        # [1, mel, T*fps] -> [T, mel, fps]
        mel = mel.view(1, mel_bins, T, fps).permute(2, 1, 3, 0).squeeze(-1).contiguous()
        return mel  # [T, mel_bins, fps]

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        v = self._load_video_frames(row["video_path"], row["video_start_frame"], row["video_end_frame"])
        a = self._load_audio_slices(row["audio_path"], row["audio_start_s"], row["audio_end_s"])

        if self.is_unlabeled:
            y = torch.tensor(-1, dtype=torch.long)
        else:
            y = torch.tensor(int(row["label_idx"]), dtype=torch.long)

        # 鏂板锛氳繑鍥炰竴涓ǔ瀹氱殑鈥滄牱鏈敮涓€ID鈥濄€傝繖閲岀洿鎺ョ敤绱㈠紩 idx锛坕nt锛夛紝涔熷彲浠ユ崲鎴?row["clip_id"] 鐨勫搱甯屻€?
        ids = torch.tensor(idx, dtype=torch.long)
        return v, a, y, ids

# 鏃х増锛堜繚鐣欏吋瀹癸級锛氬鏋滀綘鐨勮€佷唬鐮佸彧鎺ユ敹 3 鍏冪粍锛屽彲浠ョ户缁敤杩欎釜
def safe_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    # 鍏煎 3/4 鍏冪粍
    if len(batch) == 0:
        return None
    if len(batch[0]) == 3:
        v = torch.stack([b[0] for b in batch], dim=0)
        a = torch.stack([b[1] for b in batch], dim=0)
        y = torch.stack([b[2] for b in batch], dim=0).long()
        return v, a, y
    else:
        v = torch.stack([b[0] for b in batch], dim=0)
        a = torch.stack([b[1] for b in batch], dim=0)
        y = torch.stack([b[2] for b in batch], dim=0).long()
        ids = torch.stack([b[3] for b in batch], dim=0).long()
        return v, a, y, ids
# 鏂扮増锛堟帹鑽愮敤浜庢棤鏍囩 DataLoader锛夛細鏄庣‘杩斿洖 ids
# 鏂扮増锛堟帹鑽愮敤浜庢棤鏍囩 DataLoader锛夛細鏄庣‘杩斿洖 ids
def safe_collate_fn_with_ids(batch):
    batch = [b for b in batch if b is not None]
    v = torch.stack([b[0] for b in batch], dim=0)          # [B,T,3,H,W]
    a = torch.stack([b[1] for b in batch], dim=0)          # [B,T,mel,f]
    y = torch.stack([b[2] for b in batch], dim=0).long()   # [B]
    ids = torch.stack([b[3] for b in batch], dim=0).long() # [B]
    return v, a, y, ids


