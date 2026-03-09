# -*- coding: utf-8 -*-
"""
CAVA (Causal Audio-Visual Alignment)

Phase 2 refactor:
- LearnableDelay
- SoftTemporalShift
- DisplacementAwareCausalMask
- AlignmentGate
- CAVAModule (integration + compatibility outputs)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def _normalize_seq(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=-1, eps=EPS)


class LearnableDelay(nn.Module):
    """Map learnable theta to [delta_low, delta_high] with sigmoid."""

    def __init__(self, delta_low: float, delta_high: float, init_mid: bool = True):
        super().__init__()
        if float(delta_high) < float(delta_low):
            raise ValueError("delta_high must be >= delta_low")
        self.register_buffer("delta_low", torch.tensor(float(delta_low)))
        self.register_buffer("delta_high", torch.tensor(float(delta_high)))
        init = 0.0 if init_mid else -2.0
        self.theta = nn.Parameter(torch.tensor(init, dtype=torch.float32))

    def forward(self, batch_size: int) -> torch.Tensor:
        theta = torch.clamp(self.theta, -12.0, 12.0)
        delta = self.delta_low + (self.delta_high - self.delta_low) * torch.sigmoid(theta)
        return delta.expand(int(batch_size))


class SoftTemporalShift(nn.Module):
    """Right shift with linear interpolation for each sample."""

    def forward(self, audio_seq: torch.Tensor, delta_frames: torch.Tensor) -> torch.Tensor:
        if audio_seq.ndim != 3:
            raise ValueError(f"audio_seq must be [B,T,D], got {tuple(audio_seq.shape)}")
        b, t, d = audio_seq.shape
        if t <= 1:
            return audio_seq

        if delta_frames.ndim == 0:
            delta_frames = delta_frames.view(1).expand(b)
        delta = delta_frames.view(b, 1, 1).clamp(min=0.0, max=float(t - 1))

        n = torch.floor(delta)
        alpha = (delta - n).to(audio_seq.dtype)
        n = n.long()

        tidx = torch.arange(t, device=audio_seq.device).view(1, t, 1)
        idx0 = torch.clamp(tidx - n, 0, t - 1)
        idx1 = torch.clamp(idx0 + 1, 0, t - 1)
        a0 = torch.gather(audio_seq, 1, idx0.expand(b, t, d))
        a1 = torch.gather(audio_seq, 1, idx1.expand(b, t, d))
        return (1.0 - alpha) * a0 + alpha * a1


def soft_shift_right(audio_seq: torch.Tensor, delta_frames: torch.Tensor) -> torch.Tensor:
    """Compatibility wrapper used by older call sites."""
    return SoftTemporalShift()(audio_seq, delta_frames)


class DisplacementAwareCausalMask(nn.Module):
    """
    Build M_delta(t, tau) with causal constraint tau <= t.
    Hard mode: within |tau - (t-delta)| <= window_size.
    Gaussian mode: exp(-0.5 * ((tau - (t-delta))/window_size)^2), then causal-trim.
    """

    def __init__(
        self,
        window_size: int = 5,
        mask_type: str = "hard",
        multi_scale: bool = False,
    ):
        super().__init__()
        if int(window_size) < 1:
            raise ValueError("window_size must be >= 1")
        mask_type = str(mask_type).lower()
        if mask_type not in {"hard", "gaussian"}:
            raise ValueError("mask_type must be one of: hard, gaussian")
        self.window_size = int(window_size)
        self.mask_type = mask_type
        self.multi_scale = bool(multi_scale)

    def _single_scale(self, delta: torch.Tensor, tlen: int, w: int, device) -> torch.Tensor:
        b = delta.shape[0]
        t = torch.arange(tlen, device=device, dtype=torch.float32).view(1, tlen, 1)
        tau = torch.arange(tlen, device=device, dtype=torch.float32).view(1, 1, tlen)
        center = t + delta.view(b, 1, 1).float()
        center = torch.maximum(center, torch.zeros_like(center))
        center = torch.minimum(center, t)

        if self.mask_type == "hard":
            raw = (torch.abs(tau - center) <= float(w)).float()
        else:
            sigma = max(float(w), 1.0)
            raw = torch.exp(-0.5 * ((tau - center) / sigma) ** 2)

        causal = (tau <= t).float()
        raw = raw * causal
        denom = raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return raw / denom

    def forward(self, delta_frames: torch.Tensor, tlen: int) -> torch.Tensor:
        if int(tlen) <= 0:
            raise ValueError("tlen must be positive")
        if delta_frames.ndim == 0:
            delta_frames = delta_frames.view(1)

        if not self.multi_scale:
            return self._single_scale(delta_frames, int(tlen), self.window_size, delta_frames.device)

        windows = [self.window_size, self.window_size * 2, self.window_size * 4]
        ms = [self._single_scale(delta_frames, int(tlen), w, delta_frames.device) for w in windows]
        out = torch.stack(ms, dim=0).mean(dim=0)
        denom = out.sum(dim=-1, keepdim=True).clamp_min(EPS)
        return out / denom


class AlignmentGate(nn.Module):
    """Predict gate in configured range from aligned audio/video features."""

    def __init__(self, d_model: int, gate_min: float = 0.05, gate_max: float = 0.95):
        super().__init__()
        if float(gate_max) <= float(gate_min):
            raise ValueError("gate_max must be > gate_min")
        self.gate_min = float(gate_min)
        self.gate_max = float(gate_max)
        hidden = max(64, min(4 * int(d_model), 2048))
        self.net = nn.Sequential(
            nn.Linear(3 * int(d_model), hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )

    def forward(self, video_seq: torch.Tensor, audio_seq: torch.Tensor) -> torch.Tensor:
        v = _normalize_seq(video_seq)
        a = _normalize_seq(audio_seq)
        x = torch.cat([a, v, a * v], dim=-1)
        b, t, _ = x.shape
        logits = self.net(x.reshape(b * t, -1)).reshape(b, t, 1)
        g = torch.sigmoid(torch.clamp(logits, -12.0, 12.0))
        return g.clamp(min=self.gate_min, max=self.gate_max)


@dataclass
class CAVAConfigResolved:
    delta_low: float
    delta_high: float
    d_model: int
    window_size: int
    mask_type: str
    multi_scale: bool
    gate_min: float
    gate_max: float
    dist_max_delay: int


def _resolve_gate_range(cava_cfg: Dict) -> tuple[float, float]:
    # Explicit range has highest priority.
    if "gate_range" in cava_cfg and isinstance(cava_cfg["gate_range"], (list, tuple)) and len(cava_cfg["gate_range"]) == 2:
        return float(cava_cfg["gate_range"][0]), float(cava_cfg["gate_range"][1])

    mode = str(cava_cfg.get("gate_range_mode", "paper")).lower()
    if mode == "legacy":
        return 0.01, 0.99

    # paper baseline
    return float(cava_cfg.get("gate_min", 0.05)), float(cava_cfg.get("gate_max", 0.95))


class CAVAModule(nn.Module):
    def __init__(
        self,
        video_dim: int,
        audio_dim: int,
        d_model: int = 256,
        delta_low_frames: float = 2.0,
        delta_high_frames: float = 6.0,
        gate_clip_min: Optional[float] = None,
        gate_clip_max: Optional[float] = None,
        num_classes: Optional[int] = None,
        dist_max_delay: int = 6,
        window_size: int = 5,
        mask_type: str = "hard",
        multi_scale: bool = False,
        gate_range_mode: str = "paper",
        gate_range: Optional[list] = None,
    ):
        super().__init__()

        cava_cfg = {
            "gate_range_mode": gate_range_mode,
            "gate_range": gate_range,
            "gate_min": gate_clip_min if gate_clip_min is not None else 0.05,
            "gate_max": gate_clip_max if gate_clip_max is not None else 0.95,
        }
        gate_min, gate_max = _resolve_gate_range(cava_cfg)

        self.cfg = CAVAConfigResolved(
            delta_low=float(delta_low_frames),
            delta_high=float(delta_high_frames),
            d_model=int(d_model),
            window_size=int(window_size),
            mask_type=str(mask_type).lower(),
            multi_scale=bool(multi_scale),
            gate_min=float(gate_min),
            gate_max=float(gate_max),
            dist_max_delay=int(dist_max_delay),
        )
        self.d_model = self.cfg.d_model

        self.v_proj = nn.Linear(video_dim, self.cfg.d_model) if int(video_dim) != self.cfg.d_model else nn.Identity()
        self.a_proj = nn.Linear(audio_dim, self.cfg.d_model) if int(audio_dim) != self.cfg.d_model else nn.Identity()
        self.delay = LearnableDelay(self.cfg.delta_low, self.cfg.delta_high, init_mid=True)
        self.shift = SoftTemporalShift()
        self.mask = DisplacementAwareCausalMask(
            window_size=self.cfg.window_size,
            mask_type=self.cfg.mask_type,
            multi_scale=self.cfg.multi_scale,
        )
        self.gate = AlignmentGate(
            d_model=self.cfg.d_model,
            gate_min=self.cfg.gate_min,
            gate_max=self.cfg.gate_max,
        )
        self.class_delay = nn.Parameter(torch.zeros(num_classes)) if (num_classes is not None) else None
        self.dist_max_delay = self.cfg.dist_max_delay

        self.register_buffer("delta_low", torch.tensor(self.cfg.delta_low))
        self.register_buffer("delta_high", torch.tensor(self.cfg.delta_high))

    def _corr_scores(self, audio_seq: torch.Tensor, video_seq: torch.Tensor) -> torch.Tensor:
        if audio_seq.ndim == 2:
            audio_seq = audio_seq.unsqueeze(1)
        if video_seq.ndim == 2:
            video_seq = video_seq.unsqueeze(1)
        b, ta, _ = audio_seq.shape
        bv, tv, _ = video_seq.shape
        if b != bv:
            raise ValueError("audio/video batch mismatch")
        t = min(ta, tv)
        a = _normalize_seq(audio_seq[:, :t, :])
        v = _normalize_seq(video_seq[:, :t, :])

        md = int(self.dist_max_delay)
        scores = []
        for d in range(-md, md + 1):
            if d == 0:
                s = (a * v).sum(dim=-1).mean(dim=1)
            elif d > 0:
                s = (a[:, :-d, :] * v[:, d:, :]).sum(dim=-1).mean(dim=1) if d < t else a.new_zeros((b,))
            else:
                dd = -d
                s = (a[:, dd:, :] * v[:, :-dd, :]).sum(dim=-1).mean(dim=1) if dd < t else a.new_zeros((b,))
            scores.append(s)
        return torch.stack(scores, dim=1)

    def get_predicted_delay(self, audio_seq: torch.Tensor, video_seq: torch.Tensor) -> torch.Tensor:
        scores = self._corr_scores(audio_seq, video_seq)
        prob = F.softmax(scores, dim=1)
        md = int(self.dist_max_delay)
        return prob.argmax(dim=1) - md

    def forward(self, video_seq: torch.Tensor, audio_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        if video_seq.ndim != 3 or audio_seq.ndim != 3:
            raise ValueError("CAVAModule expects [B,T,D] inputs")
        b, t, _ = video_seq.shape
        if t != audio_seq.shape[1]:
            t = min(t, audio_seq.shape[1])
            video_seq = video_seq[:, :t, :]
            audio_seq = audio_seq[:, :t, :]

        v = F.layer_norm(self.v_proj(video_seq.float()), [self.cfg.d_model])
        a = F.layer_norm(self.a_proj(audio_seq.float()), [self.cfg.d_model])

        delta = self.delay(b)  # [B]
        a_shift = self.shift(a, delta)  # [B,T,D]
        m_delta = self.mask(delta, t)  # [B,T,T]
        a_masked = torch.bmm(m_delta, a_shift)  # [B,T,D]
        gate = self.gate(v, a_masked)  # [B,T,1]

        # Blend aligned context and shifted sequence for stability.
        audio_for_fusion = gate * a_masked + (1.0 - gate) * a_shift

        scores = self._corr_scores(a, v)
        prob = F.softmax(scores, dim=1)
        md = int(self.dist_max_delay)
        pred_delay = prob.argmax(dim=1) - md

        return {
            "audio_for_fusion": audio_for_fusion,
            "audio_aligned": audio_for_fusion,
            "audio_proj": a,
            "video_proj": v,
            "audio_seq": a,
            "causal_gate": gate,
            "causal_mask": m_delta,
            "delay_frames": delta,
            "delay_frames_cont": delta,
            "delta_low": float(self.delta_low.item()),
            "delta_high": float(self.delta_high.item()),
            "causal_prob": gate.squeeze(-1),
            "causal_prob_dist": prob,
            "pred_delay": pred_delay,
        }
