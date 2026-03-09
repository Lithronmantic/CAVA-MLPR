# -*- coding: utf-8 -*-
# src/avtop/fusion/alignment.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from src.avtop.utils.logging import get_logger

log = get_logger(__name__)

def ensure_btd(x: torch.Tensor) -> torch.Tensor:
    """统一成 (B,T,D)。含 DEBUG 输出形状变化。"""
    if x.dim() == 3:  # (B,T,D) or (B,D,T)
        out = x if x.shape[1] <= x.shape[2] else x.transpose(1, 2).contiguous()
    elif x.dim() == 4:  # (B,C,T,D')
        B,C,T,Dp = x.shape
        out = x.permute(0,2,1,3).contiguous().view(B,T,C*Dp)
    elif x.dim() == 5:  # (B,C,T,H,W)
        B,C,T,H,W = x.shape
        out = x.permute(0,2,1,3,4).contiguous().view(B,T,C*H*W)
    else:
        raise ValueError(f"[ensure_btd] unsupported shape: {x.shape}")
    if log.level <= 10:  # DEBUG
        log.debug(f"[ensure_btd] {tuple(x.shape)} -> {tuple(out.shape)}")
    return out

def _build_grid(shift: torch.Tensor, T: int) -> torch.Tensor:
    # shift: (B,1) in [-1,1]，生成 (B,T,1,2) 的采样网格（用于 1D grid_sample）
    b = shift.shape[0]
    base = torch.linspace(-1, 1, steps=T, device=shift.device, dtype=shift.dtype)
    base = base.view(1, T, 1).repeat(b, 1, 1)  # (B,T,1)
    grid_x = base + shift.view(b, 1, 1)        # 时间轴偏移
    grid_y = torch.zeros_like(grid_x)          # 伪 2D 的第二维
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,T,1,2)
    return grid.clamp(-1, 1)

class ContinuousShift1D(nn.Module):
    """
    连续偏移（可微）对齐：预测 Δt∈[-1,1]，用 grid_sample 对一条模态重采样到另一条模态。
    target: "video" 表示将音频对齐到视频（即重采样音频）；"audio" 反之。
    """
    def __init__(self, d_model: int, target: str = "video"):
        super().__init__()
        assert target in ("video", "audio")
        self.target = target
        self.predict = nn.Sequential(
            nn.Linear(d_model*2, d_model), nn.ReLU(),
            nn.Linear(d_model, 1), nn.Tanh()  # 输出 [-1,1]
        )
        log.info(f"[Align] ContinuousShift1D initialized (target={target})")

    def _shift_1d(self, x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        B,T,D = x.shape
        x_ = x.transpose(1,2).unsqueeze(-1)   # (B,D,T,1)
        grid = _build_grid(shift, T)          # (B,T,1,2)
        y = F.grid_sample(x_, grid, mode='bilinear',
                          padding_mode='border', align_corners=True).squeeze(-1)  # (B,D,T)
        return y.transpose(1,2).contiguous()  # (B,T,D)

    def predict_shift(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        v = ensure_btd(v); a = ensure_btd(a)
        v_g, a_g = v.mean(1), a.mean(1)
        shift = self.predict(torch.cat([v_g, a_g], dim=-1))  # (B,1)
        if log.level <= 10:
            log.debug(f"[Align] predict_shift: min={shift.min().item():.4f}, "
                      f"max={shift.max().item():.4f}, mean={shift.mean().item():.4f}")
        return shift

    def forward_with_shift(self, v: torch.Tensor, a: torch.Tensor, shift: torch.Tensor):
        v = ensure_btd(v); a = ensure_btd(a)
        if self.target == "video":
            return v, self._shift_1d(a, shift)
        else:
            return self._shift_1d(v, shift), a

    def forward(self, v: torch.Tensor, a: torch.Tensor):
        shift = self.predict_shift(v, a)
        return self.forward_with_shift(v, a, shift)

class SoftDTWAlignment(nn.Module):
    """Soft-DTW 对齐（简化、可微），适合短序列。"""
    def __init__(self, gamma: float = 0.1, target: str = "video"):
        super().__init__()
        assert target in ("video","audio")
        self.gamma = gamma
        self.target = target
        log.info(f"[Align] SoftDTW initialized (gamma={gamma}, target={target})")

    def forward(self, v: torch.Tensor, a: torch.Tensor):
        v = ensure_btd(v); a = ensure_btd(a)
        v_n, a_n = F.normalize(v, dim=-1), F.normalize(a, dim=-1)
        sim = torch.einsum('btd,bsd->bts', v_n, a_n)   # (B,Tv,Ta)
        Av = F.softmax(sim / self.gamma, dim=2)        # v->a
        Aa = F.softmax(sim.transpose(1,2) / self.gamma, dim=2)  # a->v
        if self.target == "video":
            a_al = torch.bmm(Av, a)
            return v, a_al
        else:
            v_al = torch.bmm(Aa, v)
            return v_al, a

class SSLAlignmentCache:
    """半监督对齐缓存（仅 shift 模式需要）。"""
    def __init__(self):
        self._shift: Optional[torch.Tensor] = None
        log.info("[Align] SSLAlignmentCache initialized.")

    def store(self, shift: torch.Tensor):
        self._shift = shift.detach()
        log.debug(f"[Align] cache.store shift: shape={tuple(self._shift.shape)}")

    def pop(self) -> Optional[torch.Tensor]:
        s = self._shift
        self._shift = None
        log.debug(f"[Align] cache.pop -> {'None' if s is None else tuple(s.shape)}")
        return s
