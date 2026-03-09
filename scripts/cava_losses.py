# -*- coding: utf-8 -*-
"""
Unified CAVA loss stack (Phase 2).
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-8


def _flatten_bt(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        b, t, d = x.shape
        return x.reshape(b * t, d)
    if x.ndim == 2:
        return x
    raise ValueError(f"expect [B,T,D] or [N,D], got {tuple(x.shape)}")


def _mask_to_weights(mask: Optional[torch.Tensor], n: int, device, dtype=torch.float32) -> torch.Tensor:
    if mask is None:
        return torch.ones(n, device=device, dtype=dtype)
    m = mask
    if m.ndim == 3 and m.size(-1) == 1:
        m = m.squeeze(-1)
    if m.ndim == 2:
        m = m.reshape(-1)
    if m.ndim != 1:
        raise ValueError(f"mask must be [B,T], [B,T,1] or [N], got {tuple(mask.shape)}")
    m = m.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    return m


def _apply_temporal_exclusion(logits: torch.Tensor, radius: int) -> torch.Tensor:
    # logits: [T,T], keep diagonal positives, mask neighbor negatives.
    if int(radius) <= 0:
        return logits
    t = logits.size(0)
    idx = torch.arange(t, device=logits.device)
    dist = torch.abs(idx.view(-1, 1) - idx.view(1, -1))
    neighbor = (dist <= int(radius)) & (~torch.eye(t, device=logits.device, dtype=torch.bool))
    return logits.masked_fill(neighbor, -1e4)


def info_nce_align(
    a: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    tau: Optional[float] = 0.07,
    temperature: Optional[float] = None,
    normalize: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    if tau is None and temperature is None:
        tau = 0.07
    if tau is None and temperature is not None:
        tau = float(temperature)
    tau = float(tau)

    a = _flatten_bt(a).float()
    v = _flatten_bt(v).float()
    if normalize:
        a = F.normalize(a, dim=-1, eps=EPS)
        v = F.normalize(v, dim=-1, eps=EPS)

    n = a.shape[0]
    if v.shape[0] != n:
        raise ValueError(f"mismatched flattened length: {n} vs {v.shape[0]}")

    logits = (a @ v.t()) / max(tau, EPS)
    logits = logits.clamp(-60.0, 60.0)
    target = torch.arange(n, device=logits.device)
    loss_i = F.cross_entropy(logits, target, reduction="none")
    w = _mask_to_weights(mask, n, logits.device, logits.dtype)

    if reduction == "none":
        return loss_i * w
    if reduction == "sum":
        return (loss_i * w).sum()
    return (loss_i * w).sum() / w.sum().clamp_min(EPS)


def corr_diag_align(
    a: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    a = _flatten_bt(a).float()
    v = _flatten_bt(v).float()
    a = a - a.mean(dim=0, keepdim=True)
    v = v - v.mean(dim=0, keepdim=True)
    a = F.normalize(a, dim=-1, eps=EPS)
    v = F.normalize(v, dim=-1, eps=EPS)

    corr = a @ v.t()
    loss_i = 1.0 - torch.diag(corr)
    w = _mask_to_weights(mask, loss_i.numel(), loss_i.device, loss_i.dtype)

    if reduction == "none":
        return loss_i * w
    if reduction == "sum":
        return (loss_i * w).sum()
    return (loss_i * w).sum() / w.sum().clamp_min(EPS)


def prior_l2(delta: torch.Tensor, mu: Optional[float], sigma: Optional[float]) -> torch.Tensor:
    if (mu is None) or (sigma is None) or (float(sigma) <= 0):
        return delta.new_zeros(())
    z = (delta.float() - float(mu)) / float(sigma)
    return (z * z).mean()


def edge_hinge(
    delta: torch.Tensor,
    low: float,
    high: float,
    margin_ratio: float = 0.25,
) -> torch.Tensor:
    if float(high) < float(low):
        raise ValueError("edge_hinge: high must >= low")
    l = float(low)
    h = float(high)
    m = float(margin_ratio) * (h - l)
    d = delta.float()
    left = F.relu((l + m) - d) / (m + EPS)
    right = F.relu(d - (h - m)) / (m + EPS)
    return (left + right).mean()


class CAVALoss(nn.Module):
    """
    total = align + beta_edge * R_edge + beta_prior * R_prior + beta_gate * R_gate

    Where align is weighted alignment term:
      align = beta_align * R_align
    """

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__()
        self.update_cfg(cfg or {})

    def update_cfg(self, cfg: Dict):
        c = dict(cfg or {})
        self.beta_align = float(c.get("lambda_cava", c.get("lambda_align", c.get("beta_align", 0.0))))
        self.beta_edge = float(c.get("lambda_edge", c.get("beta_edge", 0.0)))
        self.beta_prior = float(c.get("lambda_prior", c.get("beta_prior", 0.0)))
        self.beta_gate = float(c.get("lambda_gate", c.get("beta_gate", 0.0)))

        self.tau = float(c.get("tau_nce", c.get("tau", 0.2)))
        self.edge_margin_ratio = float(c.get("edge_margin_ratio", 0.25))
        self.prior_mu = c.get("prior_mu", c.get("delta_prior", None))
        self.prior_sigma = c.get("prior_sigma", 1.0 if self.prior_mu is not None else None)
        self.gate_target = float(c.get("gate_target", 1.0))
        self.gate_loss_type = str(c.get("gate_loss_type", "mse")).lower()

        self.negative_mode = str(c.get("negative_mode", "batch_global")).lower()
        self.temporal_exclusion_radius = int(c.get("temporal_exclusion_radius", 0))
        self._cfg = c

    def _resolve_delta_range(self, outputs: Dict) -> tuple[float, float]:
        low = outputs.get("delta_low", self._cfg.get("delta_low_frames", self._cfg.get("delta_low", -1.0)))
        high = outputs.get("delta_high", self._cfg.get("delta_high_frames", self._cfg.get("delta_high", 1.0)))
        return float(low), float(high)

    def _align_loss(self, a: torch.Tensor, v: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
        mode = self.negative_mode
        if mode == "batch_global":
            return info_nce_align(a, v, mask=gate, tau=self.tau)

        if a.ndim != 3 or v.ndim != 3:
            raise ValueError(f"{mode} requires [B,T,D] tensors")
        if a.shape != v.shape:
            raise ValueError("a and v must have same shape for intra-sequence modes")

        b, t, _ = a.shape
        an = F.normalize(a.float(), dim=-1, eps=EPS)
        vn = F.normalize(v.float(), dim=-1, eps=EPS)
        target = torch.arange(t, device=a.device)

        if gate is None:
            weight = a.new_ones((b, t))
        else:
            weight = gate.squeeze(-1) if gate.ndim == 3 else gate
            weight = weight.float().clamp(0.0, 1.0)

        loss_sum = a.new_zeros(())
        w_sum = a.new_zeros(())
        for bi in range(b):
            logits = (an[bi] @ vn[bi].transpose(0, 1)) / max(self.tau, EPS)
            logits = logits.clamp(-60.0, 60.0)
            if mode == "intra_sequence_exclude_neighbors":
                logits = _apply_temporal_exclusion(logits, self.temporal_exclusion_radius)
            elif mode != "intra_sequence_all":
                raise ValueError(f"unsupported negative_mode: {mode}")
            ce = F.cross_entropy(logits, target, reduction="none")
            w = weight[bi]
            loss_sum = loss_sum + (ce * w).sum()
            w_sum = w_sum + w.sum()
        return loss_sum / w_sum.clamp_min(EPS)

    def forward(self, outputs: Dict) -> Dict[str, torch.Tensor]:
        device = None
        if "clip_logits" in outputs and isinstance(outputs["clip_logits"], torch.Tensor):
            device = outputs["clip_logits"].device

        zero = torch.tensor(0.0, device=device) if device is not None else torch.tensor(0.0)
        a_aln = outputs.get("audio_aligned", outputs.get("audio_seq"))
        v_prj = outputs.get("video_proj", outputs.get("video_seq"))
        gate = outputs.get("causal_gate", None)
        delta = outputs.get("delay_frames_cont", outputs.get("delay_frames", None))

        r_align = zero
        if (a_aln is not None) and (v_prj is not None):
            r_align = self._align_loss(a_aln, v_prj, gate)
        loss_align = self.beta_align * r_align

        r_edge = zero
        if delta is not None:
            low, high = self._resolve_delta_range(outputs)
            r_edge = edge_hinge(delta, low, high, margin_ratio=self.edge_margin_ratio)
        loss_edge = self.beta_edge * r_edge

        r_prior = zero
        if delta is not None and self.beta_prior > 0.0:
            r_prior = prior_l2(delta, self.prior_mu, self.prior_sigma)
        loss_prior = self.beta_prior * r_prior

        r_gate = zero
        if gate is not None and self.beta_gate > 0.0:
            g = gate.float()
            if self.gate_loss_type == "l1":
                r_gate = (g - self.gate_target).abs().mean()
            else:
                r_gate = ((g - self.gate_target) ** 2).mean()
        loss_gate = self.beta_gate * r_gate

        loss_total = loss_align + loss_edge + loss_prior + loss_gate
        return {
            "loss_total": loss_total,
            "loss_align": loss_align,
            "loss_edge": loss_edge,
            "loss_prior": loss_prior,
            "loss_gate": loss_gate,
        }


def compute_cava_losses(outputs: Dict, cfg: Dict) -> Dict[str, torch.Tensor]:
    # Compatibility helper with previous function name.
    mod = CAVALoss(cfg)
    out = mod(outputs)
    return {
        "align": out["loss_align"],
        "edge": out["loss_edge"],
        "prior": out["loss_prior"],
        "gate": out["loss_gate"],
        "total": out["loss_total"],
    }


def causal_supervised_loss(audio_proj: torch.Tensor, video_proj: torch.Tensor, class_labels: torch.Tensor, cava_module, weight: float = 1.0) -> torch.Tensor:
    if cava_module is None or getattr(cava_module, "class_delay", None) is None:
        return audio_proj.new_zeros(())
    scores = cava_module._corr_scores(audio_proj, video_proj)
    prob = F.softmax(scores, dim=1)
    md = int(getattr(cava_module, "dist_max_delay", (prob.size(1) - 1) // 2))
    offsets = torch.arange(-md, md + 1, device=prob.device, dtype=prob.dtype)
    exp_dt = (prob * offsets.unsqueeze(0)).sum(1)
    dt_c = cava_module.class_delay[class_labels.to(cava_module.class_delay.device)]
    return float(weight) * F.mse_loss(exp_dt, dt_c)


def causal_self_supervised_loss(audio_proj: torch.Tensor, video_proj: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    return info_nce_align(audio_proj, video_proj, mask=None, tau=float(temperature))
