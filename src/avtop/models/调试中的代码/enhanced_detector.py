# -*- coding: utf-8 -*-
# src/avtop/models/enhanced_detector.py
import os, sys, inspect, warnings
from typing import Dict, Optional, Sequence, Tuple, List, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.avtop.utils.logging import get_logger

log = get_logger(__name__)

# 依赖
from src.avtop.fusion.alignment import (
    ContinuousShift1D, SoftDTWAlignment, SSLAlignmentCache, ensure_btd as align_btd
)
from src.avtop.fusion.early_fusion import BackboneWrapper, EarlyFusion, ensure_btd as early_btd

# 如果你有自定义的 CoAttention / Backbones，可在此导入；下面有兜底实现
try:
    from src.avtop.fusion.coattention import EnhancedCoAttention
except Exception as e:
    EnhancedCoAttention = None
    warnings.warn(f"[Model] EnhancedCoAttention not found: {e}. Will fallback to Concat.")

# -------------------- 兜底 backbone / 融合 / 头 --------------------
class SimpleVideoCNN(nn.Module):
    def __init__(self, in_channels=3, out_dim=512):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),        nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor):
        """
        接受 (B, C, T, H, W) 或 (B*T, C, H, W)，返回 (B,T,out_dim) 或 (N,out_dim)
        """
        if x.ndim == 5:
            B, C, T, H, W = x.shape
            if C not in (1, 3) and T in (1, 3):
                x = x.permute(0, 2, 1, 3, 4).contiguous()
                B, C, T, H, W = x.shape

            # (B, C, T, H, W) -> (B*T, C, H, W)
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)

            feat = self.backbone(x)  # (B*T, 128, 1, 1)
            feat = feat.flatten(1)  # (B*T, 128)
            feat = self.fc(feat)  # (B*T, out_dim = video_dim)
            D = feat.size(1)
            feat = feat.view(B, T, D)  # (B, T, video_dim)
            # 可留一轮调试
            # print(f"[SimpleVideoCNN] return feat: {feat.shape}")
            return feat

        elif x.ndim == 4:
            feat = self.backbone(x).flatten(1)  # (N, 128)
            feat = self.fc(feat)  # (N, out_dim)
            return feat

        else:
            raise RuntimeError(f"[SimpleVideoCNN] bad input: {tuple(x.shape)}")


class SimpleAudioCNN(nn.Module):
    def __init__(self, out_dim=256, in_channels=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=1), nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),         nn.ReLU(True), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, 3, padding=1),        nn.ReLU(True),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(256, out_dim)

    def _fix(self, x):
        if x.dim() == 2:  # [B,T]
            x = x.unsqueeze(1)
        elif x.dim() == 3:  # [B,C,T] or [B,T,D]
            if x.shape[1] != 1 and x.shape[-1] != 1:
                x = x.mean(dim=-1, keepdim=False).unsqueeze(1)
        elif x.dim() == 4:  # [B,S,1,L]
            B,S,one,L = x.shape
            if one == 1: x = x.view(B,1,S*L)
            else:        x = x.view(B, x.shape[1], -1)
        else:
            raise RuntimeError(f"[SimpleAudioCNN] bad input: {x.shape}")
        return x

    def forward(self, x):
        x = self._fix(x)                   # -> [B,1,T]
        x = self.conv(x).squeeze(-1)       # [B,256]
        x = self.fc(x).unsqueeze(1)        # [B,1,D]
        return x

class SimpleConcatFusion(nn.Module):
    def __init__(self, video_dim, audio_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(video_dim+audio_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )
    def forward(self, v, a):
        Tv, Ta = v.shape[1], a.shape[1]
        if Tv != Ta:
            a = F.interpolate(a.transpose(1,2), size=Tv, mode='linear',
                              align_corners=False).transpose(1,2)
        return self.proj(torch.cat([v,a], dim=-1))

class SimpleMILHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(in_dim,128), nn.Tanh(), nn.Linear(128,1))
        self.cls  = nn.Linear(in_dim, num_classes)
    def forward(self, x):
        w = self.attn(x).squeeze(-1)     # [B,T]
        w = torch.softmax(w, dim=-1)
        bag = torch.sum(x * w.unsqueeze(-1), dim=1)
        clip_logits = self.cls(bag)
        seg_logits  = self.cls(x)
        return {"clip_logits": clip_logits, "seg_logits": seg_logits, "weights": w}

# -------------------- 主模型 --------------------
class EnhancedAVDetector(nn.Module):
    """
    - Early-Fusion: 分段门控残差（可选）
    - Alignment: shift / softdtw / none
    - Fusion: EnhancedCoAttention（如可用）或回退到 concat
    - Head: MIL + 可选视频/音频辅助头
    """
    def __init__(self, config: Dict):
        super().__init__()
        self.cfg = config
        m = config.get("model", {})
        f = config.get("fusion", {})

        self.video_dim = int(m.get("video_dim", 512))
        self.audio_dim = int(m.get("audio_dim", 256))
        self.v_autoproj = nn.Identity()
        self.a_autoproj = nn.Identity()
        self.num_classes = int(m.get("num_classes", 2))
        log.info(f"[Model] dims: video={self.video_dim}, audio={self.audio_dim}, num_classes={self.num_classes}")

        # Backbones（你可替换成自家封装；此处提供兜底）
        self.video_backbone = SimpleVideoCNN(out_dim=self.video_dim)
        self.audio_backbone = SimpleAudioCNN(out_dim=self.audio_dim)

        # Early-Fusion
        ef = m.get("early_fusion", {})
        self.enable_early = bool(ef.get("enable", False))
        if self.enable_early:
            self.v_wrap = BackboneWrapper(self.video_backbone, ef.get("layer_refs_v", []) or [])
            self.a_wrap = BackboneWrapper(self.audio_backbone, ef.get("layer_refs_a", []) or [])
            early_dim = int(ef.get("dim", min(self.video_dim, self.audio_dim)))
            early_hidden = int(ef.get("hidden", max(64, early_dim // 2)))
            early_stages = list(ef.get("stages", [1]))
            match_time = self.cfg.get("model", {}).get("alignment", {}).get("target", "video")
            self.early = EarlyFusion(
                dim=early_dim, hidden=early_hidden, stages=early_stages,
                in_dim_v=self.video_dim, in_dim_a=self.audio_dim, match_time=match_time
            )
            log.info(f"[Model] Early-Fusion enabled: dim={early_dim}, hidden={early_hidden}, "
                     f"stages={early_stages}, proj_v={self.video_dim}->{early_dim}, "
                     f"proj_a={self.audio_dim}->{early_dim}, match_time={match_time}")

        # Alignment
        al = m.get("alignment", {})
        self.align_mode = al.get("mode", "none") if al.get("enable", False) else "none"
        align_target = al.get("target", "video")
        if self.align_mode == "shift":
            d_model = int(f.get("d_model", min(self.video_dim, self.audio_dim)))
            self.aligner = ContinuousShift1D(d_model=d_model, target=align_target)
            self.ssl_align_cache = SSLAlignmentCache()
        elif self.align_mode == "softdtw":
            self.aligner = SoftDTWAlignment(gamma=0.1, target=align_target)
            self.ssl_align_cache = None
        else:
            self.aligner = None
            self.ssl_align_cache = None
        log.info(f"[Model] Alignment mode={self.align_mode}, target={align_target}")

        # ------------------ Fusion ------------------
        ftype = str(f.get("type", "concat")).lower()
        d_model = int(f.get("d_model", min(self.video_dim, self.audio_dim)))

        # 关键：根据是否启用 Early-Fusion 决定传入融合层的输入维度
        if self.enable_early:
            fusion_in_v = int(m.get("early_fusion", {}).get("dim", min(self.video_dim, self.audio_dim)))
            fusion_in_a = fusion_in_v  # 早融合后两个分支都被映射到同一 early_dim
        else:
            fusion_in_v = self.video_dim
            fusion_in_a = self.audio_dim

        log.info(f"[Model] Fusion inputs: v_dim={fusion_in_v}, a_dim={fusion_in_a}, out_d_model={d_model}")

        if ftype in ("coattention", "coattn") and EnhancedCoAttention is not None:
            self.fusion = EnhancedCoAttention(
                video_dim=fusion_in_v, audio_dim=fusion_in_a,
                d_model=d_model,
                bottleneck_dim=f.get("bottleneck_dim", 128),
                num_layers=f.get("num_layers", 2),
                num_heads=f.get("num_heads", 8),
                dropout=f.get("dropout", 0.1)
            )
            self.fusion_type = "coattention"
        else:
            if EnhancedCoAttention is None and ftype in ("coattention", "coattn"):
                warnings.warn("[Model] CoAttention not found. Fallback to Concat.")
            # 这里用“实际输入维”构建 Concat 线性层
            self.fusion = SimpleConcatFusion(fusion_in_v, fusion_in_a, d_model)
            self.fusion_type = "concat"

        log.info(f"[Model] Fusion={self.fusion_type}, d_model={d_model}")

        # ------------------ Heads ------------------
        self.mil_head = SimpleMILHead(d_model, self.num_classes)

        # 关键：根据是否启用 Early-Fusion，确定“辅助头”输入维度
        aux_in_v = int(m.get("early_fusion", {}).get("dim", min(self.video_dim,
                                                                self.audio_dim))) if self.enable_early else self.video_dim
        aux_in_a = aux_in_v if self.enable_early else self.audio_dim

        if self.cfg.get("use_aux_heads", True):
            self.video_aux_head = nn.Sequential(
                nn.Linear(aux_in_v, 128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, self.num_classes)
            )
            self.audio_aux_head = nn.Sequential(
                nn.Linear(aux_in_a, 128), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(128, self.num_classes)
            )
            log.info(f"[Model] Aux heads: video_in={aux_in_v}, audio_in={aux_in_a}")
        else:
            self.video_aux_head = None
            self.audio_aux_head = None

        n_params = sum(p.numel() for p in self.parameters())
        log.info(f"[Model] Total params: {n_params:,}")

    # -------- feature extraction --------
    def _extract_features(self, video: torch.Tensor, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enable_early:
            v = align_btd(self.video_backbone(video))
            a = align_btd(self.audio_backbone(audio))
            if log.level <= 10:
                log.debug(f"[Model] feats(no-early): v{tuple(v.shape)} a{tuple(a.shape)}")
            return v, a
        # Early-Fusion：分段采集 -> 指定 stage 融合 -> 取最后段
        v_st = self.v_wrap(video)  # List[(B,T,D)]
        a_st = self.a_wrap(audio)
        v_st, a_st = self.early(v_st, a_st)
        v, a = v_st[-1], a_st[-1]
        if log.level <= 10:
            log.debug(f"[Model] feats(early): v{tuple(v.shape)} a{tuple(a.shape)}")
        return v, a

    # -------- forward --------
    def forward(self, video, audio, return_aux=True, return_attn=False):
        v, a = self._extract_features(video, audio)
        v = self.v_autoproj(v)  # (B,T,video_dim)
        a = self.a_autoproj(a)  # (B,T,audio_dim)
        if self.aligner is not None:
            if self.align_mode == "shift" and self.ssl_align_cache is not None:
                v, a = self.aligner(v, a)  # 常规前向：直接预测并应用
            else:
                v, a = self.aligner(v, a)
            if log.level <= 10:
                log.debug(f"[Model] after align: v{tuple(v.shape)} a{tuple(a.shape)}")

        if self.fusion_type == "coattention":
            if return_attn:
                fused, z_v, z_a, attn_info = self.fusion(v, a, return_attn=True)
            else:
                fused, z_v, z_a = self.fusion(v, a)
                attn_info = None
        else:
            fused = self.fusion(v, a)
            z_v, z_a = v.mean(1), a.mean(1)
            attn_info = None
        if log.level <= 10:
            log.debug(f"[Model] fused: {tuple(fused.shape)} z_v:{tuple(z_v.shape)} z_a:{tuple(z_a.shape)}")

        mil = self.mil_head(fused)
        out = {
            "clip_logits": mil["clip_logits"],
            "seg_logits":  mil["seg_logits"],
            "mil_weights": mil["weights"],
            "fused": fused, "video_feat": v, "audio_feat": a
        }
        if return_aux:
            out["z_v"] = z_v; out["z_a"] = z_a
            if self.video_aux_head is not None:
                out["video_logits"] = self.video_aux_head(v.mean(1))
            if self.audio_aux_head is not None:
                out["audio_logits"] = self.audio_aux_head(a.mean(1))
        if return_attn and attn_info is not None:
            out["attn_weights"] = attn_info
        return out
    # -------- 从特征层继续前向（跳过 backbone/对齐）--------
    def forward_from_features(self, v_feat: torch.Tensor, a_feat: torch.Tensor,
                              return_aux: bool = True, return_attn: bool = False):
        """
        参数:
          v_feat, a_feat: (B,T,D) 的特征（通常来自 align_[predict/use]_cached）
        行为:
          不再做特征提取与对齐，直接进入融合 + MIL + 辅助头。
        返回: 与 forward 一致的字典键
        """
        v, a = v_feat, a_feat  # 已对齐的 (B,T,D)
        # 融合
        if self.fusion_type == "coattention":
            if return_attn:
                fused, z_v, z_a, attn_info = self.fusion(v, a, return_attn=True)
            else:
                fused, z_v, z_a = self.fusion(v, a)
                attn_info = None
        else:
            fused = self.fusion(v, a)
            z_v, z_a = v.mean(1), a.mean(1)
            attn_info = None

        mil = self.mil_head(fused)
        out = {
            "clip_logits": mil["clip_logits"],
            "seg_logits":  mil["seg_logits"],
            "mil_weights": mil["weights"],
            "fused": fused, "video_feat": v, "audio_feat": a
        }
        if return_aux:
            out["z_v"] = z_v; out["z_a"] = z_a
            if self.video_aux_head is not None:
                out["video_logits"] = self.video_aux_head(v.mean(1))
            if self.audio_aux_head is not None:
                out["audio_logits"] = self.audio_aux_head(a.mean(1))
        if return_attn and attn_info is not None:
            out["attn_weights"] = attn_info
        return out


    # -------- SSL 对齐共享 --------
    # 1) 让 align_predict_and_cache 可返回 shift
    @torch.no_grad()
    def align_predict_and_cache(self, video: torch.Tensor, audio: torch.Tensor, return_shift: bool = False):
        """
        弱增强路径：预测并缓存 shift（仅 shift 模式有效）
        return_shift=True 时额外返回 shift（B,1）
        """
        assert self.align_mode == "shift" and self.ssl_align_cache is not None, "align_predict_and_cache 仅在 shift 模式可用"
        v, a = self._extract_features(video, audio)
        shift = self.aligner.predict_shift(v, a)  # (B,1)
        self.ssl_align_cache.store(shift)
        v_al, a_al = self.aligner.forward_with_shift(v, a, shift)
        return (v_al, a_al, shift) if return_shift else (v_al, a_al)

    # 2) 给 Student 提供一个注入 shift 的入口（跨模型转移）
    @torch.no_grad()
    def set_cached_shift(self, shift: torch.Tensor):
        """
        将外部提供的 shift 写入本模型缓存（用于把 Teacher 预测的 shift 传给 Student）
        """
        assert self.align_mode == "shift" and self.ssl_align_cache is not None, "set_cached_shift 仅在 shift 模式可用"
        dev = next(self.parameters()).device
        self.ssl_align_cache.store(shift.to(dev))

    # 3) 保留 align_use_cached 不变（用之前缓存的 shift 对 strong 增强做对齐）
    @torch.no_grad()
    def align_use_cached(self, video: torch.Tensor, audio: torch.Tensor):
        assert self.align_mode == "shift" and self.ssl_align_cache is not None, "align_use_cached 仅在 shift 模式可用"
        shift = self.ssl_align_cache.pop()
        assert shift is not None, "SSL align cache empty. 先调用 align_predict_and_cache() 或 set_cached_shift()"
        v, a = self._extract_features(video, audio)
        return self.aligner.forward_with_shift(v, a, shift)
