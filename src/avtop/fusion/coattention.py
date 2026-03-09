# -*- coding: utf-8 -*-
# src/avtop/fusion/coattention.py
"""
EnhancedCoAttention（稳健版）
- 输入：v:(B,Tv,Dv)、a:(B,Ta,Da)，自动统一到 d_model，并按 match_time 对齐时间长度
- 多层双向 co-attention，使用 Learnable Queries 做“注意力瓶颈”（更稳定，无固定长度假设）
- 输出：
  fused:(B,T,d_model)  （融合后的时序特征）
  z_v:(B,d_model), z_a:(B,d_model) （模态全局向量，供对比学习/辅助头）
  可选 attn_info（return_attn=True）记录每层 v<-a / a<-v 的注意力权重摘要
"""

from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # 统一项目内日志
    from src.avtop.utils.logging import get_logger
except Exception:
    # 兜底：无项目日志工具时用标准 logging
    import logging
    def get_logger(name):
        logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        return logging.getLogger(name)

log = get_logger(__name__)

# ----------------------------- 基础工具 -----------------------------
def _ensure_btd(x: torch.Tensor) -> torch.Tensor:
    """把常见张量统一成 (B,T,D)。"""
    if x.dim()==3:
        # 可能是 (B,T,D) 或 (B,D,T)
        return x if x.shape[1] <= x.shape[2] else x.transpose(1,2).contiguous()
    if x.dim()==4:  # (B,C,T,D')
        B,C,T,Dp = x.shape
        return x.permute(0,2,1,3).contiguous().view(B,T,C*Dp)
    if x.dim()==5:  # (B,C,T,H,W)
        B,C,T,H,W = x.shape
        return x.permute(0,2,1,3,4).contiguous().view(B,T,C*H*W)
    raise ValueError(f"[coatt._ensure_btd] Unsupported shape: {tuple(x.shape)}")

def _match_time(v: torch.Tensor, a: torch.Tensor, to: str="video") -> Tuple[torch.Tensor, torch.Tensor]:
    """把两个序列在时间长度上对齐（线性插值，无截断）。"""
    Tv, Ta = v.shape[1], a.shape[1]
    if Tv == Ta:
        return v, a
    if to == "video":
        a2 = F.interpolate(a.transpose(1,2), size=Tv, mode="linear", align_corners=False).transpose(1,2)
        return v, a2
    else:
        v2 = F.interpolate(v.transpose(1,2), size=Ta, mode="linear", align_corners=False).transpose(1,2)
        return v2, a

# ----------------------------- 模块实现 -----------------------------
class _FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, mult*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mult*d_model, d_model),
            nn.Dropout(dropout)
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        return self.ln(x + self.net(x))

class _BiCoAttnLayer(nn.Module):
    """
    单层双向 Co-Attention（含 FFN），可选用“瓶颈 Key/Value”减少跨模态注意力代价：
      - v <- a_bottleneck
      - a <- v_bottleneck
    其中 bottleneck 由 Learnable Queries 汇聚得到，避免固定长度假设。
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float,
                 bottleneck_tokens: int):
        super().__init__()
        self.d_model = d_model
        self.bottleneck_tokens = int(bottleneck_tokens)

        # learnable queries 用于从每个模态提取 M 个摘要 token
        if self.bottleneck_tokens > 0:
            self.q_a2v = nn.Parameter(torch.randn(self.bottleneck_tokens, d_model))
            self.q_v2a = nn.Parameter(torch.randn(self.bottleneck_tokens, d_model))
            nn.init.xavier_uniform_(self.q_a2v)
            nn.init.xavier_uniform_(self.q_v2a)

        self.attn_v_from_a = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.attn_a_from_v = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        # 从源模态提摘要（queries attend to source）
        if self.bottleneck_tokens > 0:
            self.attn_a_reduce = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
            self.attn_v_reduce = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)

        self.ln_v1 = nn.LayerNorm(d_model); self.ln_v2 = nn.LayerNorm(d_model)
        self.ln_a1 = nn.LayerNorm(d_model); self.ln_a2 = nn.LayerNorm(d_model)
        self.ffn_v = _FeedForward(d_model, dropout)
        self.ffn_a = _FeedForward(d_model, dropout)
        self.drop = nn.Dropout(dropout)

    def _reduce(self, src: torch.Tensor, q_param: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """用 learnable queries 汇聚出 M 个摘要 token。"""
        # src:(B,T,D)  q_param:(M,D) -> q:(B,M,D)
        B = src.size(0)
        q = q_param.unsqueeze(0).expand(B, -1, -1)
        # queries attend to src -> (B,M,D), 同时可返回注意力权重
        out, w = self.attn_a_reduce(query=q, key=src, value=src, need_weights=True)
        return out, w  # (B,M,D), (B, M, T)

    def forward(self, v: torch.Tensor, a: torch.Tensor, need_weights: bool=False):
        """
        v,a: (B,T,D)
        返回更新后的 v,a，以及可选的注意力信息（用于可视化/调试）
        """
        attn_logs: Dict[str, List[torch.Tensor]] = {} if need_weights else None

        # 先从每个模态抽取瓶颈摘要（如果启用）
        if self.bottleneck_tokens > 0:
            a_sum, w_a = self._reduce(a, self.q_a2v)  # (B,M,D), (B,M,Ta)
            v_sum, w_v = self._reduce(v, self.q_v2a)  # (B,M,D), (B,M,Tv)
            if need_weights:
                attn_logs.setdefault("a_reduce", []).append(w_a)
                attn_logs.setdefault("v_reduce", []).append(w_v)
            k_a = v = v  # 占位便于阅读
            k_v = a = a  # 占位便于阅读
            # 用摘要作为跨模态的 K/V
            kv_a = a_sum  # a->v
            kv_v = v_sum  # v->a
        else:
            kv_a = a  # 全长度作为 K/V
            kv_v = v

        # v <- a 方向
        v2, w_va = self.attn_v_from_a(query=v, key=kv_a, value=kv_a, need_weights=need_weights)
        v = self.ln_v1(v + self.drop(v2))
        v = self.ln_v2(v + self.drop(self.ffn_v(v)))

        # a <- v 方向
        a2, w_av = self.attn_a_from_v(query=a, key=kv_v, value=kv_v, need_weights=need_weights)
        a = self.ln_a1(a + self.drop(a2))
        a = self.ln_a2(a + self.drop(self.ffn_a(a)))

        if need_weights:
            attn_logs.setdefault("v_from_a", []).append(w_va)  # (B, h, T_v, len(kv_a))
            attn_logs.setdefault("a_from_v", []).append(w_av)  # (B, h, T_a, len(kv_v))
        return v, a, attn_logs

class EnhancedCoAttention(nn.Module):
    """
    Co-attention 融合层（稳健）
    - 将 (B,Tv,Dv)、(B,Ta,Da) 投影到统一 d_model
    - 用 Learnable Queries 做瓶颈（bottleneck_tokens=M；M=0 则用全长度注意力）
    - 双向多头 co-attention 堆叠 num_layers 层
    - 输出 fused=(v+a) 线性投影后的时序特征，以及 z_v/z_a 的时序平均全局向量
    """
    def __init__(self,
                 video_dim: int,
                 audio_dim: int,
                 d_model: int = 256,
                 bottleneck_dim: int = 128,   # 这里解释为“瓶颈 token 数 M”，可与旧配置复用
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 match_time: str = "video"):
        super().__init__()
        assert match_time in ("video","audio")
        self.d_model = int(d_model)
        self.match_time = match_time
        # 输入投影
        self.v_in = nn.Linear(video_dim, d_model) if video_dim != d_model else nn.Identity()
        self.a_in = nn.Linear(audio_dim, d_model) if audio_dim != d_model else nn.Identity()
        # Co-attention 堆叠
        M = int(max(0, bottleneck_dim))  # 允许 0（表示不用瓶颈，直接全长度交互）
        self.layers = nn.ModuleList([
            _BiCoAttnLayer(d_model=d_model, num_heads=num_heads, dropout=dropout,
                           bottleneck_tokens=M)
            for _ in range(num_layers)
        ])
        # 融合头：把 v 与 a 拼接后映射回 d_model
        self.fuse_proj = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        log.info(f"[CoAttn] init: d_model={d_model}, layers={num_layers}, heads={num_heads}, "
                 f"bottleneck_tokens={M}, match_time={match_time}")

    def forward(self, v: torch.Tensor, a: torch.Tensor, return_attn: bool=False):
        """
        v,a: (B,T,*) or 其他形状（内部会统一为 (B,T,D)）
        return:
          fused:(B,T,d_model), z_v:(B,d_model), z_a:(B,d_model), [attn_info]
        """
        v = _ensure_btd(v); a = _ensure_btd(a)
        # 时间对齐（默认以视频为基准）
        v, a = _match_time(v, a, to=self.match_time)

        # 投影到统一维度
        v = self.v_in(v); a = self.a_in(a)

        # 逐层双向 co-attention
        attn_info = {"a_reduce": [], "v_reduce": [], "v_from_a": [], "a_from_v": []} if return_attn else None
        for i, blk in enumerate(self.layers):
            need_w = return_attn
            v, a, logs = blk(v, a, need_weights=need_w)
            if need_w and logs is not None:
                for k in logs.keys():
                    attn_info[k].append(logs[k])

        # 融合（逐时刻）并计算全局向量
        fused = self.fuse_proj(torch.cat([v, a], dim=-1))  # (B,T,d_model)
        z_v = v.mean(dim=1)
        z_a = a.mean(dim=1)

        if log.level <= 10:  # DEBUG
            log.debug(f"[CoAttn] v:{tuple(v.shape)} a:{tuple(a.shape)} fused:{tuple(fused.shape)} "
                      f"z_v:{tuple(z_v.shape)} z_a:{tuple(z_a.shape)}")

        if return_attn:
            # 精简一下权重体积：只保留每层平均到 batch 和头后的二维热度图（可视化更轻）
            summary = {}
            for k, lst in attn_info.items():
                if not lst:
                    summary[k] = []
                    continue
                # lst: List[Tensor]，每层一个，形状各异；这里做适度摘要
                comp = []
                for w in lst:
                    # w 可能是 (B, heads, Q, K) 或 (B, M, T)；统一到 (Q, K) 的 batch/头均值
                    dims = w.dim()
                    if dims == 4:
                        comp.append(w.mean(dim=(0,1)))  # -> (Q,K)
                    elif dims == 3:
                        comp.append(w.mean(dim=0))      # -> (Q,K)
                    else:
                        comp.append(w)
                summary[k] = comp
            return fused, z_v, z_a, summary

        return fused, z_v, z_a


# ===================== Adapter for EnhancedAVDetector =====================
import torch
import torch.nn as nn

class CoAttentionFusion(nn.Module):
    """
    适配 EnhancedAVTopDetector / EnhancedAVDetector 期望的接口：
    forward(video_feat, audio_feat) -> (fused, aux_dict)
      aux_dict:
        - 'video_emb': (B, D)
        - 'audio_emb': (B, D)
    注意：不强制提供 'video_seq' / 'audio_seq'，Detector 将在需要时回退使用输入序列。
    """
    def __init__(self,
                 video_dim: int,
                 audio_dim: int,
                 d_model: int = 256,
                 num_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bottleneck_dim: int = 0,
                 match_time: str = "video"):
        super().__init__()
        # 复用你已有的稳定版 EnhancedCoAttention
        self.core = EnhancedCoAttention(
            video_dim=video_dim,
            audio_dim=audio_dim,
            d_model=d_model,
            bottleneck_dim=bottleneck_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            match_time=match_time,
        )

    def forward(self, video_feat: torch.Tensor, audio_feat: torch.Tensor):
        # EnhancedCoAttention 返回 (fused, z_v, z_a) 或 (fused, z_v, z_a, attn)
        out = self.core(video_feat, audio_feat, return_attn=False)
        fused, z_v, z_a = out[:3]
        aux = {
            "video_emb": z_v,
            "audio_emb": z_a,
        }
        return fused, aux
