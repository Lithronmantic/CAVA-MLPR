# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTemporalEncoder(nn.Module):
    """
    统一的时序编码器：
    - 若输入是 3D 序列特征: (B, T, D) -> 直接进入编码器
    - 若输入是 5D 帧序列:   (B, T, C, H, W) -> 空间全局平均池化 -> (B, T, C) -> 线性投影到 d_model
    - 支持 (video_only) 或 (video, audio) 两路；两路时先在特征维拼接再投影
    输出: (B, T, d_model)
    """
    def __init__(self, d_model: int = 256, n_layers: int = 2, use_transformer: bool = True, nhead: int = 8):
        super().__init__()
        self.d_model = d_model
        self.use_transformer = use_transformer

        # 先留一个输入投影（当 concat 或从 C 维映射过来时使用）
        self.in_proj = None  # 延迟创建：根据首个 forward 的实际输入通道数建
        if use_transformer:
            layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            self.encoder = nn.LSTM(input_size=d_model, hidden_size=d_model // 2, num_layers=n_layers,
                                   batch_first=True, bidirectional=True)

        self.out_dim = d_model

    @staticmethod
    def _to_seq_feat(x: torch.Tensor) -> torch.Tensor:
        """
        将输入规整为 (B, T, D)：
        - 3D: (B, T, D) -> 原样返回
        - 5D: (B, T, C, H, W) -> GAP 空间维 -> (B, T, C)
        其它维度数会报错
        """
        if x.dim() == 3:
            return x
        if x.dim() == 5:
            # 全局空间平均: (B,T,C,H,W)->(B,T,C)
            return x.mean(dim=(-1, -2))
        raise ValueError(f"SimpleTemporalEncoder expects 3D (B,T,D) or 5D (B,T,C,H,W), got {tuple(x.shape)}")

    def _ensure_in_proj(self, din: int):
        if self.in_proj is None or getattr(self.in_proj, 'in_features', None) != din:
            self.in_proj = nn.Linear(din, self.d_model)

    def forward(self, video: torch.Tensor, audio: torch.Tensor = None):
        """
        接受 1 路或 2 路。两路时先拼接后统一投影到 d_model。
        - video: (B,T,D) 或 (B,T,C,H,W)
        - audio: (B,T,D) 或 (B,T,C,H,W) 或 None
        """
        v = self._to_seq_feat(video)     # (B,T,Dv)
        if audio is not None:
            a = self._to_seq_feat(audio) # (B,T,Da)
            x = torch.cat([v, a], dim=-1)  # (B,T,Dv+Da)
        else:
            x = v

        self._ensure_in_proj(x.size(-1))
        x = self.in_proj(x)              # (B,T,d_model)

        if self.use_transformer:
            # TransformerEncoder (batch_first=True): (B, T, E)
            x = self.encoder(x)
        else:
            # LSTM: 返回 (B,T,d_model)（双向时拼接）
            x, _ = self.encoder(x)

        return x  # (B,T,d_model)
