import torch
import torch.nn as nn

class OfflineEncoder(nn.Module):
    """
    直接接收 [B,T,Dv] / [B,T,Da] 的预提特征（由 Dataset 提供），
    通过轻量 BiGRU 统一到 d_model 维；接口与 SimpleTemporalEncoder 对齐。
    """
    def __init__(self, d_v_in: int, d_a_in: int, d_model: int = 128):
        super().__init__()
        self.v_proj = nn.Linear(d_v_in, d_model)
        self.a_proj = nn.Linear(d_a_in, d_model)
        self.v_rnn  = nn.GRU(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.a_rnn  = nn.GRU(d_model, d_model//2, batch_first=True, bidirectional=True)
        self.out    = nn.Linear(d_model, d_model)  # 统一维度

    def forward(self, video_feats, audio_feats):
        v = self.v_proj(video_feats);  a = self.a_proj(audio_feats)
        v, _ = self.v_rnn(v);          a, _ = self.a_rnn(a)
        return self.out(v), self.out(a)     # [B,T,D], [B,T,D]
