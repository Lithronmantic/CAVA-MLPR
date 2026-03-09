import torch
import torch.nn as nn

class CFAFusion(nn.Module):
    """
    Cross-Modal Feature Gating with residual mixing.
    输入: v,a ∈ [B,T,D]；输出: z ∈ [B,T,D]
    """
    def __init__(self, d_in: int, d_hidden: int = None, dropout: float = 0.1):
        super().__init__()
        d_h = d_hidden or d_in
        self.g_v = nn.Sequential(nn.Linear(2*d_in, d_h), nn.ReLU(inplace=True),
                                 nn.Linear(d_h, d_in), nn.Sigmoid())
        self.g_a = nn.Sequential(nn.Linear(2*d_in, d_h), nn.ReLU(inplace=True),
                                 nn.Linear(d_h, d_in), nn.Sigmoid())
        self.mix = nn.Sequential(nn.Linear(2*d_in, d_h), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(d_h, d_in))
        self.ln = nn.LayerNorm(d_in)

    def forward(self, v, a):
        x = torch.cat([v, a], dim=-1)      # [B,T,2D]
        gv = self.g_v(x); ga = self.g_a(x) # 门控 ∈ (0,1)
        fused = gv * v + ga * a
        fused = fused + self.mix(x)        # 残差混合
        return self.ln(fused), {"gv": gv, "ga": ga}
