# src/avtop/mil/dynamic_routing.py
import torch
import torch.nn as nn

class DynamicRoutingMIL(nn.Module):
    def __init__(self, d_in: int, num_classes: int = 2,
                 tau: float = 2.0, topk: int = None):
        super().__init__()
        self.router = nn.Sequential(
            nn.Linear(d_in, d_in), nn.ReLU(inplace=True), nn.Linear(d_in, 1)
        )
        self.head   = nn.Linear(d_in, num_classes)
        self.tau    = float(tau)      # 温度；>1 更平滑
        self.topk   = topk            # 若设置，则用 Top-K 平均

    def set_tau(self, tau: float):
        self.tau = float(tau)

    def forward(self, z: torch.Tensor):
        # z: [B, T, D]
        scores = self.router(z).squeeze(-1)         # [B, T]
        if self.topk is not None and self.topk > 0:
            k = min(int(self.topk), scores.size(1))
            idx = scores.topk(k, dim=1).indices
            w = torch.zeros_like(scores).scatter(1, idx, 1.0 / k)  # Top-K 均匀
        else:
            w = torch.softmax(scores / self.tau, dim=1)            # 温度软注意力
        clip = (z * w.unsqueeze(-1)).sum(dim=1)                    # [B, D]
        return {
            "clip_logits": self.head(clip),        # [B, C]
            "segment_logits": self.head(z),        # [B, T, C]
            "weights": w                           # [B, T]
        }
