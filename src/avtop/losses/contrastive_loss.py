# -*- coding: utf-8 -*-
# src/avtop/losses/contrastive_loss.py
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed import is_initialized as ddp_is_init
from src.avtop.utils.logging import get_logger

log = get_logger(__name__)

def _gather_all(x: torch.Tensor) -> torch.Tensor:
    if not ddp_is_init(): return x
    import torch.distributed as dist
    world = dist.get_world_size()
    xs = [torch.zeros_like(x) for _ in range(world)]
    dist.all_gather(xs, x.contiguous())
    return torch.cat(xs, dim=0)

class EnhancedInfoNCE(nn.Module):
    """
    强化 InfoNCE：
      - 可学习温度 tau
      - Top-k 硬负（排除正对/同类）
      - DDP all_gather 扩充负样本池
    """
    def __init__(self, dim: int, temperature: float = 0.07, hard_k: int = 64, margin: float = 0.1):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(float(temperature)))
        self.hard_k = int(hard_k)
        self.margin = float(margin)
        log.info(f"[InfoNCE] init: tau={temperature}, hard_k={hard_k}, margin={margin}")

    def forward(self, z_v: torch.Tensor, z_a: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        # 归一化
        z_v = F.normalize(z_v, dim=-1)
        z_a = F.normalize(z_a, dim=-1)

        zv_all = _gather_all(z_v)
        za_all = _gather_all(z_a)
        sim = (z_v @ za_all.t()) / self.tau.clamp(1e-3, 1e3)  # (B, B_all)

        # 主损：正样本匹配同 index（假定 batch 对齐）
        target = torch.arange(z_v.size(0), device=z_v.device)
        loss_main = F.cross_entropy(sim, target)

        # 硬负
        with torch.no_grad():
            mask_pos = torch.zeros_like(sim, dtype=torch.bool)
            mask_pos[torch.arange(z_v.size(0), device=z_v.device),
                     torch.arange(z_v.size(0), device=z_v.device)] = True
            if labels is not None:
                labs_local = labels
                labs_all = _gather_all(labels)
                same = labs_local.unsqueeze(1).eq(labs_all.unsqueeze(0))
                mask_pos |= same
            sim_neg = sim.masked_fill(mask_pos, float('-inf'))
            k = min(self.hard_k, sim_neg.size(1)-1)
            topk_vals, _ = torch.topk(sim_neg, k=k, dim=1)  # (B,k)

        loss_hard = F.relu(topk_vals + self.margin).mean()
        loss = loss_main + 0.5 * loss_hard

        if log.level <= 20:
            log.info(f"[InfoNCE] loss={loss.item():.4f} main={loss_main.item():.4f} "
                     f"hard={loss_hard.item():.4f} tau={self.tau.item():.4f}")

        metrics = {
            "loss": float(loss.detach().cpu()),
            "loss_main": float(loss_main.detach().cpu()),
            "loss_hard_neg": float(loss_hard.detach().cpu()),
            "tau": float(self.tau.detach().cpu())
        }
        return loss, metrics
