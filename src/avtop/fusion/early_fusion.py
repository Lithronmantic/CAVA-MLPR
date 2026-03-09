# -*- coding: utf-8 -*-
# src/avtop/fusion/early_fusion.py
from typing import List, Sequence, Tuple, Optional
import torch
from torch import nn
import torch.nn.functional as F
from src.avtop.utils.logging import get_logger

log = get_logger(__name__)

def ensure_btd(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 3:
        return x if x.shape[1] <= x.shape[2] else x.transpose(1,2).contiguous()
    elif x.dim() == 4:
        B,C,T,Dp = x.shape
        return x.permute(0,2,1,3).contiguous().view(B,T,C*Dp)
    elif x.dim() == 5:
        B,C,T,H,W = x.shape
        return x.permute(0,2,1,3,4).contiguous().view(B,T,C*H*W)
    else:
        raise ValueError(f"[early_fusion.ensure_btd] unsupported: {x.shape}")

# -*- coding: utf-8 -*-
# src/avtop/fusion/early_fusion.py (只展示修改后的 BackboneWrapper；其余代码保持你当前版本)

class BackboneWrapper(nn.Module):
    """用 forward hook 采集中间层输出为 (B,T,D)。layer_refs 为空时返回最终输出。"""
    def __init__(self, backbone: nn.Module, layer_refs: Optional[Sequence[str]] = None):
        super().__init__()
        self.backbone = backbone
        self.layer_refs = [r for r in (layer_refs or []) if isinstance(r, str) and r.strip() != ""]
        self._handles, self._collected = [], []

        # 方便你查看骨干里有什么模块
        name2mod = dict(self.backbone.named_modules())
        if get_logger(__name__).level <= 10:  # DEBUG
            all_names = list(name2mod.keys())
            # 打印前 50 个模块名（避免刷屏）
            log.debug(f"[Early] available backbone modules (first 50/{len(all_names)}): {all_names[:50]}")

        def _resolve_one(ref: str) -> Optional[str]:
            if ref in name2mod:
                return ref
            # 模糊匹配：先 endswith，再包含
            ends = [n for n in name2mod.keys() if n.endswith(ref)]
            if len(ends) == 1:
                return ends[0]
            if len(ends) > 1:
                # 取更长的（更具体）的名字
                ends.sort(key=len, reverse=True)
                log.warning(f"[Early] multiple matches for '{ref}' -> choose '{ends[0]}' among {ends[:3]}...")
                return ends[0]
            substr = [n for n in name2mod.keys() if ref in n]
            if len(substr) == 1:
                return substr[0]
            if len(substr) > 1:
                substr.sort(key=len, reverse=True)
                log.warning(f"[Early] multiple substring matches for '{ref}' -> choose '{substr[0]}' among {substr[:3]}...")
                return substr[0]
            # 全部找不到
            log.warning(f"[Early] layer ref '{ref}' not found in backbone. It will be skipped.")
            return None

        resolved = []
        for ref in self.layer_refs:
            hit = _resolve_one(ref)
            if hit is not None:
                resolved.append(hit)

        self.layer_refs = resolved
        if self.layer_refs:
            for name in self.layer_refs:
                h = name2mod[name].register_forward_hook(self._hook(name))
                self._handles.append(h)
            log.info(f"[Early] BackboneWrapper hooks -> {self.layer_refs}")
        else:
            log.warning("[Early] layer_refs is empty or none resolved; wrapper will return final output only.")

    def _hook(self, name: str):
        def fn(_m, _inp, out):
            y = ensure_btd(out)
            self._collected.append(y)
            if log.level <= 10:
                log.debug(f"[Early] hook@{name}: {tuple(out.shape)} -> {tuple(y.shape)}")
        return fn

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._collected = []
        out = self.backbone(x)
        if not self.layer_refs:
            self._collected.append(ensure_btd(out))
        else:
            if log.level <= 10:
                shapes = [tuple(t.shape) for t in self._collected]
                log.debug(f"[Early] collected stages: {shapes}")
        return self._collected

    def close(self):
        for h in self._handles: h.remove()
        self._handles.clear()
        log.info("[Early] BackboneWrapper hooks removed.")


class GatedResidualFuse(nn.Module):
    """门控残差早融合块（要求输入已对齐到相同 (B,T,D)）。"""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.v_gate = nn.Sequential(nn.Linear(dim*2, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.v_proj = nn.Sequential(nn.Linear(dim*2, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.a_gate = nn.Sequential(nn.Linear(dim*2, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.a_proj = nn.Sequential(nn.Linear(dim*2, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, v: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 这里假设 v/a 形状已经相同
        x_va = torch.cat([v, a], dim=-1)
        x_av = torch.cat([a, v], dim=-1)
        v_out = v + self.sigmoid(self.v_gate(x_va)) * self.v_proj(x_va)
        a_out = a + self.sigmoid(self.a_gate(x_av)) * self.a_proj(x_av)
        return v_out, a_out

class EarlyFusion(nn.Module):
    """
    在指定 stage 索引做门控残差融合，并在内部自动完成：
      1) 通道维映射：Dv/Da -> dim
      2) 时间长度匹配：对齐到 'match_time'（'video' 或 'audio'）
    """
    def __init__(self, dim: int, hidden: int, stages: Sequence[int],
                 in_dim_v: Optional[int] = None, in_dim_a: Optional[int] = None,
                 match_time: str = "video"):
        super().__init__()
        assert match_time in ("video","audio")
        self.stages = list(stages)
        self.blocks = nn.ModuleDict({str(i): GatedResidualFuse(dim, hidden) for i in self.stages})
        self.match_time = match_time

        # 维度映射层
        self.proj_v = nn.Linear(in_dim_v if in_dim_v is not None else dim, dim) if (in_dim_v and in_dim_v != dim) else nn.Identity()
        self.proj_a = nn.Linear(in_dim_a if in_dim_a is not None else dim, dim) if (in_dim_a and in_dim_a != dim) else nn.Identity()

        log.info(f"[Early] Enabled at stages={self.stages}, dim={dim}, hidden={hidden}, "
                 f"proj_v={'Linear' if isinstance(self.proj_v, nn.Linear) else 'Id'}, "
                 f"proj_a={'Linear' if isinstance(self.proj_a, nn.Linear) else 'Id'}, "
                 f"match_time={self.match_time}")

    @staticmethod
    def _match_time(v: torch.Tensor, a: torch.Tensor, to: str) -> Tuple[torch.Tensor, torch.Tensor]:
        Tv, Ta = v.shape[1], a.shape[1]
        if Tv == Ta:
            return v, a
        if to == "video":
            a2 = F.interpolate(a.transpose(1,2), size=Tv, mode="linear", align_corners=False).transpose(1,2)
            return v, a2
        else:
            v2 = F.interpolate(v.transpose(1,2), size=Ta, mode="linear", align_corners=False).transpose(1,2)
            return v2, a

    def forward(self, v_stages: List[torch.Tensor], a_stages: List[torch.Tensor]):
        assert len(v_stages) == len(a_stages), "stage length mismatch"
        out_v, out_a = [], []
        for idx, (v, a) in enumerate(zip(v_stages, a_stages)):
            v = ensure_btd(v); a = ensure_btd(a)
            # (1) 映射到统一维度
            v = self.proj_v(v)
            a = self.proj_a(a)
            # (2) 匹配时间长度
            v, a = self._match_time(v, a, self.match_time)
            if log.level <= 10:
                log.debug(f"[Early] stage{idx} after adapt: v{tuple(v.shape)} a{tuple(a.shape)}")
            # (3) 若该 stage 需要融合，则门控残差
            if str(idx) in self.blocks:
                v, a = self.blocks[str(idx)](v, a)
            out_v.append(v); out_a.append(a)

        if log.level <= 10:
            shapes = [tuple(t.shape) for t in out_v]
            log.debug(f"[Early] after fuse, stages shapes: {shapes}")
        return out_v, out_a
