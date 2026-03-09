# -*- coding: utf-8 -*-
# src/avtop/trainers/semisup_mean_teacher.py
from typing import Callable, Dict, Any, Optional
from copy import deepcopy
import math, torch
import torch.nn.functional as F
from torch import nn

from src.avtop.utils.ema import ModelEMA
from src.avtop.utils.logging import get_logger

log = get_logger(__name__)
# from torch.cuda.amp import autocast, GradScaler
# 原：from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler
import torch  # 确保有 torch 可用来判断设备


def kl_with_temperature(p_logits, q_logits, T: float = 2.0):
    p = F.log_softmax(p_logits / T, dim=-1)
    q = F.softmax(q_logits / T, dim=-1)
    return F.kl_div(p, q, reduction='batchmean') * (T * T)

def ramp_up(step, max_step, max_w=1.0):
    t = min(1.0, step / max(1, int(0.3*max_step)))
    return max_w * math.exp(-5 * (1 - t) * (1 - t))

class SemiSupTrainer:
    """Mean-Teacher + FixMatch 统一 Trainer（含对齐共享与详细日志）"""
    def __init__(self,
                 model_student: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 labeled_loader,
                 unlabeled_loader,
                 *,
                 weak_aug: Optional[Callable] = None,
                 strong_aug: Optional[Callable] = None,
                 sup_criterion: Optional[Callable] = None,
                 unsup_weight_max: float = 1.0,
                 fixmatch_thresh: float = 0.95,
                 consistency_T: float = 2.0,
                 total_steps: int = 100000,
                 device: str = "cuda",
                 use_amp: bool = True,
                 contrastive: Optional[nn.Module] = None):
        self.device = device
        # 自动确定 AMP 设备类型
        if isinstance(self.device, str):
            dev_str = self.device
        else:
            dev_str = str(self.device)

        if dev_str.startswith("cuda") and torch.cuda.is_available():
            self.amp_device_type = "cuda"
        elif dev_str.startswith("mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            self.amp_device_type = "mps"
        else:
            self.amp_device_type = "cpu"

        # GradScaler 新接口（只有在非 CPU 时才启用 AMP）
        self.scaler = GradScaler(self.amp_device_type, enabled=use_amp and self.amp_device_type != "cpu")
        self.use_amp = bool(use_amp and self.amp_device_type != "cpu")
        self.model_s = model_student.to(device)
        self.model_t = deepcopy(model_student).to(device).eval()
        for p in self.model_t.parameters(): p.requires_grad = False
        self.ema_t = ModelEMA(self.model_t, decay=0.999)

        self.opt = optimizer
        self.sup_criterion = sup_criterion or nn.BCEWithLogitsLoss()
        self.l_loader = labeled_loader
        self.u_loader = unlabeled_loader
        self.weak_aug = weak_aug or (lambda v,a: (v,a))
        self.strong_aug = strong_aug or (lambda v,a: (v,a))

        self.unsup_weight_max = float(unsup_weight_max)
        self.fixmatch_thresh = float(fixmatch_thresh)
        self.consistency_T = float(consistency_T)
        self.total_steps = int(total_steps)

        self.scaler = GradScaler(enabled=use_amp)
        self.use_amp = use_amp

        self.contrastive = contrastive  # EnhancedInfoNCE or None

        log.info(f"[Trainer] MT+FixMatch init: unsup_w_max={unsup_weight_max}, "
                 f"th={fixmatch_thresh}, T={consistency_T}, amp={use_amp}")

    def _iter_unlabeled(self):
        while True:
            for b in self.u_loader:
                yield b

    def train(self, max_epochs: int = 100, log_interval: int = 50) -> Dict[str, Any]:
        self.model_s.train()
        self.opt.zero_grad(set_to_none=True)
        u_iter = self._iter_unlabeled()

        step = 0
        for epoch in range(max_epochs):
            for batch_l in self.l_loader:
                v_l, a_l, y_l = [x.to(self.device, non_blocking=True) for x in batch_l]
                lab_is_multilabel = (y_l.dtype == torch.float32 or y_l.ndim == 2)

                # ------ 监督分支 ------
                with autocast(self.amp_device_type, enabled=self.use_amp):
                    out_l = self.model_s(v_l, a_l)
                    logits_l = out_l["clip_logits"]
                    if lab_is_multilabel:
                        loss_sup = self.sup_criterion(logits_l, y_l.float())
                    else:
                        loss_sup = nn.CrossEntropyLoss()(logits_l, y_l.long())

                # ------ 无监督分支（弱->Teacher，强->Student，共享 shift）------
                v_u, a_u = next(u_iter)
                v_u = v_u.to(self.device, non_blocking=True)
                a_u = a_u.to(self.device, non_blocking=True)

                # 弱增强给 Teacher
                v_w, a_w = self.weak_aug(v_u, a_u)
                with torch.no_grad():
                    shift = None
                    v_al = a_al = None
                    if hasattr(self.model_t, "align_predict_and_cache"):
                        v_al, a_al, shift = self.model_t.align_predict_and_cache(v_w, a_w, return_shift=True)
                        if shift is not None and hasattr(self.model_s, "set_cached_shift"):
                            self.model_s.set_cached_shift(shift)
                    # Teacher 前向：优先使用对齐后的特征，避免再次预测不一致的 shift
                    if v_al is not None and a_al is not None and hasattr(self.model_t, "forward_from_features"):
                        out_t = self.model_t.forward_from_features(v_al, a_al)
                    else:
                        out_t = self.model_t(v_w, a_w)
                    t_logits = out_t["clip_logits"]
                    t_prob = torch.sigmoid(t_logits)
                    mask = (t_prob >= self.fixmatch_thresh).float()

                # 强增强给 Student（并复用 Teacher 的 shift）
                v_s, a_s = self.strong_aug(v_u, a_u)
                used_cached = False
                if hasattr(self.model_s, "align_use_cached") and shift is not None:
                    v_s, a_s = self.model_s.align_use_cached(v_s, a_s)
                    used_cached = True

                if used_cached and hasattr(self.model_s, "forward_from_features"):
                    out_s = self.model_s.forward_from_features(v_s, a_s)
                else:
                    out_s = self.model_s(v_s, a_s)

                s_logits = out_s["clip_logits"]

                with autocast(self.amp_device_type, enabled=self.use_amp):
                    # 一致性（KL）
                    loss_cons = kl_with_temperature(s_logits, t_logits, T=self.consistency_T)

                    # FixMatch（多标签：BCE with logits；按 mask 选择高置信伪标签）
                    pseudo = (t_prob >= self.fixmatch_thresh).float()
                    loss_fix = (nn.BCEWithLogitsLoss(reduction="none")(s_logits, pseudo) * mask).mean()

                    # ramp-up 权重
                    wu = ramp_up(step, self.total_steps, self.unsup_weight_max)
                    loss_unsup = wu * (0.5 * loss_cons + 0.5 * loss_fix)

                    # 可选对比学习（使用模型输出的 z_v/z_a）
                    loss_ctr = torch.tensor(0., device=self.device)
                    if self.contrastive is not None:
                        z_v = out_s.get("z_v", None)
                        z_a = out_s.get("z_a", None)
                        if (z_v is not None) and (z_a is not None):
                            lc, mtr = self.contrastive(z_v, z_a, labels=None)
                            loss_ctr = lc
                            if log.level <= 20:
                                log.info(f"[Ctr] step={step} tau={mtr['tau']:.4f} loss={mtr['loss']:.4f}")

                    loss = loss_sup + loss_unsup + loss_ctr

                # ------ 反传与优化 ------
                self.scaler.scale(loss).backward()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad(set_to_none=True)

                # 更新 teacher（同步 + EMA）
                with torch.no_grad():
                    self.model_t.load_state_dict(self.model_s.state_dict(), strict=False)
                self.ema_t.update(self.model_t)

                if step % log_interval == 0:
                    pos_ratio = float(mask.mean().detach().cpu())
                    log.info(f"[Step {step:06d}] sup={loss_sup.item():.4f} unsup={loss_unsup.item():.4f} "
                             f"cons={loss_cons.item():.4f} fix={loss_fix.item():.4f} wu={wu:.3f} mask_pos={pos_ratio:.3f}")
                step += 1

            log.info(f"[Epoch {epoch + 1}/{max_epochs}] done.")

