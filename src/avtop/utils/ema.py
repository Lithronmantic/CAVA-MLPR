# -*- coding: utf-8 -*-
# src/avtop/utils/ema.py
import sys
import io
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from typing import Optional
import copy, torch
from torch import nn
from src.avtop.utils.logging import get_logger

log = get_logger(__name__)

class ModelEMA:
    """Exponential Moving Average (EMA) for model weights/buffers."""
    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 device: Optional[torch.device] = None, include_buffers: bool = True):
        self.decay = decay
        self.device = device
        self.include_buffers = include_buffers
        self.shadow = {}
        self.backup = {}

        n_params = 0
        for n, p in model.named_parameters():
            #if p.requires_grad:
                t = p.detach().clone()
                if device is not None: t = t.to(device)
                self.shadow[n] = t
                n_params += t.numel()

        if include_buffers:
            for n, b in model.named_buffers():
                t = b.detach().clone()
                if device is not None: t = t.to(device)
                self.shadow[f"buf::{n}"] = t
        log.info(f"[EMA] Initialized with decay={decay}, tracked_params={n_params:,}")

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            #if not p.requires_grad: continue
            x = p.detach()
            if self.device is not None: x = x.to(self.device)
            self.shadow[n].mul_(d).add_(x, alpha=1.0 - d)
        if self.include_buffers:
            for n, b in model.named_buffers():
                key = f"buf::{n}"
                x = b.detach()
                if self.device is not None: x = x.to(self.device)
                self.shadow[key].copy_(x)

    @torch.no_grad()
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.backup[n] = p.detach().clone()
            p.data.copy_(self.shadow[n])
        if self.include_buffers:
            for n, b in model.named_buffers():
                key = f"buf::{n}"
                self.backup[key] = b.detach().clone()
                b.data.copy_(self.shadow[key])
        log.info("[EMA] Shadow applied to model for eval/export.")

    @torch.no_grad()
    def restore(self, model: nn.Module):
        assert self.backup, "No backup found; call apply_shadow() first."
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            p.data.copy_(self.backup[n])
        if self.include_buffers:
            for n, b in model.named_buffers():
                key = f"buf::{n}"
                b.data.copy_(self.backup[key])
        self.backup = {}
        log.info("[EMA] Model weights restored to training state.")

    def state_dict(self):
        return dict(decay=self.decay, include_buffers=self.include_buffers,
                    shadow=copy.deepcopy(self.shadow))

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        self.include_buffers = bool(state["include_buffers"])
        self.shadow = state["shadow"]
        self.backup = {}
        log.info("[EMA] State loaded.")
