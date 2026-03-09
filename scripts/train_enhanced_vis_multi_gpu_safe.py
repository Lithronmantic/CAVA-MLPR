#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import random
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch
import yaml

from strong_trainer import StrongTrainer
try:
    from config_system import resolve_runtime_config
except Exception:
    from scripts.config_system import resolve_runtime_config


def _to_tensor_safe(x, device):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (float, int, bool)):
        return torch.tensor(float(x), device=device)
    return x


def _sanitize_output(obj, device):
    if isinstance(obj, dict):
        return {k: _sanitize_output(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [_sanitize_output(v, device) for v in obj]
        return type(obj)(out)
    return _to_tensor_safe(obj, device)


def _patch_model_forward_for_dp(model):
    if model is None:
        return
    orig_forward = model.forward

    def wrapped_forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        # Use clip_logits device if available, otherwise infer from first tensor arg.
        dev = None
        if isinstance(out, dict) and torch.is_tensor(out.get("clip_logits", None)):
            dev = out["clip_logits"].device
        if dev is None:
            for a in args:
                if torch.is_tensor(a):
                    dev = a.device
                    break
        if dev is None:
            dev = torch.device("cpu")
        return _sanitize_output(out, dev)

    model.forward = wrapped_forward


def main():
    parser = argparse.ArgumentParser(description="Enhanced Training Entry (Multi-GPU Safe)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/run")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--multi_gpu", action="store_true", help="Enable single-process DataParallel")
    parser.add_argument("--num_gpus", type=int, default=0, help="GPU count for DataParallel (0=auto)")
    parser.add_argument("--runtime_batch_size", type=int, default=0, help="Runtime micro-batch size override")
    parser.add_argument("--seed", type=int, default=-1, help="Override seed from config when >= 0")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise TypeError(f"Config must parse to dict: {args.config}")
    cfg = resolve_runtime_config(cfg)

    tr = cfg.setdefault("training", {})
    if args.multi_gpu:
        tr["multi_gpu"] = True
        if args.num_gpus and args.num_gpus > 0:
            tr["num_gpus"] = int(args.num_gpus)
    if args.runtime_batch_size and args.runtime_batch_size > 0:
        tr["runtime_batch_size"] = int(args.runtime_batch_size)
    if args.seed is not None and int(args.seed) >= 0:
        cfg["seed"] = int(args.seed)

    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print("\n" + "=" * 80)
    print("Enhanced Semi-Supervised Training Script (MULTI-GPU SAFE ENTRY)")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Seed:   {seed}")
    print("=" * 80)

    st = StrongTrainer(cfg, args.output, args.checkpoint)
    # Patch both student and teacher to make DataParallel gather robust.
    student = st.model.module if isinstance(st.model, torch.nn.DataParallel) else st.model
    _patch_model_forward_for_dp(student)
    if getattr(st, "teacher", None) is not None:
        _patch_model_forward_for_dp(st.teacher)

    if args.diagnose:
        st.model.eval()
        b = next(iter(st.loader_l))
        if isinstance(b, (list, tuple)) and len(b) == 4:
            v, a, y, _ = b
        else:
            v, a, y = b
        if hasattr(y, "ndim") and y.ndim == 2:
            y = y.argmax(dim=1)
        v, a = v.to(st.device), a.to(st.device)
        with torch.no_grad():
            out = st._forward(v, a, use_amp=st.amp_enabled)
            logits = out["clip_logits"] if isinstance(out, dict) else out
            print(f"[DIAG] logits shape={tuple(logits.shape)}")
        print("[DIAG] Done.")
        return

    st.train()


if __name__ == "__main__":
    mp.freeze_support()
    main()

