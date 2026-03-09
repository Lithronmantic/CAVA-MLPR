#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, argparse, yaml, random, numpy as np, torch
import multiprocessing as mp
from pathlib import Path

# 鍥哄畾鍞竴鏉ユ簮锛氬彧寮曠敤锛屼笉鑷畾涔?
from dataset import AVFromCSV, safe_collate_fn
from strong_trainer import StrongTrainer
try:
    from config_system import resolve_runtime_config
except Exception:
    from scripts.config_system import resolve_runtime_config

# --- Robust batch unpack ---
def _unpack_batch(b):
    """
    鏀寔浠ヤ笅鏍煎紡锛?
      - tuple/list: (v, a, y) 鎴?(v, a, y, ids/meta)
      - dict: {'video':..., 'audio':..., 'label':..., 'ids':...}锛堥敭鍚嶅ぇ灏忓啓涓嶆晱鎰燂級
    杩斿洖: (v, a, y, ids_or_meta)锛涜嫢鏃?ids/meta 鍒欎负 None
    """
    if isinstance(b, dict):
        # 灏濊瘯甯歌閿紙澶у皬鍐欎笉鏁忔劅锛?
        keys = {k.lower(): k for k in b.keys()}
        def _req(name):
            if name not in keys:
                raise KeyError(f"batch dict 缂哄皯蹇呴』閿? {name}")
            return b[keys[name]]
        v = _req('video')
        a = _req('audio')
        y = _req('label')
        ids = b.get(keys.get('ids')) if 'ids' in keys else (b.get(keys.get('meta')) if 'meta' in keys else None)
        return v, a, y, ids

    if isinstance(b, (list, tuple)):
        if len(b) >= 3:
            v, a, y = b[:3]
            ids = b[3] if len(b) >= 4 else None
            return v, a, y, ids

    raise ValueError(f"Unsupported batch structure: type={type(b)}, len={len(b) if hasattr(b,'__len__') else 'N/A'}")


def main():
    parser = argparse.ArgumentParser(description='馃殌 Clean Training Entry')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/run")
    parser.add_argument("--diagnose", action="store_true",
                        help="浠呭仛鏋勫缓/棣栨壒娆?鍓嶅悜璇婃柇锛屼笉杩涘叆璁粌")
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint (optional)')
    parser.add_argument("--multi_gpu", action="store_true", help="Enable single-process DataParallel")
    parser.add_argument("--num_gpus", type=int, default=0, help="GPU count for DataParallel (0=auto)")
    parser.add_argument("--runtime_batch_size", type=int, default=0, help="Runtime micro-batch size override")
    parser.add_argument("--seed", type=int, default=-1, help="Override seed from config when >= 0")
    args = parser.parse_args()

    # 寮哄埗鎶?YAML 瑙ｆ瀽鎴?dict
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
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    print("\n" + "="*80)
    print("馃殌 Enhanced Semi-Supervised Training Script (CLEAN ENTRY)")
    print("="*80)
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Seed:   {seed}")
    print("="*80)

    if args.diagnose:
        # 鏋勫缓 Trainer锛堝唴閮ㄤ細鏋勫缓 dataloader / model / optimizer 绛夛級
        st = StrongTrainer(cfg, args.output)
        print("[DIAG] Building one train batch and forward ...")
        st.model.eval()
        try:
            it = iter(st.loader_l)
            b = next(it)
        except StopIteration:
            raise RuntimeError("Training set is empty; please check labeled_csv or data filtering.")

        with torch.no_grad():
            # 鉁?浣跨敤鍋ュ．瑙ｅ寘锛氬吋瀹?3/4 鍏冪粍鍜?dict
            v, a, y, _ = _unpack_batch(b)
            # 鏍囩缁村害锛氳嫢鏄?one-hot [B,C]锛岃浆鎴?index [B]
            if hasattr(y, "ndim") and y.ndim == 2:
                y = y.argmax(dim=1)
            v, a = v.to(st.device), a.to(st.device)

            # 鍓嶅悜
            out = st._forward(v, a, use_amp=st.amp_enabled)
            if out is None:
                raise RuntimeError("Forward returned None; please check model/input.")
            logits = out["clip_logits"] if isinstance(out, dict) and "clip_logits" in out else out

            # 棰濆鎵撳嵃鍏抽敭褰㈢姸锛屼究浜庡揩閫熷畾浣嶉棶棰?
            vshape = tuple(v.shape) if hasattr(v, "shape") else type(v)
            ashape = tuple(a.shape) if hasattr(a, "shape") else type(a)
            yshape = tuple(y.shape) if hasattr(y, "shape") else type(y)
            lshape = tuple(logits.shape) if hasattr(logits, "shape") else type(logits)
            print(f"[DIAG] batch: v={vshape}, a={ashape}, y={yshape}")
            print(f"[DIAG] model forward OK, logits shape={lshape}")

            # 濡傛湁 CAVA锛屾墦鍗板叧閿緟鍔╅噺鍙敤鎬?
            if isinstance(out, dict):
                flags = {k: (out.get(k) is not None) for k in
                         ["audio_seq", "audio_aligned", "video_proj", "causal_gate",
                          "delay_frames", "causal_prob", "causal_prob_dist", "pred_delay"]}
                print(f"[DIAG] CAVA keys: {flags}")

        print("[DIAG] Done.")
        return

    print("Using StrongTrainer for training...")
    try:
        st = StrongTrainer(cfg, args.output)
        st.train()
    except Exception:
        # 鎵撳嵃瀹屾暣鍫嗘爤锛岄伩鍏嶅彧鐪嬪埌鈥淭raceback鈥濅絾鏃犵粏鑺?
        import traceback
        traceback.print_exc()
        raise

    print("\n" + "="*80)
    print("鉁?All Done!")
    print("="*80)
    print(f"馃搨 Results:     {args.output}")
    print(f"馃捑 Checkpoints: {args.output}/checkpoints/")
    print(f"馃摑 Logs:        {args.output}/logs/ (鎸夐渶杩藉姞)")


if __name__ == "__main__":
    # Windows 瀹夊叏鍏ュ彛
    mp.freeze_support()
    main()
