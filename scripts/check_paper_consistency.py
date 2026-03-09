#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from config_system import (
    audit_against_paper_exact,
    load_paper_exact_config,
    load_yaml,
    resolve_runtime_config,
)

STRICT_KEYS = [
    "tau_nce",
    "window_size",
    "gate_range",
    "lambda_edge",
    "lambda_prior",
    "lambda_gate",
    "delta_prior",
    "lambda_cava",
    "ema_momentum",
    "mil_topk_ratio",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "epochs",
    "inner_lr_alpha",
    "eq7_negative_definition",
    "dataset_nominal_fps",
    "working_fps",
    "delta_low_frames",
    "delta_high_frames",
    "split_ratio",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--strict", action="store_true", help="exit non-zero if not paper_exact")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    cfg = resolve_runtime_config(load_yaml(args.config))
    paper = load_paper_exact_config(repo_root)
    summary = audit_against_paper_exact(cfg, paper)

    print("[PAPER_CONSISTENCY]")
    print(f"is_paper_exact={summary['is_paper_exact']}")
    print(f"num_diffs={summary['num_diffs']}")
    for row in summary["diffs"][:50]:
        print(f"- {row['key']}: current={row['current']} | paper_exact={row['paper_exact']}")

    out = repo_root / "outputs" / "paper_consistency_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={out}")

    if args.strict:
        cur = summary.get("current", {})
        ref = summary.get("paper_exact", {})
        missing = [k for k in STRICT_KEYS if (k not in cur) or (k not in ref)]
        strict_diffs = [k for k in STRICT_KEYS if (k in cur and k in ref and cur[k] != ref[k])]
        eq7_ok = cur.get("eq7_negative_definition") in {"paper_formula_exact", "paper_text_exact"}
        print(f"strict_keys={STRICT_KEYS}")
        print(f"strict_missing={missing}")
        print(f"strict_diffs={strict_diffs}")
        print(f"strict_eq7_mode_valid={eq7_ok}")
        if missing or strict_diffs or (not eq7_ok):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
