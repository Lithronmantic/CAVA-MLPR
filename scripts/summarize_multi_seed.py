#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True, help="directory containing seed run subfolders")
    ap.add_argument("--metric_file", default="eval_metrics.json")
    args = ap.parse_args()

    root = Path(args.runs_root)
    rows = []
    for p in sorted(root.glob("*")):
        mf = p / args.metric_file
        if not mf.exists():
            continue
        try:
            obj = json.loads(mf.read_text(encoding="utf-8"))
            rows.append(
                {
                    "run": p.name,
                    "accuracy": float(obj.get("accuracy", 0.0)),
                    "macro_f1": float(obj.get("macro_f1", 0.0)),
                    "weighted_f1": float(obj.get("weighted_f1", 0.0)),
                }
            )
        except Exception:
            continue

    if not rows:
        print("No valid runs found.")
        return

    acc = np.array([r["accuracy"] for r in rows], dtype=np.float64)
    f1 = np.array([r["macro_f1"] for r in rows], dtype=np.float64)
    summary = {
        "num_runs": len(rows),
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=0)),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=0)),
        "runs": rows,
    }

    out = root / "multi_seed_summary.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved={out}")
    print(summary)


if __name__ == "__main__":
    main()
