#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path

from config_system import extract_key_config, load_yaml, resolve_runtime_config


def _read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="outputs/repro_report.md")
    args = ap.parse_args()

    cfg = resolve_runtime_config(load_yaml(args.config))
    key_cfg = extract_key_config(cfg)
    run_dir = Path(args.run_dir)
    audit = _read_json(run_dir / "stats" / "config_audit.json")
    eval_metrics = _read_json(run_dir / "eval_metrics.json")
    loss_hist = _read_json(run_dir / "loss_history.json")

    lines = []
    lines.append("# Repro Report")
    lines.append("")
    lines.append("## Config Snapshot")
    for k, v in key_cfg.items():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Eq7 Ambiguity")
    eq7_mode = key_cfg.get("eq7_negative_definition", "paper_formula_exact")
    lines.append(f"- eq7_negative_definition: {eq7_mode}")
    lines.append("- available options: paper_formula_exact | paper_text_exact")
    lines.append("- ambiguity: formula suggests sequence-local denominator; text suggests same-batch negatives.")
    if eq7_mode == "paper_formula_exact":
        lines.append("- chosen policy: follow formula (sequence-local).")
        lines.append("- note: this may conflict with paper text description.")
    else:
        lines.append("- chosen policy: follow text (same-batch negatives).")
        lines.append("- note: this may conflict with formula notation.")
    lines.append("")

    lines.append("## FPS And Prior Semantics")
    lines.append(f"- dataset_nominal_fps: {key_cfg.get('dataset_nominal_fps')}")
    lines.append(f"- working_fps: {key_cfg.get('working_fps')}")
    lines.append(f"- delay_range_frames: [{key_cfg.get('delta_low_frames')}, {key_cfg.get('delta_high_frames')}]")
    lines.append(f"- delta_prior: {key_cfg.get('delta_prior')}")
    if key_cfg.get("dataset_nominal_fps") != key_cfg.get("working_fps"):
        lines.append("- interpretation: working_fps is a downsampled training-time axis, not original capture rate.")
    lines.append("")

    lines.append("## Config Audit")
    if audit is None:
        lines.append("- config_audit.json: missing")
    else:
        lines.append(f"- is_paper_exact: {audit.get('is_paper_exact')}")
        lines.append(f"- num_diffs: {audit.get('num_diffs')}")
        for row in audit.get("diffs", [])[:20]:
            lines.append(f"- diff {row.get('key')}: current={row.get('current')} vs paper={row.get('paper_exact')}")
    lines.append("")

    lines.append("## Eval Metrics")
    if eval_metrics is None:
        lines.append("- eval_metrics.json: missing")
    else:
        for k in ["accuracy", "macro_f1", "weighted_f1"]:
            if k in eval_metrics:
                lines.append(f"- {k}: {eval_metrics[k]}")
    lines.append("")

    lines.append("## Train Loss Summary")
    if loss_hist is None:
        lines.append("- loss_history.json: missing")
    else:
        for k in ["total_loss", "sup_loss", "cava_loss"]:
            arr = loss_hist.get(k, [])
            if arr:
                lines.append(f"- {k}: first={arr[0]:.6f}, last={arr[-1]:.6f}, n={len(arr)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved={out}")


if __name__ == "__main__":
    main()
