#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Linux-friendly CSV copies by normalizing path separators.

Usage:
  python scripts/normalize_csv_paths_linux.py --in data/train.csv --out data/train_linux.csv
"""
import argparse
import csv
from pathlib import Path


def _norm_path(s: str) -> str:
    if s is None:
        return s
    s = str(s).strip()
    if not s:
        return s
    return s.replace("\\", "/")


def normalize_csv(in_csv: Path, out_csv: Path) -> None:
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    for r in rows:
        if "video_path" in r:
            r["video_path"] = _norm_path(r["video_path"])
        if "audio_path" in r:
            r["audio_path"] = _norm_path(r["audio_path"])
        if "video" in r:
            r["video"] = _norm_path(r["video"])
        if "audio" in r:
            r["audio"] = _norm_path(r["audio"])

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_csv", required=True, help="input CSV path")
    ap.add_argument("--out", dest="out_csv", required=True, help="output CSV path")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    out_csv = Path(args.out_csv)
    normalize_csv(in_csv, out_csv)
    print(f"[OK] normalized: {in_csv} -> {out_csv}")


if __name__ == "__main__":
    main()

