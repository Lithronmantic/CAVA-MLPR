#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

from config_system import load_yaml, resolve_runtime_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = resolve_runtime_config(load_yaml(args.config))
    data = cfg.get("data", {})
    keys = ["labeled_csv", "val_csv", "unlabeled_csv"]
    print("[DATA_CHECK] start")
    ok = True
    for k in keys:
        p = data.get(k, None)
        if not p:
            print(f"- {k}: <empty>")
            continue
        exists = Path(p).exists()
        print(f"- {k}: {p} | exists={exists}")
        ok = ok and exists
    print(f"[DATA_CHECK] ready={ok}")


if __name__ == "__main__":
    main()
