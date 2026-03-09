# AVTOP: Multimodal Welding Defect Detection Project

## This Project Does What

This project is used for industrial welding defect detection with multimodal signals (video + audio), including:

- model training for defect classification/detection,
- semi-supervised training with labeled and unlabeled samples,
- evaluation and reproducibility reporting.

## How to Run

This section is for internal/research use in the current project workspace.

1. Prepare your environment
- Python 3.10+ recommended
- PyTorch + CUDA environment
- install dependencies from your project environment setup

2. Train (paper profile)
```bash
python scripts/train_enhanced_vis.py --config configs/exact.yaml --output outputs/paper_exact_seed
```

3. Evaluate
```bash
python scripts/eval_enhanced.py --config configs/exact.yaml --checkpoint 
```

4. Consistency audit
```bash
python scripts/check_paper_consistency.py --config configs/exact.yaml --strict
```

5. Build physical prior (if required by your experiment setup)
```bash
python scripts/build_physical_prior.py --config configs/paper_exact.yaml
```

Notes:
- For Linux/server runs, use the Linux config variant if your paths differ.
- For limited GPU memory, use runtime batch with gradient accumulation settings in trainer/CLI.

## Code Structure

```text
configs/
  exact.yaml
  exact_linux.yaml
  repro_safe.yaml
  research_extended.yaml

scripts/
  train_enhanced_vis.py            # training entry
  strong_trainer.py                # core training loop
  enhanced_detector.py             # main model
  cava.py                          # CAVA modules
  cava_losses.py                   # CAVA loss definitions
  meta_reweighter.py               # MLPR/meta reweighting
  meta_utils.py                    # bi-level helper utilities
  eval_enhanced.py                 # evaluation entry
  build_physical_prior.py          # prior construction
  check_paper_consistency.py       # paper consistency audit
  export_repro_report.py           # reproducibility report export

tests/
  ...                              # unit/smoke tests

docs/
  ...                              # internal review and audit docs
```

## Data Availability

The dataset used in this project cannot be shared publicly due to data authorization and collaboration constraints.

- No raw data download is provided.
- No processed private data package is provided.

## Code Release Plan

The full codebase is **not publicly released at this stage**.

After the paper is formally accepted, we plan to release selected code and essential documentation in phases.  
The full internal pipeline and all implementation details will not be disclosed at once.
