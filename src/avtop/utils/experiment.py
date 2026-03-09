# src/avtop/utils/experiment.py
from pathlib import Path
import yaml
import time

class ExperimentManager:
    def __init__(self, workdir: str = "./runs", name: str = "exp"):
        self.root = Path(workdir) / name
        self.ckpt_dir = self.root / "ckpts"
        self.log_dir  = self.root / "logs"
        self.fig_dir  = self.root / "figs"
        for d in [self.root, self.ckpt_dir, self.log_dir, self.fig_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def save_config(self, cfg):
        (self.root / "config.yaml").write_text(yaml.dump(cfg, allow_unicode=True))

    def save_metrics(self, metrics, step: int):
        (self.log_dir / f"metrics_{step}.yaml").write_text(yaml.dump(metrics))

    def save_checkpoint(self, state, step: int, fname: str = None):
        import torch
        p = self.ckpt_dir / (fname if fname else f"model_{step}.pth")
        torch.save(state, p)

    def timestamp(self):
        return time.strftime("%Y%m%d-%H%M%S")
