# src/avtop/utils/repro.py
import os, json, hashlib, platform, subprocess, random
from pathlib import Path
import numpy as np

try:
    import torch
except Exception:
    torch = None

class ReproducibilityManager:
    """
    - set_deterministic(seed): 固定 Python/NumPy/(可用则)Torch 的随机性；设置必要的环境变量
    - snapshot(): 生成代码哈希、pip freeze、系统信息等快照文件，写到 exp_dir
    """
    def __init__(self, exp_dir: str = "./runs/snapshot"):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)

    def set_deterministic(self, seed: int = 3407):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        if torch is not None:
            try:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            except Exception:
                pass
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                # older PyTorch may not support; ignore
                pass

    def snapshot(self):
        # 1) 代码哈希
        sha = hashlib.sha256()
        for root, _, files in os.walk("."):
            for fn in files:
                if fn.endswith((".py", ".yaml", ".yml")):
                    p = Path(root) / fn
                    try:
                        sha.update(p.read_bytes())
                    except Exception:
                        pass
        (self.exp_dir / "code_sha256.txt").write_text(sha.hexdigest())

        # 2) 依赖列表
        try:
            freeze = subprocess.check_output(["python", "-m", "pip", "freeze"], text=True)
            (self.exp_dir / "requirements_freeze.txt").write_text(freeze)
        except Exception as e:
            (self.exp_dir / "requirements_freeze.txt").write_text(f"pip freeze failed: {e}")

        # 3) 系统信息
        sysinfo = {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": getattr(torch, "__version__", "NA") if torch is not None else "NA",
            "cuda": getattr(getattr(torch, "version", None), "cuda", "NA") if torch is not None else "NA",
        }
        (self.exp_dir / "system_info.json").write_text(json.dumps(sysinfo, indent=2))
        return True
