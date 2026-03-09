# scripts/strong_trainer.py
# -*- coding: utf-8 -*-
"""
StrongTrainer v3.8 (Graph Safe Edition)
鉁?淇 1: RuntimeError (Second Backward) - 澧炲姞 Meta Update 鐨勫紓甯告崟鑾蜂笌瀹夊叏璺宠繃
鉁?淇 2: SDPA Warning - 寮哄埗鍦?Meta Update 鏃剁鐢?Flash Attention
鉁?淇 3: NaN Guard - 淇濈暀鎵€鏈夋搴︾啍鏂笌鑷姩鎭㈠鏈哄埗
鉁?瀹屾暣鐗? 鏃犲垹鍑?
"""
import os, json, math, random, time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import contextlib

# -------------------- 鏍稿績缁勪欢瀵煎叆 --------------------
try:
    from cava_losses import CAVALoss
    from meta_reweighter import MetaReweighter, build_mlpr_features
    from meta_utils import meta_step_first_order_from_features
    from ssl_losses import ramp_up
    from history_bank import HistoryBank
    from teacher_ema import EMATeacher
    from dataset import AVFromCSV, safe_collate_fn
    from enhanced_detector import EnhancedAVTopDetector
    from training_utils import compute_ema_decay
    from config_system import resolve_runtime_config, load_paper_exact_config, audit_against_paper_exact, save_audit_summary
except ImportError:
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.cava_losses import CAVALoss
    from scripts.meta_reweighter import MetaReweighter, build_mlpr_features
    from scripts.meta_utils import meta_step_first_order_from_features
    from scripts.ssl_losses import ramp_up
    from scripts.history_bank import HistoryBank
    from scripts.teacher_ema import EMATeacher
    from scripts.dataset import AVFromCSV, safe_collate_fn
    from scripts.enhanced_detector import EnhancedAVTopDetector
    from scripts.training_utils import compute_ema_decay
    from scripts.config_system import resolve_runtime_config, load_paper_exact_config, audit_against_paper_exact, save_audit_summary

try:
    from dataset import safe_collate_fn_with_ids
except ImportError:
    try:
        from scripts.dataset import safe_collate_fn_with_ids
    except ImportError:
        def safe_collate_fn_with_ids(batch):
            return safe_collate_fn(batch)

# -------------------- AMP 娣峰悎绮惧害宸ュ叿 --------------------
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    AMP_DEVICE_ARG = True


    def amp_autocast(device_type, enabled=True, dtype=torch.float16):
        return _autocast(device_type, enabled=enabled, dtype=dtype)


    def AmpGradScaler(device_type, enabled=True):
        return _GradScaler(device_type, enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler

    AMP_DEVICE_ARG = False


    def amp_autocast(device_type, enabled=True, dtype=torch.float16):
        return _autocast(enabled=enabled)


    def AmpGradScaler(device_type, enabled=True):
        return _GradScaler(enabled=enabled)


# -------------------- 澧炲己鐗?Focal Loss (NaN 闃叉姢) --------------------
class FocalCrossEntropy(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0, class_weights=None):
        super().__init__()
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if targets.ndim == 2: targets = targets.argmax(dim=1)

        with amp_autocast('cuda', enabled=False):
            logits_f32 = torch.clamp(logits.float(), min=-30, max=30)
            ce = F.cross_entropy(
                logits_f32, targets,
                weight=self.class_weights,
                label_smoothing=self.label_smoothing,
                reduction="none"
            )
            pt = torch.exp(-ce)
            focal_weight = (1 - pt) ** self.gamma
            loss = focal_weight * ce

            if torch.isnan(loss).any() or torch.isinf(loss).any():
                return None
            return loss.mean()


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


# -------------------- 涓昏缁冨櫒绫?--------------------
class StrongTrainer:
    def __init__(self, cfg: Dict[str, Any], out_dir: str, resume_from: Optional[str] = None):
        self.cfg = resolve_runtime_config(cfg)
        cfg = self.cfg
        self.out_dir = Path(out_dir)
        (self.out_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.out_dir / 'visualizations').mkdir(parents=True, exist_ok=True)
        self.resume_from = resume_from

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        _set_seed(int(cfg.get("seed", 42)))

        # 鍒濆鍖?TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.out_dir / 'runs'))
        print(f"馃搳 TensorBoard logs: {self.out_dir / 'runs'}")

        # 鎹熷け鍘嗗彶璁板綍鍣?
        self.loss_history = {
            'sup_loss': [], 'cava_loss': [], 'cava_align': [], 'cava_edge': [],
            'pseudo_loss': [], 'total_loss': [], 'ssl_mask_ratio': [],
            'gate_mean': [], 'gate_std': [], 'learning_rate': [], 'ema_decay': [],
            'val_acc_student': [], 'val_f1_student': [],
            'val_acc_teacher': [], 'val_f1_teacher': []
        }
        self.step_losses = {
            'sup_loss': [], 'cava_loss': [], 'pseudo_loss': [], 'total_loss': []
        }

        # AMP
        self.amp_enabled = bool(cfg.get('training', {}).get('amp', True) and self.device.type == 'cuda')
        self.scaler = AmpGradScaler(self.device_type, enabled=self.amp_enabled)
        self.amp_disable_epoch = int(cfg.get("training", {}).get("amp_disable_epoch", 100))
        self.nan_count = 0
        self.consecutive_nan = 0
        self.total_steps = 0
        self.meta_fail_count = 0

        # 缁勪欢鍒濆鍖?
        self._setup_data(cfg)
        self._setup_model(cfg)
        self._setup_optimizer(cfg)
        self._setup_mlpr(cfg)
        self._setup_ssl(cfg)

        # CAVA Setup
        self.cava_cfg = dict(cfg.get("cava", {}))
        self.cava_enabled = bool(self.cava_cfg.get("enabled", False))
        self.cava_loss_fn = CAVALoss(self.cava_cfg) if self.cava_enabled else None
        self._audit_config_against_paper_exact()

        # Checkpoint Loading
        self.start_epoch = 1
        self.best_f1 = -1.0
        self.no_improve = 0

        if self.resume_from is not None:
            self._load_checkpoint(self.resume_from)

    def _setup_data(self, cfg):
        data_cfg = cfg["data"]
        self.C = int(data_cfg["num_classes"])
        self.num_classes = self.C
        self.class_names = list(data_cfg["class_names"])
        root = data_cfg.get("data_root", "")

        l_csv = data_cfg["labeled_csv"]
        v_csv = data_cfg["val_csv"]
        u_csv = data_cfg.get("unlabeled_csv")

        self.ds_l = AVFromCSV(
            l_csv, root, self.C, self.class_names,
            video_cfg=cfg.get("video"), audio_cfg=cfg.get("audio"),
            is_unlabeled=False
        )
        self.ds_v = AVFromCSV(
            v_csv, root, self.C, self.class_names,
            video_cfg=cfg.get("video"), audio_cfg=cfg.get("audio"),
            is_unlabeled=False
        )
        self.ds_u = AVFromCSV(
            u_csv, root, self.C, self.class_names,
            video_cfg=cfg.get("video"), audio_cfg=cfg.get("audio"),
            is_unlabeled=True
        ) if (cfg.get("training", {}).get("use_ssl", False) and u_csv) else None

        self.stats = self._scan_stats(self.ds_l)
        (self.out_dir / 'stats').mkdir(exist_ok=True, parents=True)
        with open(self.out_dir / 'stats' / 'class_stats.json', 'w') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        sampler = None
        if data_cfg.get("sampler", "").lower() == "weighted":
            inv_freq = np.array(self.stats["inv_freq"], dtype=np.float32)
            sampler = self._build_sampler(self.ds_l, inv_freq)

        tr = cfg.get("training", {})
        self.effective_batch_size = int(tr.get("batch_size", 16))
        runtime_bs_cfg = tr.get("runtime_batch_size", None)
        self.bs = self._resolve_runtime_batch_size(self.effective_batch_size, runtime_bs_cfg)
        self.grad_accum_steps = max(1, int(math.ceil(self.effective_batch_size / float(self.bs))))
        if self.grad_accum_steps > 1:
            print(
                f"[RUNTIME_BATCH] effective_batch_size={self.effective_batch_size}, "
                f"runtime_batch_size={self.bs}, grad_accum_steps={self.grad_accum_steps}"
            )
        pin_mem = (self.device.type == 'cuda')

        def _to(nw, default=60):
            return 0 if int(nw) == 0 else default

        self.loader_l = DataLoader(
            self.ds_l, batch_size=self.bs, sampler=sampler, shuffle=(sampler is None),
            num_workers=int(data_cfg.get("num_workers_train", 4)), pin_memory=pin_mem,
            drop_last=True, collate_fn=safe_collate_fn, timeout=_to(data_cfg.get("num_workers_train", 4)),
            persistent_workers=(int(data_cfg.get("num_workers_train", 4)) > 0)
        )
        self.loader_v = DataLoader(
            self.ds_v, batch_size=self.bs, shuffle=False,
            num_workers=int(data_cfg.get("num_workers_val", 2)), pin_memory=pin_mem,
            drop_last=False, collate_fn=safe_collate_fn, timeout=_to(data_cfg.get("num_workers_val", 2)),
            persistent_workers=(int(data_cfg.get("num_workers_val", 2)) > 0)
        )
        self.loader_u = None
        if self.ds_u is not None:
            self.loader_u = DataLoader(
                self.ds_u, batch_size=self.bs, shuffle=True,
                num_workers=int(data_cfg.get("num_workers_unl", 4)), pin_memory=pin_mem,
                drop_last=True, collate_fn=safe_collate_fn_with_ids, timeout=_to(data_cfg.get("num_workers_unl", 4)),
                persistent_workers=(int(data_cfg.get("num_workers_unl", 4)) > 0)
            )

    def _resolve_runtime_batch_size(self, target_bs: int, runtime_bs_cfg: Optional[int]) -> int:
        if runtime_bs_cfg is not None:
            return max(1, int(runtime_bs_cfg))
        if self.device.type != "cuda":
            return max(1, int(target_bs))
        try:
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            return max(1, int(target_bs))
        bs = int(target_bs)
        if mem_gb <= 8.5:
            bs = min(bs, 16)
        elif mem_gb <= 12.5:
            bs = min(bs, 32)
        elif mem_gb <= 16.5:
            bs = min(bs, 64)
        return max(1, bs)

    def _setup_model(self, cfg):
        model_cfg = dict(cfg.get("model", {}))
        model_cfg["num_classes"] = self.C
        fusion_cfg = model_cfg.get("fusion", cfg.get("fusion", {}))
        base_model = EnhancedAVTopDetector({
            "model": model_cfg,
            "fusion": fusion_cfg,
            "cava": cfg.get("cava", {}),
            "video": cfg.get("video", {}),
            "audio": cfg.get("audio", {})
        }).to(self.device)

        tr_cfg = cfg.get("training", {})
        self.multi_gpu = bool(tr_cfg.get("multi_gpu", False))
        self.world_gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
        if self.multi_gpu and self.device.type == "cuda" and self.world_gpu_count > 1:
            req = int(tr_cfg.get("num_gpus", self.world_gpu_count))
            dev_ids = list(range(min(self.world_gpu_count, max(1, req))))
            print(f"[MULTI_GPU] enabling DataParallel on GPUs: {dev_ids}")
            self.model = nn.DataParallel(base_model, device_ids=dev_ids).to(self.device)
        else:
            self.model = base_model

        if bool(cfg.get("model", {}).get("init_bias", False)):
            self._init_bias(self.model, self.stats["pi"])

    def _student_model(self) -> nn.Module:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _state_dict_for_save(self) -> Dict[str, torch.Tensor]:
        return self._student_model().state_dict()

    @staticmethod
    def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}

    def _setup_optimizer(self, cfg):
        tr = cfg.get("training", {})
        loss_cfg = cfg.get("loss", {})

        self.loss_name = loss_cfg.get("name", "ce").lower()
        cw = loss_cfg.get("class_weights", None)
        class_weights = torch.tensor(cw, dtype=torch.float32, device=self.device) if cw is not None else None

        if self.loss_name == "focal_ce":
            self.criterion = FocalCrossEntropy(
                gamma=loss_cfg.get("gamma", 2.0),
                label_smoothing=loss_cfg.get("label_smoothing", 0.05),
                class_weights=class_weights
            ).to(self.device)
        else:
            self.criterion = nn.CrossEntropyLoss(
                weight=class_weights, label_smoothing=loss_cfg.get("label_smoothing", 0.05)
            ).to(self.device)

        self.epochs = int(tr.get("num_epochs", 30))
        base_lr = float(tr.get("learning_rate", 8e-5))
        bb_mult = float(tr.get("backbone_lr_mult", 0.1))
        self.wd = float(tr.get("weight_decay", 1e-3))
        self.grad_clip = float(tr.get("grad_clip_norm", 1.0))

        head_params, bb_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if "video_backbone" in n or "audio_backbone" in n:
                bb_params.append(p)
            else:
                head_params.append(p)

        self.opt = optim.AdamW(
            [{"params": head_params, "lr": base_lr}, {"params": bb_params, "lr": base_lr * bb_mult}],
            weight_decay=self.wd
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=self.epochs, eta_min=1e-7)

    def _setup_mlpr(self, cfg):
        self.mlpr_cfg = dict(cfg.get("mlpr", {}))
        self.mlpr_enabled = bool(self.mlpr_cfg.get("enabled", False))
        self._mlpr_feature_mode = str(self.mlpr_cfg.get("feature_mode", "legacy")).lower()

        use_hist = bool(self.mlpr_cfg.get("use_history_stats", True))
        use_cava = bool(self.mlpr_cfg.get("use_cava_signal", True))
        use_prob_vec = bool(self.mlpr_cfg.get("use_prob_vector", False))

        if self._mlpr_feature_mode == "paper_7d":
            feat_dim = 7
            self._mlpr_feature_sources = [
                "max_prob",
                "entropy",
                "margin",
                "g_bar",
                "delta_prior_deviation",
                "loss_trend",
                "student_feat_norm",
            ]
        else:
            feat_dim = 3 + 1 + (2 if use_hist else 0) + (1 if use_cava else 0) + (self.C if use_prob_vec else 0)
            self._mlpr_feature_sources = [
                "max_prob",
                "entropy",
                "margin",
                "student_feat_norm",
                "history_mean",
                "history_std",
                "g_bar",
            ] + (["teacher_prob_vector"] if use_prob_vec else [])

        self.meta = MetaReweighter(
            input_dim=feat_dim, hidden=(128, 64),
            w_clip=tuple(self.mlpr_cfg.get("weight_clip", [0.05, 0.95])), dropout=0.1
        ).to(self.device) if self.mlpr_enabled else None

        self.meta_opt = optim.Adam(self.meta.parameters(),
                                   lr=float(self.mlpr_cfg.get("meta_lr", 5e-5))) if self.mlpr_enabled else None
        self.hist_bank = HistoryBank(momentum=float(self.mlpr_cfg.get("history_momentum", 0.9))) if (
                self.mlpr_enabled and use_hist) else None

        self._mlpr_flags = {"use_hist": use_hist, "use_cava": use_cava, "use_prob_vec": use_prob_vec}
        self._mlpr_lambda_u = float(self.mlpr_cfg.get("lambda_u", 0.5))
        self._mlpr_meta_interval = int(self.mlpr_cfg.get("meta_interval", 50))
        self._mlpr_inner_lr = float(self.mlpr_cfg.get("inner_lr", 1e-4))
        if self.mlpr_enabled:
            print(f"[MLPR] feature_mode={self._mlpr_feature_mode}, feature_dim={feat_dim}")
            print(f"[MLPR] feature_sources={self._mlpr_feature_sources}")

    def _setup_ssl(self, cfg):
        tr_ssl = cfg.get("training", {})
        self.use_ssl = bool(tr_ssl.get("use_ssl", False) and self.ds_u is not None)
        ssl_cfg = cfg.get("training", {}).get("ssl", {})

        self.ema_decay_base = float(ssl_cfg.get("ema_decay_base", ssl_cfg.get("ema_decay", 0.999)))
        self.ema_decay_init = float(ssl_cfg.get("ema_decay_init", self.ema_decay_base))
        self.ssl_warmup_epochs = int(ssl_cfg.get("warmup_epochs", 3))
        self.ssl_final_thresh = float(ssl_cfg.get("final_thresh", 0.85))
        self.ssl_temp = float(ssl_cfg.get("consistency_temp", 1.0))
        self.lambda_u = float(ssl_cfg.get("lambda_u", 1.0))

        self._use_dist_align = bool(ssl_cfg.get("use_dist_align", True))
        self._cls_thr = torch.full((self.C,), self.ssl_final_thresh, device=self.device)

        if self.use_ssl:
            teacher_model_cfg = dict(cfg.get("model", {}))
            teacher_model_cfg["num_classes"] = self.C
            teacher_fusion_cfg = teacher_model_cfg.get("fusion", cfg.get("fusion", {}))

            self.teacher = EnhancedAVTopDetector(
                {
                    "model": teacher_model_cfg,
                    "fusion": teacher_fusion_cfg,
                    "cava": cfg.get("cava", {}),
                    "video": cfg.get("video", {}),
                    "audio": cfg.get("audio", {})
                }
            ).to(self.device)

            self.teacher.load_state_dict(self._state_dict_for_save(), strict=False)
            for p in self.teacher.parameters(): p.requires_grad = False
            self.teacher.eval()
        else:
            self.teacher = None

        self._pi = torch.tensor(self.stats["pi"], dtype=torch.float32, device=self.device)

    def _audit_config_against_paper_exact(self):
        try:
            repo_root = Path(__file__).resolve().parents[1]
            paper_cfg = load_paper_exact_config(repo_root)
            summary = audit_against_paper_exact(self.cfg, paper_cfg)
            self.config_audit = summary

            print("[CONFIG_AUDIT] key settings:")
            for k, v in summary["current"].items():
                print(f"  - {k}: {v}")
            if summary["is_paper_exact"]:
                print("[CONFIG_AUDIT] profile matches paper_exact.")
            else:
                print(f"[CONFIG_AUDIT] differs from paper_exact: {summary['num_diffs']} item(s)")
                for row in summary["diffs"][:20]:
                    print(f"    * {row['key']}: current={row['current']} | paper_exact={row['paper_exact']}")

            save_audit_summary(self.out_dir / "stats" / "config_audit.json", summary)
            self.loss_history["config_audit_diffs"] = [float(summary["num_diffs"])]
            if hasattr(self, "writer") and self.writer is not None:
                self.writer.add_text("ConfigAudit/is_paper_exact", str(summary["is_paper_exact"]))
                self.writer.add_text("ConfigAudit/num_diffs", str(summary["num_diffs"]))
        except Exception as exc:
            self.config_audit = {"error": str(exc)}
            print(f"[CONFIG_AUDIT] skipped: {exc}")

    def _load_checkpoint(self, checkpoint_path: str):
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            print(f"鈿狅笍 Checkpoint not found: {checkpoint_path}")
            return
        print(f"馃搨 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        try:
            sd = checkpoint['state_dict']
            self.model.load_state_dict(sd, strict=False)
        except RuntimeError as e:
            try:
                sd2 = self._strip_module_prefix(checkpoint['state_dict'])
                self._student_model().load_state_dict(sd2, strict=False)
            except RuntimeError:
                print(f"鈿狅笍 Warning during model loading: {e}")

        if self.teacher is not None:
            try:
                self.teacher.load_state_dict(self._state_dict_for_save(), strict=False)
            except RuntimeError as e:
                print(f"鈿狅笍 Warning during teacher loading: {e}")

        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
        if 'best_f1' in checkpoint:
            self.best_f1 = checkpoint['best_f1']
        print(f"馃幆 Checkpoint loaded! Resuming from epoch {self.start_epoch} (Best F1: {self.best_f1:.4f})")

    def _scan_stats(self, ds_l) -> Dict[str, Any]:
        C = self.C
        counts = np.zeros(C, dtype=np.int64)
        n = len(ds_l)
        for i in range(n):
            try:
                item = ds_l.rows[i]
                idx = item.get("label_idx")
                if idx is not None and 0 <= idx < C:
                    counts[idx] += 1
            except Exception:
                continue
        total = counts.sum()
        pi = (counts / total) if total > 0 else np.ones(C, dtype=np.float32) / C
        inv = 1.0 / np.clip(counts.astype(np.float32), 1.0, None)
        inv = inv / inv.mean()
        return {
            "counts": counts.tolist(),
            "pi": pi.astype(np.float32).tolist(),
            "inv_freq": inv.astype(np.float32).tolist(),
            "total": int(total),
        }

    def _build_sampler(self, ds_l, inv_freq):
        weights = []
        for r in ds_l.rows:
            idx = int(r.get("label_idx", 0))
            if 0 <= idx < len(inv_freq):
                weights.append(inv_freq[idx])
            else:
                weights.append(1.0)
        return WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights))

    def _init_bias(self, model, pi):
        pi_tensor = torch.tensor(pi, dtype=torch.float32, device=self.device)

        def _try_set_bias(linear: nn.Linear):
            if isinstance(linear, nn.Linear) and linear.bias is not None:
                with torch.no_grad():
                    log_pi = torch.log(torch.clamp(pi_tensor, min=1e-8)).to(linear.bias.device)
                    linear.bias.copy_(log_pi)
                return True
            return False

        if hasattr(model, 'mil_head') and hasattr(model.mil_head, 'frame_classifier'):
            for m in reversed(list(model.mil_head.frame_classifier)):
                if _try_set_bias(m): break
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            _try_set_bias(model.classifier)

    # v3.4 鏂板锛氭鏌ユā鍨嬫槸鍚﹀凡缁忔崯鍧?(鍏ㄦ槸 NaN)
    def _check_model_health(self):
        for name, param in self.model.named_parameters():
            if not torch.isfinite(param).all():
                print(f"馃拃 Model corrupted at layer: {name}")
                return False
        return True

    # v3.4 鏂板锛氳嚜鍔ㄥ洖婊氫笌闄嶇骇
    def _perform_auto_recovery(self):
        print("\n馃殤 [Auto-Recovery] Model poisoning detected!")
        print("馃攧 Rolling back to best_f1.pth...")

        ckpt_path = self.out_dir / 'checkpoints' / 'best_f1.pth'
        if not ckpt_path.exists():
            ckpt_path = self.out_dir / 'checkpoints' / 'latest.pth'

        if not ckpt_path.exists():
            print("鉂?No checkpoint found. Cannot recover. Aborting.")
            raise RuntimeError("Model collapsed and no checkpoint to restore.")

        self._load_checkpoint(str(ckpt_path))

        print("馃搲 Degrading training mode for stability:")
        print("   1. Disabling AMP (FP16 -> FP32)")
        self.amp_enabled = False
        self.scaler = AmpGradScaler(self.device_type, enabled=False)

        print("   2. Reducing Learning Rate by 50%")
        for param_group in self.opt.param_groups:
            param_group['lr'] *= 0.5

        self.consecutive_nan = 0
        self.nan_count = 0
        print("鉁?Recovery complete. Resuming training in Safe Mode.\n")

    def _reset_scaler_if_needed(self):
        if self.scaler.is_enabled():
            self.scaler = AmpGradScaler(self.device_type, enabled=True)
        self.opt.zero_grad(set_to_none=True)
        self.consecutive_nan += 1

        # v3.4: 杩炵画瑙﹀彂鐔旀柇瓒呰繃 5 娆★紝鎴栬€呮ā鍨嬪凡缁?NaN锛岃Е鍙戣嚜鍔ㄥ洖婊?
        if self.consecutive_nan >= 5 or not self._check_model_health():
            self._perform_auto_recovery()

    def _ema_update(self, epoch: int):
        if self.teacher is None: return
        stu = self._student_model()
        ema_now = compute_ema_decay(
            epoch=epoch,
            ema_decay_base=self.ema_decay_base,
            ema_decay_init=self.ema_decay_init,
            warmup_epochs=self.ssl_warmup_epochs
        )
        with torch.no_grad():
            for t_p, s_p in zip(self.teacher.parameters(), stu.parameters()):
                # v3.6 Fixed s_param -> s_p
                t_p.data.mul_(ema_now).add_(s_p.data, alpha=1.0 - ema_now)
            for t_b, s_b in zip(self.teacher.buffers(), stu.buffers()):
                t_b.copy_(s_b)
        self._last_ema_decay = ema_now

    # ============================================================================
    # 淇敼1锛氬湪 strong_trainer.py 涓壘鍒?_meta_update_step 鏂规硶锛堝ぇ绾︾590琛岋級
    # 瀹屾暣鏇挎崲杩欎釜鏂规硶
    # ============================================================================

    def _meta_update_step(self, step_count: int):
        """v3.9: 瀹屽叏闅旂鐨勫厓瀛︿範鏇存柊"""
        if not self.mlpr_enabled or self.meta is None or self.meta_opt is None:
            return

        try:
            # 馃敶 鍏抽敭淇1锛氫繚瀛樺苟鍒囨崲妯″瀷鐘舵€?
            was_training = self.model.training
            self.model.eval()  # 鍒囨崲鍒拌瘎浼版ā寮忥紝闃叉BatchNorm鍒涘缓璁＄畻鍥?

            # 馃敶 鍏抽敭淇2锛氭竻绌烘墍鏈夋搴?
            self.model.zero_grad(set_to_none=True)
            self.opt.zero_grad(set_to_none=True)
            self.meta_opt.zero_grad(set_to_none=True)

            # 1. 鍑嗗楠岃瘉闆嗘暟鎹?
            with torch.no_grad():
                try:
                    val_iter = getattr(self, '_val_iter_for_meta', None)
                    if val_iter is None:
                        val_iter = iter(self.loader_v)
                        self._val_iter_for_meta = val_iter
                    val_batch = next(val_iter)
                except StopIteration:
                    self._val_iter_for_meta = iter(self.loader_v)
                    val_batch = next(self._val_iter_for_meta)

                if len(val_batch) == 4:
                    v_val, a_val, y_val, _ = val_batch
                else:
                    v_val, a_val, y_val = val_batch

                v_val = v_val.to(self.device).float()
                a_val = a_val.to(self.device).float()
                y_val = y_val.argmax(dim=1).to(self.device) if y_val.ndim == 2 else y_val.to(self.device)

                if not hasattr(self, '_last_train_batch') or not hasattr(self, '_last_w_features'):
                    if was_training:
                        self.model.train()
                    return

                v_tr, a_tr, y_tr = self._last_train_batch
                w_features = self._last_w_features

            # 2. 鎵ц鍏冨涔犳洿鏂帮紙闅旂鐨勭幆澧冿級
            try:
                with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True):
                    meta_logs = self._simple_meta_step(
                        v_tr=v_tr, a_tr=a_tr, yhat_tr=y_tr, w_features=w_features,
                        v_val=v_val, a_val=a_val, y_val=y_val
                    )

                if step_count % 100 == 0:
                    self.writer.add_scalar('Meta/Val_Loss_Approx', meta_logs["meta_val_loss"], step_count)
                    self.writer.add_scalar('Meta/Train_Loss_Approx', meta_logs["meta_train_loss"], step_count)
                    self.writer.add_scalar('Meta/W_Mean', meta_logs["w_mean"], step_count)
                    self.writer.add_scalar('Meta/W_Std', meta_logs["w_std"], step_count)
                    self.writer.add_scalar('Meta/W_Min', meta_logs["w_min"], step_count)
                    self.writer.add_scalar('Meta/W_Max', meta_logs["w_max"], step_count)

            finally:
                # 馃敶 鍏抽敭淇3锛氬畬鍏ㄦ竻鐞?
                self.model.zero_grad(set_to_none=True)
                self.opt.zero_grad(set_to_none=True)
                self.meta_opt.zero_grad(set_to_none=True)

                # 娓呯悊CUDA缂撳瓨
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # 馃敶 鍏抽敭淇4锛氭仮澶嶈缁冩ā寮?
                if was_training:
                    self.model.train()

        except Exception as e:
            self.meta_fail_count += 1
            if self.meta_fail_count < 10:
                print(f"鈿狅笍 [鍏冨涔犳洿鏂拌烦杩嘳 {e}")

            # 绱ф€ユ竻鐞?
            self.model.zero_grad(set_to_none=True)
            self.opt.zero_grad(set_to_none=True)
            if self.meta_opt is not None:
                self.meta_opt.zero_grad(set_to_none=True)

    def _simple_meta_step(
            self,
            v_tr: torch.Tensor,
            a_tr: torch.Tensor,
            yhat_tr: torch.Tensor,
            w_features: torch.Tensor,
            v_val: torch.Tensor,
            a_val: torch.Tensor,
            y_val: torch.Tensor) -> Dict[str, float]:
        """瀹屾暣涓€闃惰繎浼煎弻灞備紭鍖栭棴鐜細weights -> inner step -> val step -> update meta."""
        return meta_step_first_order_from_features(
            student_model=self._student_model(),
            meta_net=self.meta,
            meta_opt=self.meta_opt,
            w_features=w_features,
            v_tr=v_tr,
            a_tr=a_tr,
            yhat_tr=yhat_tr,
            v_val=v_val,
            a_val=a_val,
            y_val=y_val,
            lr_inner=self._mlpr_inner_lr,
        )

    def _save_loss_history(self):
        with open(self.out_dir / 'loss_history.json', 'w') as f:
            json.dump(self.loss_history, f, indent=2)

    def _plot_all_visualizations(self):
        """鎭㈠鎵€鏈夊彲瑙嗗寲鍔熻兘"""
        viz_dir = self.out_dir / 'visualizations'
        self._plot_main_losses(viz_dir / 'main_losses.png')
        self._plot_cava_details(viz_dir / 'cava_details.png')
        self._plot_validation_metrics(viz_dir / 'validation_metrics.png')
        self._plot_training_dynamics(viz_dir / 'training_dynamics.png')
        if len(self.step_losses['total_loss']) > 0:
            self._plot_smooth_step_curves(viz_dir / 'smooth_step_losses.png')

    def _plot_main_losses(self, save_path):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Training Loss Curves Overview', fontsize=18, fontweight='bold', y=0.995)
        epochs = range(1, len(self.loss_history['total_loss']) + 1)

        axes[0, 0].plot(epochs, self.loss_history['total_loss'], 'b-', label='Total Loss')
        axes[0, 0].set_title('Total Loss');
        axes[0, 0].legend()

        axes[0, 1].plot(epochs, self.loss_history['sup_loss'], 'g-', label='Supervised Loss')
        axes[0, 1].set_title('Supervised Loss');
        axes[0, 1].legend()

        axes[0, 2].plot(epochs, self.loss_history['cava_loss'], 'r-', label='CAVA Loss')
        axes[0, 2].set_title('CAVA Loss');
        axes[0, 2].legend()

        axes[1, 0].plot(epochs, self.loss_history['cava_align'], 'orange', label='Align Loss')
        axes[1, 0].plot(epochs, self.loss_history['cava_edge'], 'purple', label='Edge Loss')
        axes[1, 0].set_title('CAVA Components');
        axes[1, 0].legend()

        axes[1, 1].plot(epochs, self.loss_history['pseudo_loss'], 'cyan', label='Pseudo Loss')
        axes[1, 1].set_title('Pseudo Label Loss');
        axes[1, 1].legend()

        if len(self.loss_history['gate_mean']) > 0:
            axes[1, 2].plot(epochs, self.loss_history['gate_mean'], 'magenta', label='Gate Mean')
            axes[1, 2].set_title('Causal Gate');
            axes[1, 2].legend()

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _plot_cava_details(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('CAVA Detailed Analysis', fontsize=16)
        epochs = range(1, len(self.loss_history['total_loss']) + 1)

        axes[0, 0].plot(epochs, self.loss_history['cava_align'], 'orange', marker='o')
        axes[0, 0].set_title('InfoNCE Alignment Loss')

        axes[0, 1].plot(epochs, self.loss_history['cava_edge'], 'purple', marker='s')
        axes[0, 1].set_title('Edge Hinge Loss')

        if len(self.loss_history['gate_std']) > 0:
            mean_vals = np.array(self.loss_history['gate_mean'])
            std_vals = np.array(self.loss_history['gate_std'])
            axes[1, 0].plot(epochs, mean_vals, 'b-', label='Mean')
            axes[1, 0].fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, alpha=0.3, label='卤1 Std')
            axes[1, 0].set_title('Causal Gate Statistics')
            axes[1, 0].legend()

        axes[1, 1].plot(epochs, self.loss_history['cava_loss'], 'r-')
        axes[1, 1].set_title('Total CAVA Loss')

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _plot_validation_metrics(self, save_path):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        epochs = range(1, len(self.loss_history['val_f1_student']) + 1)

        axes[0].plot(epochs, self.loss_history['val_f1_student'], 'b-', marker='o', label='Student')
        axes[0].plot(epochs, self.loss_history['val_f1_teacher'], 'r--', marker='s', label='Teacher')
        axes[0].set_title('F1 Score (Macro)');
        axes[0].legend()

        if len(self.loss_history['val_acc_student']) > 0:
            axes[1].plot(epochs, self.loss_history['val_acc_student'], 'b-', marker='o', label='Student')
            axes[1].plot(epochs, self.loss_history['val_acc_teacher'], 'r--', marker='s', label='Teacher')
            axes[1].set_title('Accuracy');
            axes[1].legend()

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _plot_training_dynamics(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        epochs = range(1, len(self.loss_history['learning_rate']) + 1)

        axes[0, 0].plot(epochs, self.loss_history['learning_rate'], 'g-')
        axes[0, 0].set_title('Learning Rate');
        axes[0, 0].set_yscale('log')

        if len(self.loss_history['ssl_mask_ratio']) > 0:
            axes[0, 1].plot(epochs, self.loss_history['ssl_mask_ratio'], 'c-')
            axes[0, 1].set_title('SSL Mask Ratio')

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _plot_smooth_step_curves(self, save_path):
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        window = 50

        def smooth(d): return np.convolve(d, np.ones(window) / window, mode='valid') if len(d) > window else d

        if len(self.step_losses['total_loss']) > 0:
            axes[0, 0].plot(smooth(self.step_losses['total_loss']), 'b-', alpha=0.8)
            axes[0, 0].set_title('Total Loss (Step)')

            axes[0, 1].plot(smooth(self.step_losses['sup_loss']), 'g-', alpha=0.8)
            axes[0, 1].set_title('Supervised Loss (Step)')

            axes[1, 0].plot(smooth(self.step_losses['cava_loss']), 'r-', alpha=0.8)
            axes[1, 0].set_title('CAVA Loss (Step)')

            axes[1, 1].plot(smooth(self.step_losses['pseudo_loss']), 'c-', alpha=0.8)
            axes[1, 1].set_title('Pseudo Loss (Step)')

        plt.tight_layout();
        plt.savefig(save_path);
        plt.close()

    def _forward_model(self, model, v, a, *, return_aux=True, use_amp=True):
        if use_amp and self.amp_enabled:
            with amp_autocast(self.device_type, enabled=True):
                return model(v, a, return_aux=return_aux)
        return model(v, a, return_aux=return_aux)

    def _safe_forward(self, v, a, use_amp=True):
        try:
            return self._forward_model(self.model, v, a, return_aux=True, use_amp=use_amp)
        except RuntimeError as e:
            if "NaN" in str(e):
                self.nan_count += 1
                self._reset_scaler_if_needed()
                return None
            raise e

    def _forward(self, v, a, use_amp=True):
        return self._safe_forward(v, a, use_amp=use_amp)

    @torch.no_grad()
    def _validate(self, epoch: int):
        def _eval(m):
            m.eval()
            all_y, all_p = [], []
            for b in self.loader_v:
                if len(b) == 4:
                    v, a, y, _ = b
                else:
                    v, a, y = b
                v, a = v.to(self.device), a.to(self.device)
                y = y.argmax(dim=1) if y.ndim == 2 else y
                out = self._forward_model(m, v, a, return_aux=True, use_amp=False)
                logits = out['clip_logits'] if isinstance(out, dict) else out
                all_p.append(F.softmax(logits, dim=1).cpu().numpy())
                all_y.append(y.cpu().numpy())

            if len(all_y) == 0: return {"acc": 0.0, "f1_macro": 0.0}

            y_true = np.concatenate(all_y)
            y_prob = np.concatenate(all_p)
            y_pred = y_prob.argmax(1)
            from sklearn.metrics import accuracy_score, f1_score
            return {"acc": accuracy_score(y_true, y_pred), "f1_macro": f1_score(y_true, y_pred, average='macro')}

        stu = _eval(self.model)
        tea = _eval(self.teacher) if self.teacher else {"acc": 0, "f1_macro": 0}

        self.loss_history['val_f1_student'].append(stu['f1_macro'])
        self.loss_history['val_f1_teacher'].append(tea['f1_macro'])
        self.loss_history['val_acc_student'].append(stu['acc'])
        self.loss_history['val_acc_teacher'].append(tea['acc'])

        self.writer.add_scalar('Val/F1_Student', stu['f1_macro'], epoch)
        self.writer.add_scalar('Val/F1_Teacher', tea['f1_macro'], epoch)
        self.writer.add_scalar('Val/Acc_Student', stu['acc'], epoch)

        return {"student": stu, "teacher": tea}

    def train(self):
        print("\n" + "=" * 60)
        print("馃幆 Starting Training (v3.8 Graph Safe - Full Source)...")
        print("=" * 60 + "\n")

        for epoch in range(self.start_epoch, self.epochs + 1):
            tr_metrics = self._train_epoch(epoch)
            val_res = self._validate(epoch)
            self.scheduler.step()

            if self.teacher is not None:
                ema_epoch = compute_ema_decay(
                    epoch=epoch,
                    ema_decay_base=self.ema_decay_base,
                    ema_decay_init=self.ema_decay_init,
                    warmup_epochs=self.ssl_warmup_epochs,
                )
                self.loss_history['ema_decay'].append(float(ema_epoch))
                self.writer.add_scalar('SSL/ema_decay_epoch', float(ema_epoch), epoch)
                print(
                    f"[EMA] epoch={epoch} decay={ema_epoch:.6f} "
                    f"(init={self.ema_decay_init:.6f}, base={self.ema_decay_base:.6f})"
                )

            current_lr = self.opt.param_groups[0]['lr']
            self.loss_history['learning_rate'].append(current_lr)
            self.writer.add_scalar('Train/learning_rate', current_lr, epoch)
            self.writer.add_scalar('Train/Loss', tr_metrics['total'], epoch)

            f1_stu = val_res["student"]["f1_macro"]
            if f1_stu > getattr(self, "best_f1", -1.0):
                self.best_f1 = f1_stu
                save_dict = {
                    "epoch": epoch,
                    "state_dict": self._state_dict_for_save(),
                    "best_f1": self.best_f1
                }
                torch.save(save_dict, self.out_dir / 'checkpoints' / 'best_f1.pth')
                self.no_improve = 0
                print(f"鉁?New best Student F1: {self.best_f1:.4f}")
            else:
                self.no_improve += 1

            torch.save({
                "epoch": epoch,
                "state_dict": self._state_dict_for_save()
            }, self.out_dir / 'checkpoints' / 'latest.pth')

            print(f"[Epoch {epoch}/{self.epochs}] Loss={tr_metrics['total']:.4f} | "
                  f"Val Stu F1={f1_stu:.4f} | Val Tea F1={val_res['teacher']['f1_macro']:.4f}")

            patience = int(self.cfg.get("training", {}).get("early_stop_patience", 10))
            if patience > 0 and self.no_improve >= patience:
                print(f"馃洃 Early stopping triggered after {patience} epochs.")
                break

        self.writer.close()
        self._save_loss_history()
        self._plot_all_visualizations()
        print("\n鉁?Training complete!")

    def _train_epoch(self, epoch: int):
        if epoch >= self.amp_disable_epoch and self.amp_enabled:
            self.amp_enabled = False
            self.scaler = AmpGradScaler(self.device_type, enabled=False)

        self.model.train()
        if self.teacher: self.teacher.eval()

        epoch_losses = {k: 0.0 for k in
                        ['sup_loss', 'cava_loss', 'pseudo_loss', 'total_loss', 'ssl_mask_ratio', 'gate_mean',
                         'gate_std']}
        tot = 0.0;
        nb = 0

        ssl_active = self.use_ssl and (epoch > self.ssl_warmup_epochs)
        u_iter = iter(self.loader_u) if ssl_active else None

        pbar = tqdm(self.loader_l, desc=f"Epoch {epoch}/{self.epochs}")
        self.opt.zero_grad(set_to_none=True)
        num_batches = len(self.loader_l)
        for bi, b in enumerate(pbar):
            if isinstance(b, (list, tuple)) and len(b) == 4:
                v, a, y, _ = b
            else:
                v, a, y = b

            v, a = v.to(self.device), a.to(self.device)
            y = y.argmax(dim=1).to(self.device) if y.ndim == 2 else y.to(self.device)

            # ---------------- 1. Supervised Forward ----------------
            with amp_autocast(self.device_type, enabled=self.amp_enabled):
                out = self._safe_forward(v, a, use_amp=self.amp_enabled)
                if out is None: continue

                if torch.isnan(out['clip_logits']).any():
                    self.nan_count += 1;
                    self._reset_scaler_if_needed();
                    continue

                sup_loss = self.criterion(out['clip_logits'], y)

                if sup_loss is None or torch.isnan(sup_loss):
                    self.nan_count += 1;
                    self._reset_scaler_if_needed();
                    continue

                cava_loss = torch.tensor(0.0, device=self.device)
                if self.cava_enabled:
                    try:
                        g = out.get("causal_gate", None)
                        self.cava_loss_fn.update_cfg(self.cava_cfg)
                        c_logs = self.cava_loss_fn(out)
                        cava_loss = c_logs["loss_total"]
                        epoch_losses['cava_align'] += c_logs["loss_align"].detach().item()
                        epoch_losses['cava_edge'] += c_logs["loss_edge"].detach().item()
                        if g is not None:
                            epoch_losses['gate_mean'] += g.mean().item()
                            epoch_losses['gate_std'] += g.std().item()
                    except Exception:
                        pass

                epoch_losses['sup_loss'] += sup_loss.item()
                epoch_losses['cava_loss'] += cava_loss.item()

                if self.mlpr_enabled:
                    with torch.no_grad():
                        self._last_train_batch = (v.detach(), a.detach(), y.detach())

                # ---------------- 2. SSL / MLPR Forward ----------------
                pseudo_loss = torch.tensor(0.0, device=self.device)
                if ssl_active:
                    try:
                        bu = next(u_iter)
                    except StopIteration:
                        u_iter = iter(self.loader_u);
                        bu = next(u_iter)

                    if len(bu) == 4:
                        vu, au, _, ids_u = bu
                    else:
                        vu, au, _ = bu; ids_u = None
                    vu, au = vu.to(self.device), au.to(self.device)

                    with torch.no_grad():
                        tout = self.teacher(vu, au)
                        t_logits = tout['clip_logits']
                        t_prob = F.softmax(t_logits, dim=1)

                        if self._use_dist_align:
                            q = t_prob.mean(dim=0).clamp(min=1e-8)
                            p_target = self._pi / (q + 1e-8)
                            p_target = p_target / p_target.sum()
                            t_prob = t_prob * p_target.unsqueeze(0)
                            t_prob = t_prob / t_prob.sum(dim=1, keepdim=True)

                        t_max, t_idx = t_prob.max(dim=1)
                        mask = t_max.ge(self.ssl_final_thresh).float()

                    sout_u = self.model(vu, au)
                    s_logits_u = sout_u['clip_logits']

                    if torch.isnan(s_logits_u).any():
                        self.nan_count += 1;
                        self._reset_scaler_if_needed();
                        continue

                    # Replace lines ~820-892 in strong_trainer.py with this fixed version:

                    w_eff = mask
                    if self.mlpr_enabled and self.meta is not None:
                        try:
                            # 1. Extract features
                            stu_feat = None
                            if 'fusion_token' in sout_u:
                                ftok = sout_u['fusion_token']
                                stu_feat = ftok.mean(dim=tuple(range(1, ftok.dim()))) if ftok.dim() > 2 else ftok
                            else:
                                v_emb = sout_u.get('video_emb', None)
                                a_emb = sout_u.get('audio_emb', None)
                                if v_emb is not None and a_emb is not None:
                                    if v_emb.dim() > 2: v_emb = v_emb.mean(dim=1)
                                    if a_emb.dim() > 2: a_emb = a_emb.mean(dim=1)
                                    stu_feat = torch.cat([v_emb, a_emb], dim=-1)

                            hist_mu, hist_std = None, None
                            if self.hist_bank is not None and ids_u is not None:
                                id_list = ids_u.cpu().tolist() if torch.is_tensor(ids_u) else ids_u
                                h_mu, h_sd = self.hist_bank.query([int(x) for x in id_list])
                                hist_mu = h_mu.to(self.device)
                                hist_std = h_sd.to(self.device)

                            cava_gate_mean = None
                            if 'causal_gate' in sout_u and sout_u['causal_gate'] is not None:
                                cg = sout_u['causal_gate']
                                cava_gate_mean = cg.mean(dim=tuple(range(1, cg.dim()))).view(-1, 1)

                            # Build features (detach all inputs)
                            feats = build_mlpr_features(
                                teacher_prob=t_prob.detach(),
                                student_feat=stu_feat.detach() if stu_feat is not None else None,
                                history_mean=hist_mu,
                                history_std=hist_std,
                                cava_gate_mean=cava_gate_mean.detach() if cava_gate_mean is not None else None,
                                use_prob_vector=self._mlpr_flags["use_prob_vec"],
                                feature_mode=self._mlpr_feature_mode,
                                delay_frames=sout_u.get("delay_frames_cont", sout_u.get("delay_frames", None)),
                                delta_prior=float(self.cava_cfg.get("delta_prior", 0.0)),
                                loss_trend=hist_mu,
                            )

                            # 2. 鉁?FIX: Meta Network Forward in no_grad context
                            # This prevents creating a graph that will conflict with meta_update later
                            with torch.no_grad():
                                w = self.meta(feats)

                            # 3. Soft-Start Reweighting
                            w_eff = (w * mask.unsqueeze(1) + 0.3).clamp(0.0, 1.0)

                            # 4. Cache for Meta Update (must be in no_grad context)
                            with torch.no_grad():
                                self._last_train_batch = (v.detach(), a.detach(), t_idx.detach())
                                self._last_w_features = feats.detach()

                            # 5. History Update
                            if self.hist_bank is not None and ids_u is not None:
                                with torch.no_grad():
                                    s_log_prob = F.log_softmax(s_logits_u, dim=1)
                                    kl = F.kl_div(s_log_prob, t_prob.detach(), reduction='none').sum(dim=1)
                                    self.hist_bank.update(id_list, kl.detach())

                        except Exception as e:
                            # Fallback to simple masking if anything goes wrong
                            w_eff = mask
                            if self.meta_fail_count < 10:
                                print(f"鈿狅笍 [MLPR Weight Generation Failed] {e}")
                                self.meta_fail_count += 1

                    if w_eff.sum() > 0:
                        loss_u_elem = F.cross_entropy(s_logits_u, t_idx, reduction='none')
                        if w_eff.dim() > 1: w_eff = w_eff.squeeze(1)
                        pseudo_loss = (loss_u_elem * w_eff).mean()
                        epoch_losses['ssl_mask_ratio'] += (w_eff > 0).float().mean().item()

                epoch_losses['pseudo_loss'] += pseudo_loss.item()
                loss = sup_loss + cava_loss + self.lambda_u * pseudo_loss
                epoch_losses['total_loss'] += loss.item()
                loss_scaled = loss / float(self.grad_accum_steps)
                do_step = ((bi + 1) % self.grad_accum_steps == 0) or ((bi + 1) == num_batches)

                # ---------------- 3. Optimization with Guard ----------------
                if self.scaler.is_enabled():
                    self.scaler.scale(loss_scaled).backward()
                    if do_step:
                        self.scaler.unscale_(self.opt)
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            self.nan_count += 1
                            self._reset_scaler_if_needed()
                            continue
                        self.scaler.step(self.opt)
                        self.scaler.update()
                else:
                    loss_scaled.backward()
                    if do_step:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            self.nan_count += 1
                            self.opt.zero_grad()
                            continue
                        self.opt.step()

                self.consecutive_nan = 0
                if do_step:
                    self.total_steps += 1
                    self._ema_update(epoch)
                    if (self.total_steps % 100 == 0) and hasattr(self, "_last_ema_decay"):
                        self.writer.add_scalar("SSL/ema_decay", float(self._last_ema_decay), self.total_steps)
                        print(f"[EMA] step={self.total_steps} epoch={epoch} decay={self._last_ema_decay:.6f}")

                    # Meta Update (With SDPA Fix)
                    if self.mlpr_enabled and (self.total_steps % self._mlpr_meta_interval == 0):
                        self.opt.zero_grad(set_to_none=True)
                        self._meta_update_step(self.total_steps)
                        self.opt.zero_grad(set_to_none=True)
                        self.model.zero_grad(set_to_none=True)

                    self.opt.zero_grad(set_to_none=True)

                tot += loss.item()
                nb += 1
                pbar.set_postfix(loss=f"{loss.item():.4f}", nan=self.nan_count)

        for k in epoch_losses: epoch_losses[k] /= max(1, nb)
        return {"total": epoch_losses['total_loss'], "sup": epoch_losses['sup_loss']}


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--output', type=str, default='./outputs/train_v3')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()

    with open(args.config, 'r') as f: cfg = yaml.safe_load(f)
    trainer = StrongTrainer(cfg, args.output, args.checkpoint)
    trainer.train()

