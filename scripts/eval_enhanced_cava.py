#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StrongEvaluator with Publication-Quality Visualizations
(Fixed for EnhancedAVTopDetector output structure)
"""
import os, sys, argparse, yaml, random, numpy as np, torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             roc_auc_score, confusion_matrix, roc_curve,
                             precision_recall_curve, auc)
from sklearn.manifold import TSNE
import json
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import warnings

warnings.filterwarnings('ignore')

# 固定唯一来源
from dataset import AVFromCSV, safe_collate_fn

# 尝试导入 EnhancedAVTopDetector
try:
    from enhanced_detector import EnhancedAVTopDetector

    print("✓ Loaded EnhancedAVTopDetector from root")
except ImportError:
    try:
        from scripts.enhanced_detector import EnhancedAVTopDetector

        print("✓ Loaded EnhancedAVTopDetector from scripts")
    except ImportError:
        raise ImportError("无法找到 EnhancedAVTopDetector，请检查文件位置")

# 设置matplotlib参数
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',  # 避免 serif 字体缺失问题
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_json_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist() if obj.numel() > 1 else float(obj.item())
    elif isinstance(obj, (list, tuple)):
        return [_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


class StrongEvaluator:
    """增强版评估器 - 适配训练好的网络结构"""

    def __init__(self, cfg: dict, checkpoint_path: str, out_dir: str):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.out_dir / 'visualizations'
        self.vis_dir.mkdir(exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _set_seed(int(cfg.get("seed", 42)))

        print(f"📥 Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device)

        data_cfg = cfg["data"]
        self.C = int(data_cfg["num_classes"])
        self.num_classes = self.C
        self.class_names = list(data_cfg["class_names"])

        # 构建模型
        model_cfg = dict(cfg.get("model", {}))
        model_cfg["num_classes"] = self.C
        fusion_cfg = model_cfg.get("fusion", cfg.get("fusion", {}))

        print(f"📦 构建模型: EnhancedAVTopDetector")
        self.model = EnhancedAVTopDetector({
            "model": model_cfg,
            "fusion": fusion_cfg,
            "cava": cfg.get("cava", {}),
            "data": data_cfg  # 传入data配置以防万一
        })

        # 加载权重
        sd = ckpt.get('state_dict', ckpt)
        # 移除可能的 module. 前缀
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)

        if missing: print(f"  ⚠ Missing keys: {len(missing)} (e.g. {missing[:3]})")
        if unexpected: print(f"  ⚠ Unexpected keys: {len(unexpected)}")

        self.model.to(self.device)
        self.model.eval()

        self.has_cava = hasattr(self.model, 'cava') and self.model.cava is not None

        # 数据集
        root = data_cfg.get("data_root", "")
        vcfg = cfg.get("video", {})
        acfg = cfg.get("audio", {})
        # 优先使用 test_csv，没有则用 val_csv
        test_csv = data_cfg.get("test_csv") or data_cfg.get("val_csv")

        if not test_csv or not os.path.exists(test_csv):
            raise FileNotFoundError(f"测试集 CSV 不存在: {test_csv}")

        self.ds_test = AVFromCSV(test_csv, root, self.C, self.class_names, vcfg, acfg, is_unlabeled=False)

        self.loader = torch.utils.data.DataLoader(
            self.ds_test,
            batch_size=int(cfg.get("training", {}).get("batch_size", 16)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers_val", 4)),
            pin_memory=(self.device.type == 'cuda'),
            collate_fn=safe_collate_fn
        )

    def _unpack_batch(self, b):
        if isinstance(b, dict):
            # 简单处理大小写键
            keys = {k.lower(): k for k in b.keys()}
            v = b[keys.get('video', 'video')]
            a = b[keys.get('audio', 'audio')]
            y = b[keys.get('label', 'label')]
            ids = b.get(keys.get('ids', 'ids'))
            return v, a, y, ids
        if isinstance(b, (list, tuple)):
            v, a, y = b[:3]
            ids = b[3] if len(b) >= 4 else None
            return v, a, y, ids
        raise ValueError(f"Unsupported batch type: {type(b)}")

    @torch.no_grad()
    def evaluate(self):
        all_preds, all_labels, all_probs, all_ids = [], [], [], []
        # 特征容器
        all_video_feats, all_audio_feats, all_fused_feats = [], [], []
        all_cava_gates, all_cava_delays = [], []
        all_frame_preds, all_attention_weights = [], []
        # ---- scalar analysis containers (cheap, collect ALL samples) ----
        all_gate_mean_scalar, all_pred_delay_scalar, all_gt_delay_scalar = [], [], []

        print("\n" + "=" * 80)
        print("🔍 开始评估...")
        print("=" * 80)

        for batch_idx, batch in enumerate(tqdm(self.loader, desc="Evaluating")):
            v, a, y, ids = self._unpack_batch(batch)

            if hasattr(y, "ndim") and y.ndim == 2:
                y = y.argmax(dim=1)

            v, a, y = v.to(self.device), a.to(self.device), y.to(self.device)

            out = self.model(v, a, return_aux=True)  # 必须 return_aux=True

            # 兼容字典或直接Tensor返回
            logits = out["clip_logits"] if isinstance(out, dict) else out
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if ids is not None:
                all_ids.extend(ids)

            # ---- CAVA scalar analysis (collect ALL samples, low memory) ----
            if isinstance(out, dict):
                # GT delay via xcorr proxy on raw inputs
                try:
                    gt_delay_b = self._compute_gt_delay_batch(v, a)  # [B]
                    all_gt_delay_scalar.extend(gt_delay_b.tolist())
                except Exception as e:
                    # GT delay is optional; keep evaluation robust
                    pass

                # learned delay and gate (if available)
                if self.has_cava and ("pred_delay" in out) and (out["pred_delay"] is not None):
                    pd = out["pred_delay"].detach().float().cpu().view(-1).numpy()
                    all_pred_delay_scalar.extend(pd.tolist())
                if self.has_cava and ("causal_gate" in out) and (out["causal_gate"] is not None):
                    g = out["causal_gate"].detach().float().cpu()
                    if g.ndim == 3:
                        g = g.mean(dim=-1)  # [B,T]
                    gm = g.mean(dim=1).view(-1).numpy()
                    all_gate_mean_scalar.extend(gm.tolist())

            # --- 收集可视化数据 (仅前100个batch以节省内存) ---
            if batch_idx < 100 and isinstance(out, dict):
                # 1. 单模态特征
                if "video_proj" in out:
                    all_video_feats.append(out["video_proj"].cpu().numpy())
                elif "video_emb" in out:
                    all_video_feats.append(out["video_emb"].cpu().numpy())

                if "audio_aligned" in out:
                    all_audio_feats.append(out["audio_aligned"].cpu().numpy())
                elif "audio_emb" in out:
                    all_audio_feats.append(out["audio_emb"].cpu().numpy())

                # 2. 融合特征 (Mapping Mismatch Fix: fused_feat -> fusion_token)
                if "fusion_token" in out:
                    all_fused_feats.append(out["fusion_token"].cpu().numpy())
                elif "fused_feat" in out:
                    all_fused_feats.append(out["fused_feat"].cpu().numpy())

                # 3. CAVA 数据
                if self.has_cava:
                    if "causal_gate" in out and out["causal_gate"] is not None:
                        all_cava_gates.append(out["causal_gate"].cpu().numpy())
                    if "pred_delay" in out and out["pred_delay"] is not None:
                        all_cava_delays.append(out["pred_delay"].cpu().numpy())

                # 4. 帧级预测 (Mapping Mismatch Fix: frame_logits -> seg_logits)
                # seg_logits: [B, T, C]
                if "seg_logits" in out:
                    frame_probs = torch.softmax(out["seg_logits"], dim=-1)
                    all_frame_preds.append(frame_probs.cpu().numpy())
                elif "frame_logits" in out:
                    frame_probs = torch.softmax(out["frame_logits"], dim=-1)
                    all_frame_preds.append(frame_probs.cpu().numpy())

                # 5. 注意力权重 (Mapping Mismatch Fix: attention_weights -> weights (MIL) or aux)
                # 优先寻找 Co-Attention 的 Attention Map
                if "attn_weights" in out:
                    all_attention_weights.append(out["attn_weights"].cpu().numpy())
                elif "attention_weights" in out:
                    all_attention_weights.append(out["attention_weights"].cpu().numpy())
                elif "weights" in out:  # Fallback 到 MIL Temporal Weights
                    # MIL weights 是 [B, T]，为了可视化兼容性，我们可以 unsqueeze
                    # 或者可视化函数那边适配
                    w = out["weights"].cpu().numpy()
                    all_attention_weights.append(w)

        # 转换数据类型
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 计算指标
        results = self._compute_metrics(all_labels, all_preds, all_probs)
        self._save_results(results, all_labels, all_preds, all_probs, all_ids)

        # 生成可视化
        print("\n" + "=" * 80)
        print("📊 生成可视化报告...")
        print("=" * 80)

        self._visualize_confusion_matrix(all_labels, all_preds)
        self._visualize_roc_pr_curves(all_labels, all_probs)

        if len(all_cava_gates) > 0:
            self._visualize_cava_analysis(all_cava_gates, all_cava_delays, all_labels, all_preds)

        # ---- New: Delay consistency + Gate filtering analysis ----
        if len(all_pred_delay_scalar) > 0 and len(all_gt_delay_scalar) > 0:
            self._visualize_delay_consistency(np.array(all_pred_delay_scalar), np.array(all_gt_delay_scalar))
            if len(all_gate_mean_scalar) > 0:
                self._visualize_gate_delay_relation(
                    np.array(all_gate_mean_scalar),
                    np.array(all_pred_delay_scalar),
                    np.array(all_gt_delay_scalar)
                )
                self._visualize_gate_filter_curve(np.array(all_gate_mean_scalar), all_labels, all_preds)

        if len(all_fused_feats) > 0:
            # 截取对应长度的 label
            n_feat = sum(len(b) for b in all_fused_feats)
            self._visualize_feature_space(all_video_feats, all_audio_feats, all_fused_feats,
                                          all_labels[:n_feat])

        if len(all_frame_preds) > 0:
            self._visualize_frame_predictions(all_frame_preds, all_labels, all_preds)

        if len(all_attention_weights) > 0:
            self._visualize_attention_weights(all_attention_weights)

        print(f"\n✓ 所有可视化已保存至: {self.vis_dir}")
        return results
    # -------------------------------------------------------------------------
    # CAVA Analysis Helpers (Delay GT via xcorr + gate-quality analysis)
    # -------------------------------------------------------------------------
    def _extract_video_activity(self, v: torch.Tensor) -> np.ndarray:
        """
        Compute a simple video activity/brightness curve per sample.
        Supports:
          - [B, T, C, H, W]
          - [B, C, T, H, W]
        Returns: act_v [B, T]
        """
        v_cpu = v.detach().float().cpu()
        if v_cpu.ndim != 5:
            # fallback: flatten and treat as single-step
            B = v_cpu.shape[0]
            return np.zeros((B, 1), dtype=np.float32)

        num_frames = int(self.cfg.get("video", {}).get("num_frames", v_cpu.shape[1]))
        # try infer layout
        if v_cpu.shape[1] == num_frames:
            frames = v_cpu  # [B,T,C,H,W]
        elif v_cpu.shape[2] == num_frames:
            frames = v_cpu.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,C,H,W]
        else:
            # heuristic: smaller dim is likely T
            if v_cpu.shape[1] <= v_cpu.shape[2]:
                frames = v_cpu
            else:
                frames = v_cpu.permute(0, 2, 1, 3, 4).contiguous()

        brightness = frames.mean(dim=(2, 3, 4))  # [B,T]
        # activity = abs temporal derivative (prepend first step)
        diff = torch.cat([brightness[:, :1], (brightness[:, 1:] - brightness[:, :-1]).abs()], dim=1)
        # z-score per sample
        diff = (diff - diff.mean(dim=1, keepdim=True)) / (diff.std(dim=1, keepdim=True) + 1e-6)
        return diff.numpy()

    def _extract_audio_activity(self, a: torch.Tensor) -> np.ndarray:
        """
        Compute a simple audio energy curve per sample from the provided audio tensor.
        Supports common layouts:
          - [B, 1, F, T] (Mel)
          - [B, C, F, T] (Mel, channel-first)
          - [B, T, F] or [B, F, T]
          - [B, T] (waveform or precomputed energy)
          - [B, T, F, Tf] (YOUR case: per-video-step Mel chunk, e.g., [B, 8, 80, 64])
        Returns: act_a [B, Ta]
        """
        a_cpu = a.detach().float().cpu()
        n_mels = self.cfg.get("audio", {}).get("n_mels", None)
        num_frames = int(self.cfg.get("video", {}).get("num_frames", a_cpu.shape[1] if a_cpu.ndim > 1 else 1))

        if a_cpu.ndim == 4:
            # Case-1: per-video-step mel chunks: [B, T, n_mels, Tf]  (e.g., [B,8,80,64])
            # Detect by T matching num_frames OR by (dim2==n_mels and dim1>1)
            if (a_cpu.shape[1] == num_frames) or (n_mels is not None and a_cpu.shape[2] == n_mels and a_cpu.shape[1] > 1):
                energy = a_cpu.abs().mean(dim=(2, 3))  # -> [B, T]
            else:
                # Case-2: channel-first mel: [B, C, F, T]  (common in audio frontends)
                energy = a_cpu.abs().mean(dim=(1, 2))  # -> [B, T]
        elif a_cpu.ndim == 3:
            # infer which dim is feature (often n_mels)
            if n_mels is not None and a_cpu.shape[2] == n_mels:
                energy = a_cpu.abs().mean(dim=2)  # [B,T]
            elif n_mels is not None and a_cpu.shape[1] == n_mels:
                energy = a_cpu.abs().mean(dim=1)  # [B,T]
            else:
                # heuristic: treat the larger dim as time
                if a_cpu.shape[1] >= a_cpu.shape[2]:
                    energy = a_cpu.abs().mean(dim=2)  # [B,T=dim1]
                else:
                    energy = a_cpu.abs().mean(dim=1)  # [B,T=dim2]
        elif a_cpu.ndim == 2:
            energy = a_cpu.abs()  # [B,T]
        else:
            B = a_cpu.shape[0]
            return np.zeros((B, 1), dtype=np.float32)

        energy = (energy - energy.mean(dim=1, keepdim=True)) / (energy.std(dim=1, keepdim=True) + 1e-6)
        return energy.numpy()


    @staticmethod
    def _estimate_delay_xcorr(video_curve: np.ndarray, audio_curve: np.ndarray, max_lag: int) -> int:
        """
        Estimate delay in 'video steps' using cross-correlation:
          delay > 0 => video lags audio (video happens later), consistent with the dataset prior.
        Both inputs are 1D arrays already normalized.
        """
        Tv = len(video_curve)
        if Tv <= 1:
            return 0

        # resample audio to Tv for comparable lag units
        Ta = len(audio_curve)
        if Ta != Tv:
            x = np.linspace(0.0, 1.0, Ta)
            xi = np.linspace(0.0, 1.0, Tv)
            audio_rs = np.interp(xi, x, audio_curve)
        else:
            audio_rs = audio_curve

        # correlate(video, audio) so that positive lag => video lags audio
        corr = np.correlate(video_curve, audio_rs, mode="full")
        lags = np.arange(-len(audio_rs) + 1, len(video_curve))

        max_lag = int(max(1, min(max_lag, Tv - 1)))
        mask = (lags >= -max_lag) & (lags <= max_lag)
        lag_hat = int(lags[mask][np.argmax(corr[mask])])
        return lag_hat

    def _compute_gt_delay_batch(self, v: torch.Tensor, a: torch.Tensor) -> np.ndarray:
        """
        Compute xcorr-based GT delay per sample (mini proxy for paper's xcorr GT).
        Returns: gt_delay [B] in 'video steps'
        """
        v_act = self._extract_video_activity(v)   # [B,Tv]
        a_act = self._extract_audio_activity(a)   # [B,Ta]
        Tv = v_act.shape[1]
        max_lag = int(self.cfg.get("cava", {}).get("gt_max_lag_eval", min(20, max(1, Tv - 1))))

        gt = []
        for i in range(v_act.shape[0]):
            gt.append(self._estimate_delay_xcorr(v_act[i], a_act[i], max_lag=max_lag))
        return np.asarray(gt, dtype=np.int32)

    def _visualize_delay_consistency(self, pred_delay: np.ndarray, gt_delay: np.ndarray):
        """
        Fig-9 style:
          (a) histogram overlay  (pred vs gt)
          (b) scatter pred vs gt with y=x
          (c) error distribution box/violin
        """
        pred = np.asarray(pred_delay).reshape(-1)
        gt = np.asarray(gt_delay).reshape(-1)
        n = min(len(pred), len(gt))
        pred, gt = pred[:n], gt[:n]
        err = pred - gt

        # correlation
        r = float(np.corrcoef(pred, gt)[0, 1]) if n > 2 else float("nan")

        # (a) Histogram overlay
        plt.figure(figsize=(9, 5))
        bins = np.arange(min(pred.min(), gt.min()) - 1, max(pred.max(), gt.max()) + 2)
        plt.hist(gt, bins=bins, alpha=0.55, label="xcorr GT (δ*)", density=True)
        plt.hist(pred, bins=bins, alpha=0.55, label="CAVA learned (δ)", density=True)
        plt.xlabel("Delay (video steps)")
        plt.ylabel("Density")
        plt.title(f"Delay Distribution: learned δ vs xcorr δ*  (r={r:.3f})")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(self.vis_dir / "delay_hist_overlay.png")
        plt.close()

        # (b) Scatter
        plt.figure(figsize=(6.5, 6.0))
        plt.scatter(gt, pred, s=16, alpha=0.35)
        mn = min(gt.min(), pred.min())
        mx = max(gt.max(), pred.max())
        plt.plot([mn, mx], [mn, mx], "k--", linewidth=1.0)
        plt.xlabel("xcorr GT delay δ* (video steps)")
        plt.ylabel("learned delay δ (video steps)")
        plt.title("Delay Consistency Scatter")
        plt.tight_layout()
        plt.savefig(self.vis_dir / "delay_scatter.png")
        plt.close()

        # (c) Error distribution
        plt.figure(figsize=(7.5, 5.0))
        sns.violinplot(y=err, inner="box")
        plt.axhline(0, linestyle="--", linewidth=1.0)
        plt.ylabel("δ - δ* (video steps)")
        plt.title("Delay Error Distribution")
        plt.tight_layout()
        plt.savefig(self.vis_dir / "delay_error_violin.png")
        plt.close()

        # save stats json
        stats = {
            "n": int(n),
            "pearson_r": r,
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mean_error": float(np.mean(err)),
            "std_error": float(np.std(err)),
        }
        with open(self.out_dir / "cava_delay_stats.json", "w", encoding="utf-8") as f:
            json.dump(_to_json_serializable(stats), f, indent=2, ensure_ascii=False)

    def _visualize_gate_delay_relation(self, gate_mean: np.ndarray, pred_delay: np.ndarray, gt_delay: np.ndarray):
        """
        Show that gate correlates with alignment quality:
          - scatter gate vs |δ-δ*|
        """
        g = np.asarray(gate_mean).reshape(-1)
        pred = np.asarray(pred_delay).reshape(-1)
        gt = np.asarray(gt_delay).reshape(-1)
        n = min(len(g), len(pred), len(gt))
        g, pred, gt = g[:n], pred[:n], gt[:n]
        err = np.abs(pred - gt)

        plt.figure(figsize=(7.2, 5.5))
        plt.scatter(g, err, s=14, alpha=0.35)
        plt.xlabel("Mean gate value  $\u0305g(x)$")
        plt.ylabel("|δ - δ*|  (video steps)")
        plt.title("Gate vs Alignment Error")
        plt.tight_layout()
        plt.savefig(self.vis_dir / "gate_vs_delay_error.png")
        plt.close()

    def _visualize_gate_filter_curve(self, gate_mean: np.ndarray, labels: np.ndarray, preds: np.ndarray):
        """
        Analysis-only: if we exclude low-gate samples, accuracy/macro-F1 should increase.
        Plot retain-ratio vs metrics.
        """
        g = np.asarray(gate_mean).reshape(-1)
        y = np.asarray(labels).reshape(-1)
        p = np.asarray(preds).reshape(-1)
        n = min(len(g), len(y), len(p))
        g, y, p = g[:n], y[:n], p[:n]

        qs = np.linspace(0.0, 0.9, 10)  # keep top (1-q) samples
        retain, accs, mf1 = [], [], []
        for q in qs:
            thr = np.quantile(g, q)
            keep = g >= thr
            if keep.sum() < max(10, self.num_classes):
                continue
            yy, pp = y[keep], p[keep]
            accs.append(accuracy_score(yy, pp))
            _, _, f1m, _ = precision_recall_fscore_support(yy, pp, average="macro", zero_division=0)
            mf1.append(f1m)
            retain.append(keep.mean())

        plt.figure(figsize=(7.8, 5.3))
        plt.plot(retain, accs, marker="o", label="Accuracy")
        plt.plot(retain, mf1, marker="o", label="Macro-F1")
        plt.xlabel("Retained sample ratio")
        plt.ylabel("Metric")
        plt.title("Gate Filtering Curve (analysis)")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(self.vis_dir / "gate_filter_curve.png")
        plt.close()


    def _compute_metrics(self, labels, preds, probs):
        # 1) Accuracy
        acc = accuracy_score(labels, preds)

        # 2) Per-class P/R/F1
        p_c, r_c, f1_c, support = precision_recall_fscore_support(
            labels, preds,
            labels=list(range(self.num_classes)),
            average=None,
            zero_division=0
        )

        # 3) Macro / Weighted
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )
        p_weight, r_weight, f1_weight, _ = precision_recall_fscore_support(
            labels, preds, average='weighted', zero_division=0
        )

        # 4) Confusion matrix
        cm = confusion_matrix(labels, preds)

        # 5) 打印：每类 F1 + 整体 Accuracy/Macro-F1/Weighted-F1
        print("\n" + "=" * 80)
        print("📌 Per-class F1")
        print("=" * 80)
        for i, name in enumerate(self.class_names):
            print(f"{i:>2}  {name:<22}  F1={f1_c[i]:.4f}  (support={int(support[i])})")

        print("\n" + "=" * 80)
        print("📌 Overall metrics")
        print("=" * 80)
        print(f"Accuracy:     {acc:.4f}")
        print(f"Macro-F1:     {f1_macro:.4f}")
        print(f"Weighted-F1:  {f1_weight:.4f}")

        # 6) 返回结果（写入 eval_metrics.json 也更完整）
        results = {
            "accuracy": float(acc),
            "macro_f1": float(f1_macro),
            "weighted_f1": float(f1_weight),
            "macro_precision": float(p_macro),
            "macro_recall": float(r_macro),
            "weighted_precision": float(p_weight),
            "weighted_recall": float(r_weight),
            "per_class_f1": {self.class_names[i]: float(f1_c[i]) for i in range(self.num_classes)},
            "per_class_precision": {self.class_names[i]: float(p_c[i]) for i in range(self.num_classes)},
            "per_class_recall": {self.class_names[i]: float(r_c[i]) for i in range(self.num_classes)},
            "per_class_support": {self.class_names[i]: int(support[i]) for i in range(self.num_classes)},
            "confusion_matrix": cm.tolist(),
            "num_samples": int(len(labels)),
        }
        return results

    def _save_results(self, results, labels, preds, probs, ids):
        with open(self.out_dir / "eval_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(_to_json_serializable(results), f, indent=2, ensure_ascii=False)

        # 保存详细预测结果
        np.savez(self.out_dir / "predictions.npz",
                 labels=labels, preds=preds, probs=probs,
                 ids=np.array(ids) if ids else None)

    # --- 可视化函数群 (复用原有逻辑，增强鲁棒性) ---

    def _visualize_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.vis_dir / 'confusion_matrix.png')
        plt.close()

    def _visualize_roc_pr_curves(self, labels, probs):
        # 仅针对多分类绘制 Macro-average
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(labels, classes=range(self.num_classes))
        n_classes = y_bin.shape[1]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # ROC
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            ax1.plot(fpr, tpr, alpha=0.3)
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_title('ROC Curves (per class)')

        # PR
        for i in range(n_classes):
            p, r, _ = precision_recall_curve(y_bin[:, i], probs[:, i])
            ax2.plot(r, p, alpha=0.3)
        ax2.set_title('Precision-Recall Curves (per class)')

        plt.savefig(self.vis_dir / 'roc_pr_curves.png')
        plt.close()

    def _visualize_cava_analysis(self, gates, delays, labels, preds):
        # 扁平化
        gates = np.concatenate(gates, axis=0)  # [N, T, 1] or [N, T]
        if gates.ndim == 3: gates = gates.mean(axis=-1)
        gate_mean = gates.mean(axis=1)  # [N]

        fig, ax = plt.subplots(figsize=(8, 6))

        correct = (labels[:len(gate_mean)] == preds[:len(gate_mean)])
        data = [gate_mean[correct], gate_mean[~correct]]

        ax.boxplot(data, labels=['Correct', 'Wrong'], patch_artist=True)
        ax.set_title('CAVA Gate Values: Correct vs Wrong Predictions')
        ax.set_ylabel('Mean Gate Value')

        plt.savefig(self.vis_dir / 'cava_gate_analysis.png')
        plt.close()

    def _visualize_feature_space(self, v_feats, a_feats, f_feats, labels,
                                 hard_class_names=None, max_points=1200, seed=42):
        """
        Paper-style t-SNE:
        - no colorbar
        - legend with class names on the right
        - highlight hard classes with "*" and black edge
        - fade non-hard classes
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        from matplotlib.lines import Line2D

        if not f_feats:
            return

        # --------- prepare features ----------
        fused = np.concatenate(f_feats, axis=0)
        if fused.ndim == 3:  # [B,T,D] -> [B,D]
            fused = fused.mean(axis=1)

        labels = np.asarray(labels)
        n_total = len(fused)

        # deterministic subsample
        rng = np.random.RandomState(seed)
        n = min(n_total, max_points)
        idx = rng.choice(n_total, n, replace=False)
        X = fused[idx]
        y = labels[idx]

        # PCA -> tSNE (more stable)
        pca_dim = min(50, X.shape[1])
        Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
        X_2d = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca").fit_transform(Xp)

        # --------- hard class config ----------
        if hard_class_names is None:
            # 默认：你论文里常用的困难/少数类（按你 12 类配置）
            hard_class_names = ["Porosity", "Arc Strike", "Crack", "Burn Through"]

        hard_ids = set()
        for i, name in enumerate(self.class_names):
            if name in hard_class_names:
                hard_ids.add(i)

        # --------- colors (12 classes: use tab20 first 12) ----------
        base_colors = list(plt.cm.tab20.colors)  # >= 20 colors
        colors = base_colors[:self.num_classes]

        # --------- plot ----------
        fig, ax = plt.subplots(figsize=(13.5, 8), dpi=300)
        ax.set_title("t-SNE of Fused Features (highlighted hard classes)", pad=12)

        # leave space for legend on the right
        fig.subplots_adjust(right=0.72)

        # optional: turn off grid for clean look
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # draw non-hard first (faded), then hard (highlighted)
        for c in range(self.num_classes):
            mask = (y == c)
            if mask.sum() == 0:
                continue

            if c in hard_ids:
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    s=55, alpha=0.90,
                    c=[colors[c]],
                    edgecolors="k", linewidths=0.6
                )
            else:
                ax.scatter(
                    X_2d[mask, 0], X_2d[mask, 1],
                    s=26, alpha=0.18,
                    c=[colors[c]],
                    edgecolors="none"
                )

        # legend (with * for hard)
        legend_handles = []
        for c, name in enumerate(self.class_names):
            is_hard = (c in hard_ids)
            label = f"{name} {'*' if is_hard else ''}".rstrip()

            legend_handles.append(
                Line2D(
                    [0], [0],
                    marker='o', linestyle='None',
                    markersize=9,
                    markerfacecolor=colors[c],
                    markeredgecolor='k' if is_hard else 'none',
                    markeredgewidth=0.8 if is_hard else 0.0,
                    label=label
                )
            )

        ax.legend(
            handles=legend_handles,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            borderaxespad=0.0
        )

        # save
        out_png = self.vis_dir / "tsne_features_clean.png"
        out_pdf = self.vis_dir / "tsne_features_clean.pdf"
        plt.savefig(out_png, bbox_inches="tight", dpi=600)
        plt.savefig(out_pdf, bbox_inches="tight")
        plt.close()

    def _visualize_frame_predictions(self, frame_preds, labels, preds):
        # 可视化几个样本的帧级预测曲线
        fp = np.concatenate(frame_preds, axis=0)  # [N, T, C]

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for i in range(min(4, len(fp))):
            ax = axes[i]
            sample_pred = fp[i]  # [T, C]
            true_cls = labels[i]
            pred_cls = preds[i]

            # 画真实类别的概率曲线
            ax.plot(sample_pred[:, true_cls], label=f'True: {self.class_names[true_cls]}', color='green', marker='o')
            if true_cls != pred_cls:
                ax.plot(sample_pred[:, pred_cls], label=f'Pred: {self.class_names[pred_cls]}', color='red',
                        linestyle='--')

            ax.set_ylim(0, 1.1)
            ax.set_title(f'Sample {i} Frame Predictions')
            ax.legend()
            ax.set_xlabel('Frame Index')
            ax.set_ylabel('Probability')

        plt.tight_layout()
        plt.savefig(self.vis_dir / 'frame_predictions_sample.png')
        plt.close()

    def _visualize_attention_weights(self, attn_weights):
        # 可视化注意力权重
        # attn_weights 可能是 [N, T] (MIL) 或 [N, H, T, T] (CoAttn)
        aw = np.concatenate(attn_weights, axis=0)

        if aw.ndim == 2:  # [N, T] - MIL Attention
            # 画平均注意力分布
            mean_attn = aw.mean(axis=0)
            plt.figure(figsize=(8, 4))
            plt.plot(mean_attn, marker='o')
            plt.title('Average Temporal Attention Weights (MIL)')
            plt.xlabel('Frame Index')
            plt.ylabel('Weight')
            plt.savefig(self.vis_dir / 'attention_mil_temporal.png')
            plt.close()

        elif aw.ndim >= 3:  # [N, T, T] or [N, H, T, T]
            if aw.ndim == 4: aw = aw.mean(axis=1)  # Average heads -> [N, T, T]

            # 画平均 Cross-Attention Map
            mean_map = aw.mean(axis=0)
            plt.figure(figsize=(6, 6))
            sns.heatmap(mean_map, cmap='viridis')
            plt.title('Average Attention Map')
            plt.xlabel('Key Time')
            plt.ylabel('Query Time')
            plt.savefig(self.vis_dir / 'attention_map_2d.png')
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/eval_fixed")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    evaluator = StrongEvaluator(cfg, args.checkpoint, args.output)
    evaluator.evaluate()


if __name__ == "__main__":
    main()