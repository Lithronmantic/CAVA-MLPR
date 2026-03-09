#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_enhanced_vizplus_full.py

✅ 输出指标：
  - Per-class: Precision / Recall / F1 / AUC / Support
  - Overall: Accuracy + Macro/Weighted Precision/Recall/F1 + Macro/Weighted AUC

✅ 输出图（若 forward 有相应输出则生成，否则跳过）：
  - confusion_matrix_clean.(png/pdf)
  - tsne_features_clean.(png/pdf)          hard classes 黑边高亮 + legend（无 colorbar）+ silhouette
  - attention_mil_correct_vs_wrong.png     MIL: Correct vs Wrong 平均曲线
  - attention_entropy_box.png              MIL: entropy 分布（越低越聚焦）
  - attention_mil_heatmap.png              MIL: 注意力热力图（按 entropy 排序）
  - cava_gate_violin_box.png               gate 分布（Correct vs Wrong）
  - cava_gate_roc.png                      gate 区分 wrong 的 ROC-AUC

运行示例：
python eval_enhanced_vizplus_full.py \
  --config configs/xxx.yaml \
  --checkpoint path/to/ckpt.pth \
  --output outputs/eval_ours25 \
  --tag "Ours (25%)" \
  --hard_classes Porosity "Arc Strike" Crack "Burn Through"
"""

import os, argparse, yaml, json, random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# ====== 你的工程依赖（与 eval_enhanced.py 一致） ======
from dataset import AVFromCSV, safe_collate_fn

try:
    from enhanced_detector import EnhancedAVTopDetector
except ImportError:
    from scripts.enhanced_detector import EnhancedAVTopDetector


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_json(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json(x) for x in obj]
    return obj


class StrongEvaluator:

    def __init__(self, cfg, checkpoint, out_dir, tag=""):
        self.cfg = cfg
        self.tag = tag
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir = self.out_dir / "visualizations"
        self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(int(cfg.get("seed", 42)))

        data_cfg = cfg["data"]
        self.num_classes = int(data_cfg["num_classes"])
        self.class_names = list(data_cfg["class_names"])

        # build model
        model_cfg = dict(cfg.get("model", {}))
        model_cfg["num_classes"] = self.num_classes
        fusion_cfg = model_cfg.get("fusion", cfg.get("fusion", {}))

        self.model = EnhancedAVTopDetector({
            "model": model_cfg,
            "fusion": fusion_cfg,
            "cava": cfg.get("cava", {}),
            "data": data_cfg
        }).to(self.device)

        ckpt = torch.load(checkpoint, map_location=self.device)
        sd = ckpt.get("state_dict", ckpt)
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        self.model.load_state_dict(sd, strict=False)
        self.model.eval()

        self.has_cava = hasattr(self.model, "cava") and self.model.cava is not None

        # dataset
        root = data_cfg.get("data_root", "")
        vcfg = cfg.get("video", {})
        acfg = cfg.get("audio", {})
        test_csv = data_cfg.get("test_csv") or data_cfg.get("val_csv")
        assert test_csv and os.path.exists(test_csv), f"CSV not found: {test_csv}"

        self.ds = AVFromCSV(test_csv, root, self.num_classes, self.class_names, vcfg, acfg, is_unlabeled=False)
        self.loader = torch.utils.data.DataLoader(
            self.ds,
            batch_size=int(cfg.get("training", {}).get("batch_size", 16)),
            shuffle=False,
            num_workers=int(data_cfg.get("num_workers_val", 4)),
            pin_memory=(self.device.type == "cuda"),
            collate_fn=safe_collate_fn
        )

    def _unpack(self, batch):
        if isinstance(batch, dict):
            keys = {k.lower(): k for k in batch.keys()}
            v = batch[keys.get("video", "video")]
            a = batch[keys.get("audio", "audio")]
            y = batch[keys.get("label", "label")]
            ids = batch.get(keys.get("ids", "ids"))
            return v, a, y, ids
        else:
            v, a, y = batch[:3]
            ids = batch[3] if len(batch) > 3 else None
            return v, a, y, ids

    @torch.no_grad()
    def evaluate(self, hard_class_names=None):
        all_labels, all_preds, all_probs, all_ids = [], [], [], []

        # vis cache
        fused_feats, attn_weights, cava_gates = [], [], []

        for bidx, batch in enumerate(tqdm(self.loader, desc=f"Evaluating {self.tag}")):
            v, a, y, ids = self._unpack(batch)
            if hasattr(y, "ndim") and y.ndim == 2:
                y = y.argmax(dim=1)

            v = v.to(self.device)
            a = a.to(self.device)
            y = y.to(self.device)

            out = self.model(v, a, return_aux=True)

            logits = out["clip_logits"] if isinstance(out, dict) and "clip_logits" in out else out
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            all_labels.extend(y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            if ids is not None:
                all_ids.extend(ids)

            # 只缓存前100个batch用于可视化（防止爆显存/内存）
            if bidx < 100 and isinstance(out, dict):
                # fused features
                if "fusion_token" in out:
                    fused_feats.append(out["fusion_token"].detach().cpu().numpy())
                elif "fused_feat" in out:
                    fused_feats.append(out["fused_feat"].detach().cpu().numpy())

                # MIL weights or attention map
                if "weights" in out and out["weights"] is not None:
                    attn_weights.append(out["weights"].detach().cpu().numpy())
                elif "attn_weights" in out and out["attn_weights"] is not None:
                    attn_weights.append(out["attn_weights"].detach().cpu().numpy())

                # cava gate
                if self.has_cava and "causal_gate" in out and out["causal_gate"] is not None:
                    cava_gates.append(out["causal_gate"].detach().cpu().numpy())

        all_labels = np.asarray(all_labels)
        all_preds = np.asarray(all_preds)
        all_probs = np.asarray(all_probs)

        results = self.compute_metrics(all_labels, all_preds, all_probs)
        self.save_results(results, all_labels, all_preds, all_probs, all_ids)

        # ===== visualizations =====
        self.plot_confusion_clean(all_labels, all_preds)
        if fused_feats:
            self.plot_tsne_clean(fused_feats, all_labels, hard_class_names=hard_class_names)
        if attn_weights:
            self.plot_attention_mil(attn_weights, all_labels, all_preds)
        if cava_gates:
            self.plot_cava_gate(cava_gates, all_labels, all_preds)

        print(f"\n✓ Outputs saved to: {self.out_dir}")
        print(f"✓ Figures saved to: {self.vis_dir}")
        return results

    # ---------------- Metrics ----------------
    def compute_metrics(self, labels, preds, probs):
        C = self.num_classes
        acc = accuracy_score(labels, preds)

        # per-class P/R/F1/support
        p_c, r_c, f1_c, s_c = precision_recall_fscore_support(
            labels, preds, average=None, labels=list(range(C)), zero_division=0
        )

        # macro / weighted
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)

        # AUC per class (OVR)
        y_bin = label_binarize(labels, classes=list(range(C)))
        auc_list = []
        for i in range(C):
            try:
                if y_bin[:, i].sum() == 0:
                    auc_i = np.nan
                else:
                    auc_i = roc_auc_score(y_bin[:, i], probs[:, i])
            except Exception:
                auc_i = np.nan
            auc_list.append(auc_i)
        auc_list = np.asarray(auc_list)

        # macro/weighted AUC
        macro_auc = np.nanmean(auc_list)
        weights = s_c / (s_c.sum() + 1e-12)
        weighted_auc = np.nansum(auc_list * weights)

        # print table
        print("\n" + "=" * 80)
        print("📌 Per-class metrics")
        print("=" * 80)
        print(f"{'ID':>2}  {'Class':<22} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8} {'Support':>8}")
        print("-" * 80)
        for i in range(C):
            print(f"{i:>2}  {self.class_names[i]:<22} {p_c[i]:8.4f} {r_c[i]:8.4f} {f1_c[i]:8.4f} {auc_list[i]:8.4f} {s_c[i]:8d}")

        print("\n" + "=" * 80)
        print("📌 Overall metrics")
        print("=" * 80)
        print(f"Accuracy:      {acc:.4f}")
        print(f"Macro-F1:      {f1_macro:.4f}   (Macro-P: {p_macro:.4f}, Macro-R: {r_macro:.4f})")
        print(f"Weighted-F1:   {f1_w:.4f}  (W-P: {p_w:.4f}, W-R: {r_w:.4f})")
        print(f"Macro-AUC:     {macro_auc:.4f}")
        print(f"Weighted-AUC:  {weighted_auc:.4f}")

        results = {
            "accuracy": acc,
            "macro_precision": p_macro,
            "macro_recall": r_macro,
            "macro_f1": f1_macro,
            "weighted_precision": p_w,
            "weighted_recall": r_w,
            "weighted_f1": f1_w,
            "macro_auc": macro_auc,
            "weighted_auc": weighted_auc,
            "per_class": [
                {
                    "id": i,
                    "name": self.class_names[i],
                    "precision": float(p_c[i]),
                    "recall": float(r_c[i]),
                    "f1": float(f1_c[i]),
                    "auc": float(auc_list[i]) if not np.isnan(auc_list[i]) else None,
                    "support": int(s_c[i]),
                }
                for i in range(C)
            ],
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            "num_samples": int(len(labels))
        }
        return results

    def save_results(self, results, labels, preds, probs, ids):
        with open(self.out_dir / "eval_metrics.json", "w", encoding="utf-8") as f:
            json.dump(to_json(results), f, indent=2, ensure_ascii=False)
        np.savez(self.out_dir / "predictions.npz",
                 labels=labels, preds=preds, probs=probs,
                 ids=np.array(ids) if ids else None)

    # ---------------- Figures ----------------
    def plot_confusion_clean(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
        mat = cm_norm * 100.0

        fig, ax = plt.subplots(figsize=(9.0, 7.2), dpi=200)
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")

        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks(np.arange(self.num_classes))
        ax.set_yticks(np.arange(self.num_classes))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(self.class_names, fontsize=9)

        for s in ax.spines.values():
            s.set_visible(False)

        ax.set_xticks(np.arange(self.num_classes + 1) - 0.5, minor=True)
        ax.set_yticks(np.arange(self.num_classes + 1) - 0.5, minor=True)
        ax.grid(which="minor", color="#D9D9D9", linewidth=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        title = f"Normalized Confusion Matrix {self.tag}".strip()
        ax.set_title(title)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Row-normalized (%)")
        cbar.set_ticks([0, 20, 40, 60, 80, 100])

        fig.tight_layout()
        fig.savefig(self.vis_dir / "confusion_matrix_clean.png", dpi=600, bbox_inches="tight")
        fig.savefig(self.vis_dir / "confusion_matrix_clean.pdf", bbox_inches="tight")
        plt.close()

    def plot_tsne_clean(self, f_feats, labels, hard_class_names=None, seed=42, max_points=1200):
        fused = np.concatenate(f_feats, axis=0)
        if fused.ndim == 3:
            fused = fused.mean(axis=1)

        labels = np.asarray(labels)[:len(fused)]
        rng = np.random.RandomState(seed)
        n = min(len(fused), max_points)
        idx = rng.choice(len(fused), n, replace=False)
        X = fused[idx]
        y = labels[idx]

        # PCA -> tSNE
        Xp = PCA(n_components=min(50, X.shape[1]), random_state=seed).fit_transform(X)
        X2 = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca").fit_transform(Xp)

        # silhouette (PCA-space)
        sil_all = np.nan
        try:
            if len(np.unique(y)) > 1:
                sil_all = silhouette_score(Xp, y)
        except Exception:
            pass

        if hard_class_names is None:
            hard_class_names = ["Porosity", "Arc Strike", "Crack", "Burn Through"]
        hard_ids = {i for i, n in enumerate(self.class_names) if n in set(hard_class_names)}

        colors = list(plt.cm.tab20.colors)[:self.num_classes]

        fig, ax = plt.subplots(figsize=(13.5, 8), dpi=300)
        fig.subplots_adjust(right=0.72)
        ax.set_title("t-SNE of Fused Features (highlighted hard classes)")
        ax.grid(False)
        ax.set_xticks([]); ax.set_yticks([])

        # non-hard first (faded)
        for c in range(self.num_classes):
            m = (y == c)
            if m.sum() == 0 or c in hard_ids:
                continue
            ax.scatter(X2[m, 0], X2[m, 1], s=26, alpha=0.18, c=[colors[c]], edgecolors="none")

        # hard highlighted
        for c in range(self.num_classes):
            if c not in hard_ids:
                continue
            m = (y == c)
            if m.sum() == 0:
                continue
            ax.scatter(X2[m, 0], X2[m, 1], s=58, alpha=0.92, c=[colors[c]],
                       edgecolors="k", linewidths=0.7)

        ax.text(0.02, 0.02, f"Silhouette(all)={sil_all:.3f}",
                transform=ax.transAxes, fontsize=11, ha="left", va="bottom")

        handles = []
        for c, name in enumerate(self.class_names):
            is_hard = c in hard_ids
            label = f"{name}{' *' if is_hard else ''}"
            handles.append(Line2D([0], [0], marker='o', linestyle='None',
                                  markersize=9, markerfacecolor=colors[c],
                                  markeredgecolor='k' if is_hard else 'none',
                                  markeredgewidth=0.8 if is_hard else 0.0,
                                  label=label))
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
                  frameon=False, borderaxespad=0.0)

        fig.savefig(self.vis_dir / "tsne_features_clean.png", dpi=600, bbox_inches="tight")
        fig.savefig(self.vis_dir / "tsne_features_clean.pdf", bbox_inches="tight")
        plt.close()

    def plot_attention_mil(self, attn_weights, labels, preds, seed=42, max_heatmap=60):
        aw = np.concatenate(attn_weights, axis=0)
        labels = np.asarray(labels)[:len(aw)]
        preds = np.asarray(preds)[:len(aw)]
        correct = (labels == preds)

        # 只处理 MIL temporal weights [N,T]
        if aw.ndim != 2:
            return

        N, T = aw.shape
        eps = 1e-12
        w = np.clip(aw, 0.0, None)
        w = w / (w.sum(axis=1, keepdims=True) + eps)

        mean_c = w[correct].mean(axis=0) if correct.any() else w.mean(axis=0)
        mean_w = w[~correct].mean(axis=0) if (~correct).any() else None

        entropy = -(w * np.log(w + eps)).sum(axis=1) / np.log(T + eps)

        # (1) Correct vs Wrong curve
        plt.figure(figsize=(8.8, 4.8), dpi=200)
        plt.plot(mean_c, marker='o', label="Correct")
        if mean_w is not None:
            plt.plot(mean_w, marker='o', label="Wrong")
        plt.title("Temporal Attention Weights (MIL): Correct vs Wrong")
        plt.xlabel("Frame Index"); plt.ylabel("Attention Weight")
        plt.legend(frameon=False); plt.tight_layout()
        plt.savefig(self.vis_dir / "attention_mil_correct_vs_wrong.png", dpi=600)
        plt.close()

        # (2) entropy box
        plt.figure(figsize=(6.8, 4.8), dpi=200)
        plt.boxplot([entropy[correct], entropy[~correct]], labels=["Correct", "Wrong"], patch_artist=True)
        plt.title("Attention Entropy (lower = sharper)")
        plt.ylabel("Normalized Entropy")
        plt.tight_layout()
        plt.savefig(self.vis_dir / "attention_entropy_box.png", dpi=600)
        plt.close()

        # (3) heatmap subset sorted by entropy
        rng = np.random.RandomState(seed)
        idx = np.arange(N)
        k = min(max_heatmap, N)
        sel = rng.choice(idx, k, replace=False)
        sel = sel[np.argsort(entropy[sel])]  # sharp -> flat
        heat = w[sel]

        plt.figure(figsize=(8.8, 4.8), dpi=200)
        sns.heatmap(heat, cmap="Blues", cbar=True)
        plt.title("MIL Attention Heatmap (sorted by entropy)")
        plt.xlabel("Frame Index"); plt.ylabel("Sample (subset)")
        plt.tight_layout()
        plt.savefig(self.vis_dir / "attention_mil_heatmap.png", dpi=600)
        plt.close()

    def plot_cava_gate(self, gates, labels, preds):
        gates = np.concatenate(gates, axis=0)
        if gates.ndim == 3:
            gates = gates.mean(axis=-1)  # [N,T]
        gate_mean = gates.mean(axis=1)

        labels = np.asarray(labels)[:len(gate_mean)]
        preds = np.asarray(preds)[:len(gate_mean)]
        correct = (labels == preds)
        wrong = (~correct)

        # violin + box
        fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=200)
        data = [gate_mean[correct], gate_mean[wrong]]
        ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
        ax.boxplot(data, widths=0.18, patch_artist=True,
                   boxprops=dict(facecolor="white", alpha=0.85),
                   medianprops=dict(color="black"))
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["Correct", "Wrong"])
        ax.set_ylabel("Mean Gate Value")
        ax.set_title("CAVA Gate Values: Correct vs Wrong")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(self.vis_dir / "cava_gate_violin_box.png", dpi=600)
        plt.close()

        # ROC-AUC: gate predicts wrong=1
        y_true = wrong.astype(int)
        if y_true.sum() > 0 and y_true.sum() < len(y_true):
            fpr, tpr, _ = roc_curve(y_true, gate_mean)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6.0, 5.2), dpi=200)
            plt.plot(fpr, tpr, lw=2, label=f"ROC-AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], "--", lw=1)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("Gate-based Error Detection (Wrong=1)")
            plt.legend(frameon=False, loc="lower right")
            plt.tight_layout()
            plt.savefig(self.vis_dir / "cava_gate_roc.png", dpi=600)
            plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/eval_vizplus")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--hard_classes", nargs="*", default=None,
                        help="hard classes to highlight in t-SNE, e.g., Porosity 'Arc Strike' Crack 'Burn Through'")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    evaluator = StrongEvaluator(cfg, args.checkpoint, args.output, tag=args.tag)
    evaluator.evaluate(hard_class_names=args.hard_classes)


if __name__ == "__main__":
    main()
