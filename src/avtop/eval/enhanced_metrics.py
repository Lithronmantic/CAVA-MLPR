# src/avtop/eval/enhanced_metrics.py
"""
改进的评估指标模块
重点：少数类AUPRC、MCC、带baseline对比
"""
import numpy as np
from sklearn import metrics
from typing import Dict, Tuple


class ImbalancedMetricsCalculator:
    """专门针对不平衡数据的指标计算器"""

    def __init__(self, minority_class: int = 1):
        """
        Args:
            minority_class: 少数类/关注类的标签（通常是异常类=1）
        """
        self.minority_class = minority_class

    def compute_all_metrics(self, y_true: np.ndarray, y_probs: np.ndarray,
                            threshold: float = 0.5) -> Dict[str, float]:
        """
        计算所有关键指标

        Args:
            y_true: 真实标签 (N,)
            y_probs: 预测概率 (N,) - 正类的概率
            threshold: 分类阈值

        Returns:
            包含所有指标的字典
        """
        y_pred = (y_probs >= threshold).astype(int)

        # 1. 基础统计
        n_total = len(y_true)
        n_pos = np.sum(y_true == self.minority_class)
        n_neg = n_total - n_pos
        pos_ratio = n_pos / n_total if n_total > 0 else 0

        # 2. ⭐ 少数类AUPRC（主要指标）
        if n_pos > 0 and n_neg > 0:
            precision, recall, _ = metrics.precision_recall_curve(
                y_true, y_probs, pos_label=self.minority_class
            )
            auprc_minority = metrics.auc(recall, precision)

            # 计算baseline（随机分类器的期望AP）
            auprc_baseline = pos_ratio
            auprc_gain = auprc_minority - auprc_baseline
        else:
            auprc_minority = 0.0
            auprc_baseline = 0.0
            auprc_gain = 0.0

        # 3. ⭐ MCC（Matthews相关系数，对不平衡鲁棒）
        try:
            mcc = metrics.matthews_corrcoef(y_true, y_pred)
        except:
            mcc = 0.0

        # 4. ROC-AUC（参考指标）
        if n_pos > 0 and n_neg > 0:
            roc_auc = metrics.roc_auc_score(y_true, y_probs)
        else:
            roc_auc = 0.5

        # 5. 混淆矩阵相关
        cm = metrics.confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        # 6. 少数类的 Precision, Recall, F1
        if tp + fp > 0:
            precision_minority = tp / (tp + fp)
        else:
            precision_minority = 0.0

        if tp + fn > 0:
            recall_minority = tp / (tp + fn)
        else:
            recall_minority = 0.0

        if precision_minority + recall_minority > 0:
            f1_minority = 2 * precision_minority * recall_minority / (precision_minority + recall_minority)
        else:
            f1_minority = 0.0

        # 7. 特异性（对多数类的召回率）
        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0.0

        # 8. G-Mean（几何平均，平衡两类）
        gmean = np.sqrt(recall_minority * specificity)

        return {
            # ⭐ 主要指标
            'auprc_minority': float(auprc_minority),
            'auprc_baseline': float(auprc_baseline),
            'auprc_gain': float(auprc_gain),
            'mcc': float(mcc),

            # 少数类性能
            'precision_minority': float(precision_minority),
            'recall_minority': float(recall_minority),
            'f1_minority': float(f1_minority),

            # 平衡指标
            'gmean': float(gmean),
            'specificity': float(specificity),

            # 参考指标
            'roc_auc': float(roc_auc),

            # 统计信息
            'pos_ratio': float(pos_ratio),
            'n_pos': int(n_pos),
            'n_neg': int(n_neg),
            'threshold': float(threshold),

            # 混淆矩阵
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

    def find_best_threshold(self, y_true: np.ndarray, y_probs: np.ndarray,
                            metric: str = 'mcc') -> Tuple[float, float]:
        """
        在验证集上寻找最优阈值

        Args:
            y_true: 真实标签
            y_probs: 预测概率
            metric: 优化目标 ('mcc', 'f1', 'gmean', 'youden')

        Returns:
            (best_threshold, best_metric_value)
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        best_value = -1.0
        best_thresh = 0.5

        for th in thresholds:
            y_pred = (y_probs >= th).astype(int)

            if metric == 'mcc':
                try:
                    value = metrics.matthews_corrcoef(y_true, y_pred)
                except:
                    value = 0.0
            elif metric == 'f1':
                value = metrics.f1_score(y_true, y_pred, pos_label=self.minority_class, zero_division=0)
            elif metric == 'gmean':
                cm = metrics.confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    value = np.sqrt(recall * spec)
                else:
                    value = 0.0
            elif metric == 'youden':
                # Youden's J statistic = sensitivity + specificity - 1
                cm = metrics.confusion_matrix(y_true, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    value = recall + spec - 1
                else:
                    value = 0.0
            else:
                raise ValueError(f"Unknown metric: {metric}")

            if value > best_value:
                best_value = value
                best_thresh = th

        return best_thresh, best_value

    def print_report(self, metrics_dict: Dict[str, float], name: str = "Validation"):
        """打印格式化的指标报告"""
        print(f"\n{'=' * 60}")
        print(f"{name} Metrics Report")
        print(f"{'=' * 60}")

        # 数据分布
        print(f"\n📊 Data Distribution:")
        print(f"  Positive: {metrics_dict['n_pos']} ({metrics_dict['pos_ratio']:.2%})")
        print(f"  Negative: {metrics_dict['n_neg']} ({1 - metrics_dict['pos_ratio']:.2%})")
        print(f"  Threshold: {metrics_dict['threshold']:.3f}")

        # ⭐ 主要指标
        print(f"\n⭐ Primary Metrics (Imbalance-Robust):")
        print(f"  AUPRC (Minority):  {metrics_dict['auprc_minority']:.4f}")
        print(f"  AUPRC Baseline:    {metrics_dict['auprc_baseline']:.4f}")
        print(
            f"  AUPRC Gain:        {metrics_dict['auprc_gain']:.4f} ({'↑' if metrics_dict['auprc_gain'] > 0 else '↓'})")
        print(f"  MCC:               {metrics_dict['mcc']:.4f}")

        # 少数类性能
        print(f"\n🎯 Minority Class Performance:")
        print(f"  Precision: {metrics_dict['precision_minority']:.4f}")
        print(f"  Recall:    {metrics_dict['recall_minority']:.4f}")
        print(f"  F1-Score:  {metrics_dict['f1_minority']:.4f}")

        # 平衡指标
        print(f"\n⚖️  Balanced Metrics:")
        print(f"  G-Mean:      {metrics_dict['gmean']:.4f}")
        print(f"  Specificity: {metrics_dict['specificity']:.4f}")

        # 参考指标
        print(f"\n📈 Reference:")
        print(f"  ROC-AUC: {metrics_dict['roc_auc']:.4f}")

        # 混淆矩阵
        print(f"\n📋 Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Neg    Pos")
        print(f"  Actual Neg  {metrics_dict['tn']:4d}   {metrics_dict['fp']:4d}")
        print(f"         Pos  {metrics_dict['fn']:4d}   {metrics_dict['tp']:4d}")

        print(f"{'=' * 60}\n")


def validate_model(model, dataloader, device, minority_class: int = 1):
    """
    模型验证函数（使用改进的指标）

    Returns:
        metrics_dict: 包含所有指标的字典
        probs: 预测概率 (N,)
        labels: 真实标签 (N,)
    """
    import torch

    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            video = batch["video"].to(device)
            audio = batch["audio"].to(device)
            labels = batch["label_idx"]

            # 前向传播
            out = model(video, audio)

            # 获取正类概率
            if "clip_logits" in out:
                logits = out["clip_logits"]
                probs = torch.softmax(logits, dim=-1)[:, minority_class]
            else:
                # 如果是单个logit
                probs = torch.sigmoid(out)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    # 计算指标
    calc = ImbalancedMetricsCalculator(minority_class=minority_class)

    # 先找最优阈值（基于MCC）
    best_thresh, best_mcc = calc.find_best_threshold(all_labels, all_probs, metric='mcc')

    # 用最优阈值计算所有指标
    metrics_dict = calc.compute_all_metrics(all_labels, all_probs, threshold=best_thresh)
    metrics_dict['best_threshold'] = best_thresh

    return metrics_dict, all_probs, all_labels