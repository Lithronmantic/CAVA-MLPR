from typing import Dict
import numpy as np, torch
from sklearn import metrics
def simple_metrics_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
        y = labels.cpu().numpy().astype(np.int32)
        pred = (probs >= 0.5).astype(np.int32)
        tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp + 1e-8); rec = tp / (tp + fn + 1e-8); f1 = 2*prec*rec / (prec+rec+1e-8)
        fpr, tpr, _ = metrics.roc_curve(y, probs); auc = float(metrics.auc(fpr, tpr))
        # 11-point AP
        order = np.argsort(-probs); tp_cum=0; fp_cum=0; precs=[]; recs=[]; P=int((y==1).sum())
        for i in order:
            if y[i]==1: tp_cum+=1
            else: fp_cum+=1
            precs.append(tp_cum/max(tp_cum+fp_cum,1)); recs.append(tp_cum/max(P,1))
        ap=0.0
        for r in np.linspace(0,1,11):
            mask = np.array(recs) >= r; ap += (np.array(precs)[mask].max() if mask.any() else 0.0)/11.0
        return {"f1": float(f1), "ap": float(ap), "auc": float(auc)}
