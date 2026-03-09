from typing import Dict
import torch, numpy as np
class TemporalLocalizationEvaluator:
    def __init__(self, iou_thresholds=(0.3,0.5,0.7), pos_class=1): self.iou_ths=iou_thresholds; self.pos_class=pos_class
    @staticmethod
    def _segments_from_scores(scores: torch.Tensor, thresh: float = 0.5):
        arr = (scores >= thresh).float().cpu().numpy(); segs=[]; s=None
        for t,v in enumerate(arr):
            if v and s is None: s=t
            if (not v) and (s is not None): segs.append((s, t-1, float(scores[s:t].max().item()))); s=None
        if s is not None: segs.append((s, len(arr)-1, float(scores[s:].max().item())))
        return segs
    @staticmethod
    def _tiou(a,b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1); uni = (a[1]-a[0]+1)+(b[1]-b[0]+1)-inter
        return inter/uni if uni>0 else 0.0
    def _ap_at_thresh(self, preds, gts, tiou_th=0.5):
        preds = sorted(preds, key=lambda x:-x[3]); tp=[]; fp=[]; matched={vid:[False]*len(gts.get(vid,[])) for vid in gts.keys()}
        for vid,s,e,conf in preds:
            hit=False
            if vid in gts:
                for i,(gs,ge) in enumerate(gts[vid]):
                    if not matched[vid][i] and self._tiou((s,e),(gs,ge))>=tiou_th: matched[vid][i]=True; hit=True; break
            tp.append(1 if hit else 0); fp.append(0 if hit else 1)
        tp=np.cumsum(tp); fp=np.cumsum(fp); num_gt=sum(len(v) for v in gts.values())
        if num_gt==0: return 0.0
        prec=tp/np.maximum(tp+fp,1e-8); rec=tp/num_gt; ap=0.0
        for r in np.linspace(0,1,11):
            mask=rec>=r; ap+=(prec[mask].max() if mask.any() else 0.0)/11.0
        return float(ap)
    def evaluate(self, model, val_loader, device="cpu", score_thresh=0.5):
        device=torch.device(device if torch.cuda.is_available() else "cpu"); model.eval(); all_preds=[]; gts={}
        with torch.no_grad():
            for batch in val_loader:
                vids=batch.get("clip_id",[None]*len(batch["video"]))
                video=batch["video"].to(device); audio=batch["audio"].to(device)
                out=model(audio, video); seg_logits=out.get("segment_logits", None)
                if seg_logits is None: continue
                seg_scores=torch.softmax(seg_logits, dim=-1)[..., self.pos_class] if seg_logits.ndim==3 else torch.sigmoid(seg_logits)
                for i in range(seg_scores.size(0)):
                    scores=seg_scores[i]; segs=self._segments_from_scores(scores, thresh=score_thresh)
                    all_preds += [(vids[i], s, e, conf) for (s,e,conf) in segs]
                    cur_gts=[(int(s), int(e)) for (s,e,_) in batch.get("gt_segments", [[]]*seg_scores.size(0))[i]]
                    if vids[i] not in gts: gts[vids[i]]=[]; gts[vids[i]] += cur_gts
        return {f"mAP@{th}": self._ap_at_thresh(all_preds, gts, tiou_th=th) for th in self.iou_ths}
