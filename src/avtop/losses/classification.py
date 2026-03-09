# src/avtop/losses/classification.py
import torch
import torch.nn.functional as F

# -------- Binary（兼容现在） ----------
def clip_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if targets.ndim == 1:
        targets = F.one_hot(targets.long(), num_classes=logits.size(-1)).float()
    return F.binary_cross_entropy_with_logits(logits, targets)

def mil_loss_noisyor(seg_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # seg_logits: [B,T,2] or [B,T]（取正类 logit）
    s = seg_logits[..., 1] if seg_logits.ndim == 3 else seg_logits  # [B,T]
    p = torch.sigmoid(s)                 # 非原地激活
    p = p.clamp(1e-6, 1-1e-6)           # 非原地 clamp（避免 inplace 反向问题）
    p_bag = 1.0 - torch.prod(1.0 - p, dim=1)
    return F.binary_cross_entropy(p_bag, y.float())

def segment_supervision(seg_logits: torch.Tensor,
                        y: torch.Tensor,
                        weights: torch.Tensor = None,
                        k: int = 4) -> torch.Tensor:
    """
    弱监督段级监督：
    - 负包：所有时刻→0
    - 正包：仅 Top-K 时刻→1（其余不监督）
    seg_logits: [B,T,2]
    y:          [B]
    weights:    [B,T]（来自 MIL 路由）；若 None，只做负包全负
    """
    s = seg_logits[..., 1]  # [B,T] 正类 logit
    pos = (y == 1); neg = (y == 0)
    loss = s.new_tensor(0.0)

    if neg.any():
        target_neg = torch.zeros_like(s[neg])
        loss = loss + F.binary_cross_entropy_with_logits(s[neg], target_neg)

    if weights is not None and pos.any():
        w_pos = weights[pos]                   # [B_pos,T]
        k = min(int(k), w_pos.size(1))
        topk_idx = torch.topk(w_pos, k=k, dim=1).indices
        mask = torch.zeros_like(w_pos).scatter(1, topk_idx, 1.0)  # 在新张量上原地 OK
        s_pos = s[pos]
        loss = loss + F.binary_cross_entropy_with_logits(s_pos * mask, mask)

    return loss

# -------- Multi-label（后续要用，先放好） ----------
def multilabel_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # logits: [B,K], targets: [B,K] in {0,1}
    return F.binary_cross_entropy_with_logits(logits, targets.float())

def mil_noisyor_multilabel(seg_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # seg_logits: [B,T,K], targets: [B,K]
    p = torch.sigmoid(seg_logits).clamp(1e-6, 1-1e-6)
    p_bag = 1.0 - torch.prod(1.0 - p, dim=1)     # [B,K]
    return F.binary_cross_entropy(p_bag, targets.float())

def segment_topk_supervision_multilabel(seg_logits, targets, weights=None, k=4, w=0.5):
    B,T,K = seg_logits.shape
    loss = seg_logits.new_tensor(0.0)
    neg_mask = (targets == 0).float()            # [B,K]
    if neg_mask.any():
        neg_logit = seg_logits * neg_mask.unsqueeze(1)  # 非原地
        loss = loss + F.binary_cross_entropy_with_logits(
            neg_logit, torch.zeros_like(neg_logit), reduction="mean"
        )
    if weights is not None and (targets == 1).any():
        W = weights                               # [B,T]
        k = min(int(k), T)
        topk_idx = torch.topk(W, k=k, dim=1).indices
        mask_t = torch.zeros_like(W).scatter(1, topk_idx, 1.0)  # [B,T]
        pos_mask = (targets == 1).float()         # [B,K]
        pos_logits = seg_logits * mask_t.unsqueeze(-1) * pos_mask.unsqueeze(1)
        pos_target = mask_t.unsqueeze(-1) * pos_mask.unsqueeze(1)
        loss = loss + w * F.binary_cross_entropy_with_logits(pos_logits, pos_target)
    return loss
