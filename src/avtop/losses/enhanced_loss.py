# src/avtop/losses/enhanced_loss.py
import torch
import torch.nn as nn

class RankingMILLoss(nn.Module):
    """基于报告的ranking loss + MIL"""

    def __init__(self, margin=0.5, topk=4):
        super().__init__()
        self.margin = margin
        self.topk = topk

    def forward(self, scores, labels):
        # scores: [B, T]
        # labels: [B]

        pos_mask = (labels == 1)
        neg_mask = (labels == 0)

        if pos_mask.any() and neg_mask.any():
            # 获取最高分
            pos_max = scores[pos_mask].max(dim=1)[0]  # 正样本最高分
            neg_max = scores[neg_mask].max(dim=1)[0]  # 负样本最高分

            # Ranking loss: 正样本最高分应该高于负样本
            ranking_loss = torch.clamp(
                self.margin - (pos_max.unsqueeze(1) - neg_max.unsqueeze(0)),
                min=0
            ).mean()

            # Top-K supervision for positive bags
            if pos_mask.any():
                pos_scores = scores[pos_mask]
                k = min(self.topk, pos_scores.size(1))
                topk_scores = torch.topk(pos_scores, k, dim=1)[0]
                # 鼓励top-k片段有高分
                topk_loss = -torch.log(torch.sigmoid(topk_scores)).mean()
            else:
                topk_loss = 0

            # 负样本所有片段都应该低分
            if neg_mask.any():
                neg_loss = -torch.log(1 - torch.sigmoid(scores[neg_mask])).mean()
            else:
                neg_loss = 0

            return ranking_loss + 0.5 * topk_loss + 0.5 * neg_loss

        return torch.tensor(0.0, device=scores.device)