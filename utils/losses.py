"""Dice + CrossEntropy 조합 loss, 그리고 평가용 Dice metric.

라벨 규칙: 0=background, 1=OD-ring(cup 제외 disc), 2=OC(cup)
평가할 땐 "전체 OD" = (label==1)|(label==2) 로 재구성해서
OD Dice / OC Dice를 따로 리포트해 (REFUGE 관례와 동일).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, num_classes: int = 3, dice_weight: float = 1.0, ce_weight: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # logits: (B, K, H, W) raw, target: (B, H, W) long
        ce_loss = self.ce(logits, target)

        probs = F.softmax(logits, dim=1)
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dims)
        cardinality = torch.sum(probs + target_onehot, dims)
        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        dice_loss = 1.0 - dice_per_class.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


@torch.no_grad()
def compute_dice_scores(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    """반환: dict(od=float, oc=float, mean=float)  -- 배치 평균 Dice."""
    pred = torch.argmax(logits, dim=1)  # (B, H, W)

    pred_od = (pred == 1) | (pred == 2)
    pred_oc = pred == 2
    gt_od = (target == 1) | (target == 2)
    gt_oc = target == 2

    def dice(a, b):
        a = a.float()
        b = b.float()
        inter = (a * b).sum(dim=(1, 2))
        denom = a.sum(dim=(1, 2)) + b.sum(dim=(1, 2))
        return ((2 * inter + eps) / (denom + eps)).mean().item()

    od_dice = dice(pred_od, gt_od)
    oc_dice = dice(pred_oc, gt_oc)
    return {"od": od_dice, "oc": oc_dice, "mean": (od_dice + oc_dice) / 2}
