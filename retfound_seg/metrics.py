import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CFG


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.tc = CFG.train
        self.num_classes = CFG.data.num_classes
        self.ce = nn.CrossEntropyLoss()

    def _dice(self, logits, target):
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (probs * onehot).sum(dims)
        union = probs.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + 1.0) / (union + 1.0)
        return 1.0 - dice.mean()

    def forward(self, logits, target):
        return (self.tc.ce_weight * self.ce(logits, target)
                + self.tc.dice_weight * self._dice(logits, target))


@torch.no_grad()
def dice_per_class(logits, target) -> torch.Tensor:
    num_classes = CFG.data.num_classes
    pred = logits.argmax(dim=1)
    dices = []
    for c in range(num_classes):
        p = (pred == c).float()
        t = (target == c).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        dices.append((2 * inter + 1e-6) / (union + 1e-6))
    return torch.stack(dices)


@torch.no_grad()
def disc_cup_dice(logits, target) -> tuple[torch.Tensor, torch.Tensor]:
    pred = logits.argmax(dim=1)
    out = []
    for cond in (lambda m: m >= CFG.data.label_disc_rim,
                 lambda m: m >= CFG.data.label_cup):
        p = cond(pred).float()
        t = cond(target).float()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        out.append((2 * inter + 1e-6) / (union + 1e-6))
    return out[0], out[1]
