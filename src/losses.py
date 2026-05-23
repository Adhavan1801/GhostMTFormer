# ─────────────────────────────────────────
#  src/losses.py
#  Dice + BCE + Tversky + Focal + Boundary
# ─────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Dice Loss ─────────────────────────────────────────────────
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )


# ── Binary Cross Entropy Loss ─────────────────────────────────
class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


# ── Tversky Loss ──────────────────────────────────────────────
# alpha penalises false positives, beta penalises false negatives
# setting beta > alpha focuses the model on not missing lesion pixels
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, pred, target):
        pred   = torch.sigmoid(pred)
        pred   = pred.view(-1)
        target = target.view(-1)

        TP = (pred * target).sum()
        FP = ((1 - target) * pred).sum()
        FN = (target * (1 - pred)).sum()

        return 1 - (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )


# ── Focal Loss ────────────────────────────────────────────────
# down-weights easy pixels so the model focuses on hard boundary pixels
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=1e-6):
        super().__init__()
        self.gamma  = gamma
        self.smooth = smooth

    def forward(self, pred, target):
        prob = torch.sigmoid(pred)
        bce  = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
        p_t  = prob * target + (1 - prob) * (1 - target)
        loss = bce * ((1 - p_t) ** self.gamma)
        return loss.mean()


# ── Boundary Loss ─────────────────────────────────────────────
# computes loss only on pixels near the lesion boundary
class BoundaryLoss(nn.Module):
    def __init__(self, kernel_size=5, smooth=1e-6):
        super().__init__()
        self.smooth      = smooth
        self.kernel_size = kernel_size
        self.bce         = nn.BCEWithLogitsLoss()

    def _get_boundary(self, mask):
        # dilate then subtract original to get boundary ring
        pad      = self.kernel_size // 2
        pooled   = F.max_pool2d(mask, self.kernel_size, stride=1, padding=pad)
        boundary = pooled - mask
        return boundary

    def forward(self, pred, target):
        boundary = self._get_boundary(target)
        if boundary.sum() < 1:
            return torch.tensor(0.0, device=pred.device)
        boundary_pred   = pred   * boundary
        boundary_target = target * boundary
        return self.bce(boundary_pred, boundary_target)


# ── Combined Loss ─────────────────────────────────────────────
class CombinedLoss(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        w = cfg["loss"]

        self.dice_w     = w["dice_weight"]
        self.bce_w      = w["bce_weight"]
        self.tversky_w  = w["tversky_weight"]
        self.focal_w    = w["focal_weight"]
        self.boundary_w = w["boundary_weight"]

        self.dice     = DiceLoss()
        self.bce      = BCELoss()
        self.tversky  = TverskyLoss(
            alpha=w["tversky_alpha"],
            beta=w["tversky_beta"]
        )
        self.focal    = FocalLoss(gamma=w["focal_gamma"])
        self.boundary = BoundaryLoss()

    def forward(self, pred, target):
        loss = (
            self.dice_w     * self.dice(pred, target)     +
            self.bce_w      * self.bce(pred, target)      +
            self.tversky_w  * self.tversky(pred, target)  +
            self.focal_w    * self.focal(pred, target)    +
            self.boundary_w * self.boundary(pred, target)
        )
        return loss


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    pred   = torch.randn(2, 1, 512, 512)
    target = torch.randint(0, 2, (2, 1, 512, 512)).float()

    criterion = CombinedLoss(cfg)
    loss      = criterion(pred, target)

    print(f"  Combined loss value : {loss.item():.4f}")
    print("  Loss functions OK")