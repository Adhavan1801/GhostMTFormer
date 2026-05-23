# ─────────────────────────────────────────
#  src/metrics.py
#  Dice, IoU, HD95
# ─────────────────────────────────────────

import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff


# ── Dice Coefficient ──────────────────────────────────────────
def dice_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum().item()
    return (2 * intersection + smooth) / (
        pred.sum().item() + target.sum().item() + smooth
    )


# ── Intersection over Union ───────────────────────────────────
def iou_score(pred: torch.Tensor, target: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-6) -> float:
    pred   = (torch.sigmoid(pred) > threshold).float()
    pred   = pred.view(-1)
    target = target.view(-1)

    intersection = (pred * target).sum().item()
    union        = (pred + target - pred * target).sum().item()
    return (intersection + smooth) / (union + smooth)


# ── HD95 ──────────────────────────────────────────────────────
def hd95_score(pred: torch.Tensor, target: torch.Tensor,
               threshold: float = 0.5) -> float:
    pred_np   = (torch.sigmoid(pred) > threshold).float()
    pred_np   = pred_np.squeeze().cpu().numpy()
    target_np = target.squeeze().cpu().numpy()

    # get boundary pixel coordinates
    pred_pts   = np.argwhere(pred_np   > 0.5)
    target_pts = np.argwhere(target_np > 0.5)

    # if either mask is empty return large penalty
    if len(pred_pts) == 0 or len(target_pts) == 0:
        return 373.13   # diagonal of 512x512 image

    d1 = directed_hausdorff(pred_pts,   target_pts)[0]
    d2 = directed_hausdorff(target_pts, pred_pts)[0]

    # HD95 — use 95th percentile of all surface distances
    from scipy.spatial import cKDTree
    tree1  = cKDTree(pred_pts)
    tree2  = cKDTree(target_pts)

    d_pred_to_gt, _ = tree2.query(pred_pts)
    d_gt_to_pred, _ = tree1.query(target_pts)

    all_distances = np.concatenate([d_pred_to_gt, d_gt_to_pred])
    return float(np.percentile(all_distances, 95))


# ── Batch metrics (averages over a batch) ────────────────────
def batch_metrics(preds: torch.Tensor, targets: torch.Tensor,
                  threshold: float = 0.5) -> dict:
    batch_size = preds.shape[0]
    dice_vals  = []
    iou_vals   = []
    hd95_vals  = []

    for i in range(batch_size):
        p = preds[i].unsqueeze(0)
        t = targets[i].unsqueeze(0)
        dice_vals.append(dice_score(p, t, threshold))
        iou_vals.append(iou_score(p, t, threshold))
        hd95_vals.append(hd95_score(p, t, threshold))

    return {
        "dice": np.mean(dice_vals),
        "iou":  np.mean(iou_vals),
        "hd95": np.mean(hd95_vals)
    }


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    pred   = torch.randn(2, 1, 512, 512)
    target = torch.randint(0, 2, (2, 1, 512, 512)).float()

    m = batch_metrics(pred, target)
    print(f"  Dice : {m['dice']:.4f}")
    print(f"  IoU  : {m['iou']:.4f}")
    print(f"  HD95 : {m['hd95']:.2f} px")
    print("  Metrics OK")