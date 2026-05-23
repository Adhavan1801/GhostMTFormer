# ─────────────────────────────────────────
#  src/evaluate.py
#  Final evaluation on test set
#  Loads best checkpoint, computes metrics,
#  saves visual results
# ─────────────────────────────────────────

import os
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.dataset             import get_dataloaders, load_config
from src.metrics             import dice_score, iou_score, hd95_score
from src.model.ghostmtformer import GhostMTFormer


# ── Load model from checkpoint ────────────────────────────────
def load_model(cfg, checkpoint_path, device):
    model = GhostMTFormer(cfg).to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"  Loaded checkpoint from epoch {ckpt['epoch']} "
          f"(val Dice={ckpt['best_dice']:.4f})")
    return model


# ── Test-Time Augmentation (TTA) ──────────────────────────────
# averages predictions across flipped versions of each image
# free performance boost at inference, no extra training needed
def predict_with_tta(model, image, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for flip in [None, "h", "v", "hv"]:
            x = image.clone()
            if flip == "h":
                x = torch.flip(x, dims=[3])
            elif flip == "v":
                x = torch.flip(x, dims=[2])
            elif flip == "hv":
                x = torch.flip(x, dims=[2, 3])

            out  = model(x)
            prob = torch.sigmoid(out["seg"])

            # undo flip on prediction
            if flip == "h":
                prob = torch.flip(prob, dims=[3])
            elif flip == "v":
                prob = torch.flip(prob, dims=[2])
            elif flip == "hv":
                prob = torch.flip(prob, dims=[2, 3])

            preds.append(prob)

    return torch.stack(preds).mean(dim=0)


# ── Save visual result ────────────────────────────────────────
def save_visual(image_tensor, mask_tensor, pred_tensor,
                save_path, idx):
    # denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = image_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img  = (img * std + mean).clip(0, 1)

    mask = mask_tensor.squeeze().cpu().numpy()
    pred = pred_tensor.squeeze().cpu().numpy()

    # overlay prediction on image
    overlay = img.copy()
    overlay[pred > 0.5] = overlay[pred > 0.5] * 0.5 + \
                          np.array([0, 1, 0]) * 0.5   # green overlay

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(img);            axes[0].set_title("Image");      axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
    axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Prediction");  axes[2].axis("off")
    axes[3].imshow(overlay);        axes[3].set_title("Overlay");    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"result_{idx:04d}.png"),
                dpi=100, bbox_inches="tight")
    plt.close()


# ── Main evaluation ───────────────────────────────────────────
def evaluate(cfg_path="configs/default.yaml",
             checkpoint_path="checkpoints/best_model.pth",
             save_visuals=True,
             n_visuals=20,
             use_tta=True):

    cfg    = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # directories
    masks_dir  = os.path.join(cfg["paths"]["results"], "masks")
    os.makedirs(masks_dir, exist_ok=True)

    # data — only test loader needed
    print("\n  Loading test data...")
    _, _, test_loader = get_dataloaders(cfg)
    print(f"  Test batches : {len(test_loader)}")

    # model
    model = load_model(cfg, checkpoint_path, device)
    model.eval()

    threshold = cfg["inference"]["threshold"]
    dice_vals = []
    iou_vals  = []
    hd95_vals = []

    print(f"\n  Running evaluation {'with TTA' if use_tta else ''}...\n")

    for i, (images, masks) in enumerate(tqdm(test_loader, desc="  Testing")):
        images = images.to(device)
        masks  = masks.to(device)

        # inference
        if use_tta:
            prob = predict_with_tta(model, images, device)
        else:
            with torch.no_grad():
                out  = model(images)
                prob = torch.sigmoid(out["seg"])

        pred = (prob > threshold).float()

        # per-image metrics
        for j in range(images.shape[0]):
            p = pred[j].unsqueeze(0)
            t = masks[j].unsqueeze(0)
            dice_vals.append(dice_score(p, t, threshold))
            iou_vals.append(iou_score(p, t, threshold))
            hd95_vals.append(hd95_score(p, t, threshold))

        # save visuals for first n_visuals batches
        if save_visuals and i < n_visuals:
            save_visual(images[0], masks[0], prob[0],
                        masks_dir, i)

    # final metrics
    mean_dice = np.mean(dice_vals) * 100
    mean_iou  = np.mean(iou_vals)  * 100
    mean_hd95 = np.mean(hd95_vals)
    std_dice  = np.std(dice_vals)  * 100
    std_iou   = np.std(iou_vals)   * 100

    print(f"\n  {'─' * 45}")
    print(f"  Final Test Results ({len(dice_vals)} images)")
    print(f"  {'─' * 45}")
    print(f"  Dice  : {mean_dice:.2f}% ± {std_dice:.2f}%")
    print(f"  IoU   : {mean_iou:.2f}% ± {std_iou:.2f}%")
    print(f"  HD95  : {mean_hd95:.2f} px")
    print(f"  {'─' * 45}")

    # save results to json
    results = {
        "dice_mean": round(mean_dice, 4),
        "dice_std":  round(std_dice,  4),
        "iou_mean":  round(mean_iou,  4),
        "iou_std":   round(std_iou,   4),
        "hd95_mean": round(mean_hd95, 4),
        "n_images":  len(dice_vals),
        "tta":       use_tta,
        "threshold": threshold
    }
    results_path = os.path.join(cfg["paths"]["results"], "test_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to : {results_path}")
    print(f"  Visuals saved to : {masks_dir}")

    return results


if __name__ == "__main__":
    evaluate()