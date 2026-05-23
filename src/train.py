# ─────────────────────────────────────────
#  src/train.py
#  Training loop with deep supervision,
#  mixed precision, and checkpointing
# ─────────────────────────────────────────

import os
import yaml
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np

from src.dataset         import get_dataloaders, load_config
from src.losses          import CombinedLoss
from src.metrics         import batch_metrics
from src.model.ghostmtformer import GhostMTFormer


# ── Weighted deep supervision loss ────────────────────────────
def compute_total_loss(outputs, target, criterion):
    # primary segmentation loss
    loss = criterion(outputs["seg"], target)

    # edge head loss (same criterion, lower weight)
    loss += 0.3 * criterion(outputs["edge"], target)

    # deep supervision losses (decreasing weight per stage)
    ds_weights = [0.4, 0.3, 0.2]
    for i, key in enumerate(["ds1", "ds2", "ds3"]):
        if key in outputs:
            loss += ds_weights[i] * criterion(outputs[key], target)

    # boundary map losses
    for b in outputs.get("boundaries", []):
        if b is not None:
            b_up = torch.nn.functional.interpolate(
                b, size=target.shape[2:],
                mode="bilinear", align_corners=False
            )
            loss += 0.1 * criterion(b_up, target)

    return loss


# ── One epoch training ────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, grad_clip):
    model.train()
    total_loss = 0.0
    dice_sum   = 0.0
    iou_sum    = 0.0
    n_batches  = len(loader)

    for i, (images, masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model(images)
            loss    = compute_total_loss(outputs, masks, criterion)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # metrics on primary output only
        # metrics on primary output only — no HD95 during training
        with torch.no_grad():
            pred  = (torch.sigmoid(outputs["seg"].detach()) > 0.5).float()
            pred  = pred.view(-1)
            tgt   = masks.view(-1)
            inter = (pred * tgt).sum().item()
            dice  = (2 * inter + 1e-6) / (pred.sum().item() + tgt.sum().item() + 1e-6)
            iou   = (inter + 1e-6) / (pred.sum().item() + tgt.sum().item() - inter + 1e-6)

        total_loss += loss.item()
        dice_sum   += dice
        iou_sum    += iou

        if (i + 1) % 50 == 0:
            print(f"    step [{i+1}/{n_batches}] "
                  f"loss={loss.item():.4f} "
                  f"dice={dice:.4f}")

    return {
        "loss": total_loss / n_batches,
        "dice": dice_sum   / n_batches,
        "iou":  iou_sum    / n_batches,
    }


# ── Validation ────────────────────────────────────────────────
def validate(model, loader, criterion, device, compute_hd95=False):
    model.eval()
    total_loss = 0.0
    dice_sum   = 0.0
    iou_sum    = 0.0
    n_batches  = len(loader)

    # collect a small sample for HD95
    hd95_preds   = []
    hd95_targets = []
    hd95_limit   = 30   # compute HD95 on 30 images only

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device,  non_blocking=True)

            with autocast("cuda"):
                outputs = model(images)
                loss    = compute_total_loss(outputs, masks, criterion)

            pred  = (torch.sigmoid(outputs["seg"]) > 0.5).float()
            pred_flat  = pred.view(-1)
            tgt_flat   = masks.view(-1)
            inter = (pred_flat * tgt_flat).sum().item()
            dice  = (2 * inter + 1e-6) / (
                pred_flat.sum().item() + tgt_flat.sum().item() + 1e-6)
            iou   = (inter + 1e-6) / (
                pred_flat.sum().item() + tgt_flat.sum().item() - inter + 1e-6)

            total_loss += loss.item()
            dice_sum   += dice
            iou_sum    += iou

            # collect samples for HD95
            if compute_hd95 and len(hd95_preds) < hd95_limit:
                for j in range(images.shape[0]):
                    if len(hd95_preds) < hd95_limit:
                        hd95_preds.append(pred[j].unsqueeze(0).cpu())
                        hd95_targets.append(masks[j].unsqueeze(0).cpu())

    # compute HD95 on collected subset
    hd95_val = 0.0
    if compute_hd95 and hd95_preds:
        from src.metrics import hd95_score
        scores = []
        for p, t in zip(hd95_preds, hd95_targets):
            scores.append(hd95_score(p, t))
        hd95_val = float(np.mean(scores))

    return {
        "loss": total_loss / n_batches,
        "dice": dice_sum   / n_batches,
        "iou":  iou_sum    / n_batches,
        "hd95": hd95_val
    }


# ── Main training loop ────────────────────────────────────────
def train(cfg_path="configs/default.yaml"):
    cfg    = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    # data
    print("\n  Loading data...")
    train_loader, val_loader, _ = get_dataloaders(cfg)

    # model
    model = GhostMTFormer(cfg).to(device)
    total, trainable = model.count_params()
    print(f"  Params : {total:.2f}M total | {trainable:.2f}M trainable")

    # loss, optimizer, scheduler
    criterion = CombinedLoss(cfg)
    optimizer = AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg["training"]["epochs"],
        eta_min=1e-6
    )
    scaler = GradScaler("cuda")
    grad_clip = cfg["training"]["grad_clip"]
    epochs    = cfg["training"]["epochs"]

    # checkpointing
    ckpt_dir  = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_dice = 0.0
    best_path = os.path.join(ckpt_dir, "best_model.pth")

    print(f"\n  Starting training for {epochs} epochs...\n")

    history = {k: [] for k in
               ["train_loss","train_dice","val_loss",
                "val_dice","val_iou","val_hd95"]}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(
            model, train_loader, optimizer,
            criterion, scaler, device, grad_clip
        )
        # compute HD95 every 5 epochs
        run_hd95 = (epoch % 5 == 0) or (epoch == epochs)
        val_m = validate(model, val_loader, criterion, device,
                         compute_hd95=run_hd95)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # log
        for k in history:
            src_k = k.replace("train_","").replace("val_","")
            history[k].append(
                train_m[src_k] if k.startswith("train") else val_m[src_k]
            )

        print(f"\n  Epoch {epoch}/{epochs}")
        print(f"    Train → Loss: {train_m['loss']:.4f} | Dice: {train_m['dice']:.4f} | IoU: {train_m['iou']:.4f}")
        print(f"    Val   → Loss: {val_m['loss']:.4f} | Dice: {val_m['dice']:.4f} | IoU: {val_m['iou']:.4f} | HD95: {val_m['hd95']:.2f}px")
        print(f"    LR: {lr:.6f} | Time: {elapsed:.0f}s")
        print(f"    {'─' * 55}")

        # save best
        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_dice":  best_dice,
                "cfg":        cfg
            }, best_path)
            print(f"  ✓ Saved best model  (dice={best_dice:.4f})")

    print(f"\n  Training complete. Best val Dice : {best_dice:.4f}")
    print(f"  Best model saved to : {best_path}")

    # save history
    import json
    history_path = os.path.join(cfg["paths"]["results"], "history.json")
    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"  History saved to    : {history_path}")


if __name__ == "__main__":
    train()