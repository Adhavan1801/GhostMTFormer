# ─────────────────────────────────────────
#  src/dataset.py
#  HAM10000 Dataset — loading, splitting, augmentation
# ─────────────────────────────────────────

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml


# ── Load config ───────────────────────────────────────────────
def load_config(path="configs/default.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Augmentation pipelines ────────────────────────────────────
def get_transforms(split: str, cfg: dict):
    img_size = cfg["data"]["img_size"]
    aug = cfg["augmentation"]

    if split == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5 if aug["horizontal_flip"] else 0),
            A.VerticalFlip(p=0.5 if aug["vertical_flip"] else 0),
            A.Rotate(limit=aug["rotation_limit"], p=0.5),
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),
                p=0.4
            ),
            A.ColorJitter(
                brightness=aug["brightness_limit"],
                contrast=aug["contrast_limit"],
                saturation=aug["saturation_limit"],
                p=0.5
            ),
            A.GaussNoise(p=0.2),
            A.ElasticTransform(p=0.2),
            A.GridDistortion(p=0.2),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])
    else:
        # val and test — only resize and normalize
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            ToTensorV2()
        ])


# ── Dataset class ─────────────────────────────────────────────
class HAM10000Dataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)   # binarize to 0/1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"].unsqueeze(0)  # (1, H, W)

        return image, mask


# ── Split builder ─────────────────────────────────────────────
def build_splits(cfg: dict):
    images_dir = cfg["data"]["images_dir"]
    masks_dir  = cfg["data"]["masks_dir"]
    seed       = cfg["split"]["seed"]
    train_r    = cfg["split"]["train"]
    val_r      = cfg["split"]["val"]

    # Collect matched image-mask pairs
    image_files = sorted([
        f for f in os.listdir(images_dir) if f.endswith(".jpg")
    ])

    pairs = []
    missing = 0
    for img_file in image_files:
        stem      = os.path.splitext(img_file)[0]
        mask_file = stem + "_segmentation.png"
        mask_path = os.path.join(masks_dir, mask_file)

        if os.path.exists(mask_path):
            pairs.append((
                os.path.join(images_dir, img_file),
                mask_path
            ))
        else:
            missing += 1

    if missing > 0:
        print(f"  Warning: {missing} images had no matching mask and were skipped.")

    print(f"  Total valid pairs: {len(pairs)}")

    image_paths = [p[0] for p in pairs]
    mask_paths  = [p[1] for p in pairs]

    # Train / temp split
    val_test_r = 1.0 - train_r
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, mask_paths,
        test_size=val_test_r,
        random_state=seed
    )

    # Val / test split from temp
    val_ratio_of_temp = val_r / val_test_r
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1.0 - val_ratio_of_temp,
        random_state=seed
    )

    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ── DataLoader factory ────────────────────────────────────────
def get_dataloaders(cfg: dict):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_splits(cfg)

    train_ds = HAM10000Dataset(X_train, y_train, get_transforms("train", cfg))
    val_ds   = HAM10000Dataset(X_val,   y_val,   get_transforms("val",   cfg))
    test_ds  = HAM10000Dataset(X_test,  y_test,  get_transforms("test",  cfg))

    bs           = cfg["training"]["batch_size"]
    num_workers  = cfg["data"]["num_workers"]

    train_loader = DataLoader(train_ds, batch_size=bs,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)

    return train_loader, val_loader, test_loader


# ── Quick sanity check ────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    print("\nBuilding data splits...")
    train_loader, val_loader, test_loader = get_dataloaders(cfg)

    images, masks = next(iter(train_loader))
    print(f"\n  Image batch shape : {images.shape}")
    print(f"  Mask  batch shape : {masks.shape}")
    print(f"  Image dtype       : {images.dtype}")
    print(f"  Mask  unique vals : {masks.unique()}")
    print("\n  Dataset pipeline OK")