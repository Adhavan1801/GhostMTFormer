# ─────────────────────────────────────────
#  src/model/global_encoder.py
#  CNN Global Context Encoder
#  captures long-range structure and lesion extent
# ─────────────────────────────────────────

import torch
import torch.nn as nn


# ── Depthwise Separable Convolution ───────────────────────────
# wider receptive field at lower cost than standard conv
class DSConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            # depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride,
                      1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ── Dilated Context Block ─────────────────────────────────────
# uses dilated convolutions to capture large receptive fields
# without losing spatial resolution inside a stage
class DilatedContextBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        mid = channels // 4

        self.d1 = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=1,  dilation=1,  bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.d2 = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=2,  dilation=2,  bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.d4 = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=4,  dilation=4,  bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        self.d8 = nn.Sequential(
            nn.Conv2d(channels, mid, 3, padding=8,  dilation=8,  bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(mid * 4, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(torch.cat([
            self.d1(x), self.d2(x),
            self.d4(x), self.d8(x)
        ], dim=1)) + x   # residual


# ── Global Stage ──────────────────────────────────────────────
class GlobalStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.downsample = DSConv(in_channels, out_channels, stride=stride)
        self.context    = DilatedContextBlock(out_channels)

    def forward(self, x):
        return self.context(self.downsample(x))


# ── Global Encoder ────────────────────────────────────────────
# 4 stages matching Ghost encoder scales for CFCA pairing
# input  : (B, 3, 512, 512)
# outputs: t1..t4 at 128, 64, 32, 16 spatial resolution
class GlobalEncoder(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()

        # stem: 512 → 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )

        # stage1: 256 → 128  (pairs with Ghost f1)
        self.stage1 = GlobalStage(base,      base * 2,  stride=2)

        # stage2: 128 → 64   (pairs with Ghost f2)
        self.stage2 = GlobalStage(base * 2,  base * 4,  stride=2)

        # stage3: 64 → 32    (pairs with Ghost f3)
        self.stage3 = GlobalStage(base * 4,  base * 8,  stride=2)

        # stage4: 32 → 16    (pairs with Ghost f4)
        self.stage4 = GlobalStage(base * 8,  base * 16, stride=2)

    def forward(self, x):
        s   = self.stem(x)    # (B, 64,  256, 256)
        t1  = self.stage1(s)  # (B, 128, 128, 128)
        t2  = self.stage2(t1) # (B, 256,  64,  64)
        t3  = self.stage3(t2) # (B, 512,  32,  32)
        t4  = self.stage4(t3) # (B, 1024, 16,  16)
        return t1, t2, t3, t4


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    model = GlobalEncoder(in_channels=3, base=64)
    x     = torch.randn(2, 3, 512, 512)

    t1, t2, t3, t4 = model(x)
    print(f"  t1 : {tuple(t1.shape)}")
    print(f"  t2 : {tuple(t2.shape)}")
    print(f"  t3 : {tuple(t3.shape)}")
    print(f"  t4 : {tuple(t4.shape)}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n  GlobalEncoder params : {params:.2f}M")
    print("  GlobalEncoder OK")