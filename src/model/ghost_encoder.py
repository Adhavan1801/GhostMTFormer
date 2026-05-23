# ─────────────────────────────────────────
#  src/model/ghost_encoder.py
#  Lightweight GhostNet Local Encoder
# ─────────────────────────────────────────

import torch
import torch.nn as nn


# ── Ghost Module ──────────────────────────────────────────────
# generates extra feature maps via cheap linear ops
# instead of full convolutions on all channels
class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 ratio=2, dw_size=3, stride=1):
        super().__init__()
        self.out_channels = out_channels
        init_channels  = out_channels // ratio          # primary features
        cheap_channels = out_channels - init_channels   # ghost features

        # primary convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels,
                      kernel_size, stride,
                      kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True)
        )

        # cheap depthwise convolution to generate ghost features
        self.cheap_op = nn.Sequential(
            nn.Conv2d(init_channels, cheap_channels,
                      dw_size, 1, dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(cheap_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        primary = self.primary_conv(x)
        ghost   = self.cheap_op(primary)
        return torch.cat([primary, ghost], dim=1)


# ── Ghost Bottleneck Block ────────────────────────────────────
class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1):
        super().__init__()
        self.stride = stride

        self.block = nn.Sequential(
            # expand
            GhostModule(in_channels, mid_channels),
            # depthwise conv for stride-2 downsampling
            nn.Conv2d(mid_channels, mid_channels, 3, stride,
                      1, groups=mid_channels, bias=False)
            if stride > 1 else nn.Identity(),
            nn.BatchNorm2d(mid_channels) if stride > 1 else nn.Identity(),
            # project
            GhostModule(mid_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


# ── GhostNet Encoder ─────────────────────────────────────────
# 5 stages, each halving spatial resolution
# returns skip features at each stage for decoder
class GhostEncoder(nn.Module):
    def __init__(self, in_channels=3, base=64):
        super().__init__()

        # stem: 512 → 256
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, 2, 1, bias=False),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True)
        )

        # stage1: 256 → 128
        self.stage1 = nn.Sequential(
            GhostBottleneck(base,      base,      base * 2, stride=2),
            GhostBottleneck(base * 2,  base * 2,  base * 2, stride=1),
        )

        # stage2: 128 → 64
        self.stage2 = nn.Sequential(
            GhostBottleneck(base * 2,  base * 2,  base * 4, stride=2),
            GhostBottleneck(base * 4,  base * 4,  base * 4, stride=1),
        )

        # stage3: 64 → 32
        self.stage3 = nn.Sequential(
            GhostBottleneck(base * 4,  base * 4,  base * 8, stride=2),
            GhostBottleneck(base * 8,  base * 8,  base * 8, stride=1),
            GhostBottleneck(base * 8,  base * 8,  base * 8, stride=1),
        )

        # stage4: 32 → 16
        self.stage4 = nn.Sequential(
            GhostBottleneck(base * 8,  base * 8,  base * 16, stride=2),
            GhostBottleneck(base * 16, base * 16, base * 16, stride=1),
            GhostBottleneck(base * 16, base * 16, base * 16, stride=1),
        )

    def forward(self, x):
        f0 = self.stem(x)     # (B, 64,  256, 256)
        f1 = self.stage1(f0)  # (B, 128, 128, 128)
        f2 = self.stage2(f1)  # (B, 256,  64,  64)
        f3 = self.stage3(f2)  # (B, 512,  32,  32)
        f4 = self.stage4(f3)  # (B, 1024, 16,  16)
        return f0, f1, f2, f3, f4


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    model = GhostEncoder(in_channels=3, base=64)
    x     = torch.randn(2, 3, 512, 512)

    f0, f1, f2, f3, f4 = model(x)
    print(f"  f0 : {tuple(f0.shape)}")
    print(f"  f1 : {tuple(f1.shape)}")
    print(f"  f2 : {tuple(f2.shape)}")
    print(f"  f3 : {tuple(f3.shape)}")
    print(f"  f4 : {tuple(f4.shape)}")

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n  GhostEncoder params : {params:.2f}M")
    print("  GhostEncoder OK")