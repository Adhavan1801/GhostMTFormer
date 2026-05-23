# ─────────────────────────────────────────
#  src/model/decoder.py
#  Boundary-Refined Multi-Task Decoder
#  with Deep Supervision
# ─────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Conv Block ────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ── Boundary Refinement Module (BRM) ─────────────────────────
# predicts a coarse boundary map from current features
# uses it as an attention gate to sharpen the feature map
# this forces intermediate representations to align with edges
class BRM(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # predict boundary probability map
        self.boundary_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, 1, 1)
        )

        # refine features using boundary as attention
        self.refine = nn.Sequential(
            nn.Conv2d(channels + 1, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        boundary = self.boundary_head(x)            # (B, 1, H, W)
        boundary_attn = torch.sigmoid(boundary)     # soft boundary gate
        x_refined = self.refine(
            torch.cat([x, boundary_attn], dim=1)   # inject boundary signal
        )
        return x_refined, boundary


# ── Decoder Stage ─────────────────────────────────────────────
class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels,
                 use_brm=True):
        super().__init__()
        self.use_brm = use_brm

        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        self.brm  = BRM(out_channels) if use_brm else None

    def forward(self, x, skip):
        # upsample current features to skip connection size
        x = F.interpolate(x, size=skip.shape[2:],
                          mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        boundary = None
        if self.use_brm:
            x, boundary = self.brm(x)

        return x, boundary


# ── Deep Supervision Head ─────────────────────────────────────
# produces auxiliary segmentation output at intermediate scale
# forces early decoder stages to learn meaningful representations
class DSHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x, target_size):
        out = self.head(x)
        return F.interpolate(out, size=target_size,
                             mode="bilinear", align_corners=False)


# ── Full Decoder ──────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, base=64, deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # bottleneck channels → 1024
        # Ghost skips: f3=512, f2=256, f1=128, f0=64

        # stage 1: 16→32   bottleneck(1024) + f3(512) → 512
        self.stage1 = DecoderStage(1024, 512, 512,  use_brm=True)

        # stage 2: 32→64   512 + f2(256) → 256
        self.stage2 = DecoderStage(512,  256, 256,  use_brm=True)

        # stage 3: 64→128  256 + f1(128) → 128
        self.stage3 = DecoderStage(256,  128, 128,  use_brm=True)

        # stage 4: 128→256 128 + f0(64)  → 64
        self.stage4 = DecoderStage(128,  64,  64,   use_brm=False)

        # deep supervision heads (stages 1-3)
        if deep_supervision:
            self.ds1 = DSHead(512)
            self.ds2 = DSHead(256)
            self.ds3 = DSHead(128)

        # final segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        # final edge head
        self.edge_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, bottleneck, skips, target_size):
        f0, f1, f2, f3 = skips   # Ghost encoder skips

        d1, b1 = self.stage1(bottleneck, f3)  # 16→32
        d2, b2 = self.stage2(d1,         f2)  # 32→64
        d3, b3 = self.stage3(d2,         f1)  # 64→128
        d4, _  = self.stage4(d3,         f0)  # 128→256

        # upsample d4 to full resolution
        d4 = F.interpolate(d4, size=target_size,
                           mode="bilinear", align_corners=False)

        # primary outputs
        seg_out  = self.seg_head(d4)
        edge_out = self.edge_head(d4)

        outputs = {
            "seg":  seg_out,
            "edge": edge_out,
        }

        # boundary maps from BRMs
        outputs["boundaries"] = [b for b in [b1, b2, b3] if b is not None]

        # deep supervision outputs
        if self.deep_supervision:
            outputs["ds1"] = self.ds1(d1, target_size)
            outputs["ds2"] = self.ds2(d2, target_size)
            outputs["ds3"] = self.ds3(d3, target_size)

        return outputs


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    bottleneck = torch.randn(2, 1024, 16, 16)
    f0 = torch.randn(2, 64,  256, 256)
    f1 = torch.randn(2, 128, 128, 128)
    f2 = torch.randn(2, 256,  64,  64)
    f3 = torch.randn(2, 512,  32,  32)

    decoder = Decoder(base=64, deep_supervision=True)
    outputs = decoder(bottleneck, (f0, f1, f2, f3),
                      target_size=(512, 512))

    print(f"  seg  : {tuple(outputs['seg'].shape)}")
    print(f"  edge : {tuple(outputs['edge'].shape)}")
    print(f"  ds1  : {tuple(outputs['ds1'].shape)}")
    print(f"  ds2  : {tuple(outputs['ds2'].shape)}")
    print(f"  ds3  : {tuple(outputs['ds3'].shape)}")
    print(f"  boundaries : {len(outputs['boundaries'])} maps")

    params = sum(p.numel() for p in decoder.parameters()) / 1e6
    print(f"\n  Decoder params : {params:.2f}M")
    print("  Decoder OK")