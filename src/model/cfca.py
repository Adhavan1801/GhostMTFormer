# ─────────────────────────────────────────
#  src/model/cfca.py
#  Cross-Feature Channel Attention (CFCA)
#  + Cross-Feature Fusion Bottleneck (XFF)
# ─────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Efficient Channel Attention (ECA) ─────────────────────────
# lightweight per-channel recalibration via 1D conv
# no dimensionality reduction — better than SE at fewer params
class ECA(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        # (B, C, H, W) → (B, C, 1, 1) → (B, 1, C) → conv → (B, C)
        w = self.gap(x).squeeze(-1).transpose(-1, -2)
        w = self.sig(self.conv(w)).transpose(-1, -2).unsqueeze(-1)
        return x * w


# ── Cross-Feature Cross-Attention (CFCA) ─────────────────────
# bidirectional attention between Ghost (local) and Global streams
# local attends to global context → picks up lesion extent
# global attends to local detail  → picks up fine boundaries
class CFCAModule(nn.Module):
    def __init__(self, ghost_channels, global_channels, heads=4):
        super().__init__()

        # align both streams to same channel dim
        self.align_ch = min(ghost_channels, global_channels)

        self.proj_ghost  = nn.Sequential(
            nn.Conv2d(ghost_channels,  self.align_ch, 1, bias=False),
            nn.BatchNorm2d(self.align_ch),
            nn.ReLU(inplace=True)
        )
        self.proj_global = nn.Sequential(
            nn.Conv2d(global_channels, self.align_ch, 1, bias=False),
            nn.BatchNorm2d(self.align_ch),
            nn.ReLU(inplace=True)
        )

        # ECA on each stream before attention
        self.eca_ghost  = ECA(self.align_ch)
        self.eca_global = ECA(self.align_ch)

        # cross attention: ghost queries global
        self.attn_g2t = nn.MultiheadAttention(
            self.align_ch, heads, batch_first=True, dropout=0.1
        )
        # cross attention: global queries ghost
        self.attn_t2g = nn.MultiheadAttention(
            self.align_ch, heads, batch_first=True, dropout=0.1
        )

        # project back to original channel dims
        self.out_ghost  = nn.Sequential(
            nn.Conv2d(self.align_ch, ghost_channels,  1, bias=False),
            nn.BatchNorm2d(ghost_channels),
            nn.ReLU(inplace=True)
        )
        self.out_global = nn.Sequential(
            nn.Conv2d(self.align_ch, global_channels, 1, bias=False),
            nn.BatchNorm2d(global_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ghost_feat, global_feat):
        B, _, H, W = ghost_feat.shape

        # align channels + ECA recalibration
        g = self.eca_ghost(self.proj_ghost(ghost_feat))
        t = self.eca_global(self.proj_global(global_feat))

        # align spatial sizes
        if g.shape[2:] != t.shape[2:]:
            t = F.interpolate(t, size=g.shape[2:],
                              mode="bilinear", align_corners=False)

        # ── pool to max 16×16 before attention ──
        # this keeps attention tractable regardless of input resolution
        attn_size = (min(H, 16), min(W, 16))
        g_pool = F.adaptive_avg_pool2d(g, attn_size)
        t_pool = F.adaptive_avg_pool2d(t, attn_size)

        C    = self.align_ch
        Hp,Wp = attn_size

        g_seq = g_pool.flatten(2).transpose(1, 2)  # (B, Hp*Wp, C)
        t_seq = t_pool.flatten(2).transpose(1, 2)

        g_out, _ = self.attn_g2t(g_seq, t_seq, t_seq)
        t_out, _ = self.attn_t2g(t_seq, g_seq, g_seq)

        g_out = g_out.transpose(1, 2).view(B, C, Hp, Wp)
        t_out = t_out.transpose(1, 2).view(B, C, Hp, Wp)

        # upsample attention output back to original spatial size
        g_out = F.interpolate(g_out, size=(H, W),
                              mode="bilinear", align_corners=False)
        t_out = F.interpolate(t_out, size=(H, W),
                              mode="bilinear", align_corners=False)

        ghost_out  = self.out_ghost(g_out)  + ghost_feat
        global_out = self.out_global(t_out) + F.interpolate(
            global_feat, size=ghost_feat.shape[2:],
            mode="bilinear", align_corners=False
        )

        return ghost_out, global_out


# ── Cross-Feature Fusion Bottleneck (XFF) ────────────────────
# merges deepest Ghost + Global features into compact bottleneck
class XFFBottleneck(nn.Module):
    def __init__(self, ghost_channels, global_channels, out_channels):
        super().__init__()

        self.proj_ghost  = nn.Conv2d(ghost_channels,  out_channels, 1, bias=False)
        self.proj_global = nn.Conv2d(global_channels, out_channels, 1, bias=False)

        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.eca = ECA(out_channels)

    def forward(self, ghost_feat, global_feat):
        # align spatial size
        if ghost_feat.shape[2:] != global_feat.shape[2:]:
            global_feat = F.interpolate(
                global_feat, size=ghost_feat.shape[2:],
                mode="bilinear", align_corners=False
            )

        g = self.proj_ghost(ghost_feat)
        t = self.proj_global(global_feat)

        fused = self.eca(self.fuse(g + t))
        return fused


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    # simulate stage3 pairing: Ghost f3 + Global t3
    ghost_feat  = torch.randn(2, 512, 32, 32)
    global_feat = torch.randn(2, 512, 32, 32)

    cfca = CFCAModule(ghost_channels=512, global_channels=512, heads=4)
    g_out, t_out = cfca(ghost_feat, global_feat)
    print(f"  CFCA ghost  out : {tuple(g_out.shape)}")
    print(f"  CFCA global out : {tuple(t_out.shape)}")

    # simulate XFF bottleneck
    xff = XFFBottleneck(512, 512, 512)
    bottleneck = xff(ghost_feat, global_feat)
    print(f"  XFF bottleneck  : {tuple(bottleneck.shape)}")

    params = (
        sum(p.numel() for p in cfca.parameters()) +
        sum(p.numel() for p in xff.parameters())
    ) / 1e6
    print(f"\n  CFCA + XFF params : {params:.2f}M")
    print("  CFCA module OK")