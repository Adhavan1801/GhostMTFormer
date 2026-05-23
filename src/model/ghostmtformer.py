# ─────────────────────────────────────────
#  src/model/ghostmtformer.py
#  Full GhostMTFormer Assembly
# ─────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.model.ghost_encoder  import GhostEncoder
from src.model.global_encoder import GlobalEncoder
from src.model.cfca           import CFCAModule, XFFBottleneck
from src.model.decoder        import Decoder


class GhostMTFormer(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        base   = cfg["model"]["base_channels"]      # 64
        ds     = cfg["model"]["deep_supervision"]   # True

        # ── Encoders ──────────────────────────────
        self.ghost_enc  = GhostEncoder(in_channels=3, base=base)
        self.global_enc = GlobalEncoder(in_channels=3, base=base)

        # ── CFCA at three scales ──────────────────
        # pairing Ghost f2(256) ↔ Global t1(128) 
        self.cfca1 = CFCAModule(base * 4,  base * 2,  heads=4)
        # pairing Ghost f3(512) ↔ Global t2(256) 
        self.cfca2 = CFCAModule(base * 8,  base * 4,  heads=4)
        # pairing Ghost f4(1024) ↔ Global t3(512) 
        self.cfca3 = CFCAModule(base * 16,  base * 8,  heads=4)

        # ── XFF Bottleneck ────────────────────────
        # fuses Ghost f4(1024) + Global t4(1024) → 1024
        self.xff = XFFBottleneck(base * 16, base * 16, base * 16)

        # ── Decoder ───────────────────────────────
        self.decoder = Decoder(base=base, deep_supervision=ds)

        # ── MC Dropout for uncertainty ────────────
        self.mc_dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        target_size = x.shape[2:]   # (H, W) = (512, 512)

        # ── Extract features ──────────────────────
        f0, f1, f2, f3, f4 = self.ghost_enc(x)
        t1, t2, t3, t4     = self.global_enc(x)

        # ── Cross-stream attention at 3 scales ────
        f2_enh, t1_enh = self.cfca1(f2, t1)  # 128-ch scale
        f3_enh, t2_enh = self.cfca2(f3, t2)  # 256-ch scale
        f4_enh, t3_enh = self.cfca3(f4, t3)  # 512-ch scale

        # ── Fuse deepest features → bottleneck ───
        bottleneck = self.xff(f4_enh, t4)     # (B, 1024, 16, 16)
        bottleneck = self.mc_dropout(bottleneck)

        # ── Decode with enhanced skips ────────────
        skips   = (f0, f1, f2_enh, f3_enh)
        outputs = self.decoder(bottleneck, skips, target_size)

        return outputs

    def predict(self, x, threshold=0.5):
        """Clean inference — returns binary mask only."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            prob    = torch.sigmoid(outputs["seg"])
            return (prob > threshold).float()

    def count_params(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total / 1e6, trainable / 1e6


# ── Sanity check ──────────────────────────────────────────────
if __name__ == "__main__":
    import yaml

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device : {device}")

    model = GhostMTFormer(cfg).to(device)
    x     = torch.randn(2, 3, 512, 512).to(device)

    outputs = model(x)

    print(f"\n  seg  : {tuple(outputs['seg'].shape)}")
    print(f"  edge : {tuple(outputs['edge'].shape)}")
    print(f"  ds1  : {tuple(outputs['ds1'].shape)}")
    print(f"  ds2  : {tuple(outputs['ds2'].shape)}")
    print(f"  ds3  : {tuple(outputs['ds3'].shape)}")
    print(f"  boundaries : {len(outputs['boundaries'])} maps")

    total, trainable = model.count_params()
    print(f"\n  Total params     : {total:.2f}M")
    print(f"  Trainable params : {trainable:.2f}M")
    print("\n  GhostMTFormer full model OK")