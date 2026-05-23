# GhostMTFormer

> Lightweight dual-encoder network for skin lesion segmentation on HAM10000

---

## Architecture

GhostMTFormer combines two complementary encoders with cross-attention fusion and boundary-aware decoding:

- **GhostNet Encoder** — captures local texture and fine lesion edges efficiently using Ghost modules
- **Global CNN Encoder** — captures long-range lesion structure using dilated convolutions
- **CFCA Module** — bidirectional cross-attention between the two streams at 3 scales
- **XFF Bottleneck** — fuses the deepest features from both encoders
- **Boundary-Refined Decoder** — sharpens lesion contours using Boundary Refinement Modules (BRM)
- **Deep Supervision** — auxiliary outputs at every decoder stage for stronger gradient flow
- **Grad-CAM + MC Dropout** — explainability and uncertainty estimation

---

## Dataset

[HAM10000](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification) — 10,015 dermoscopic images with expert segmentation masks

| Split | Images |
|-------|--------|
| Train | 8,012  |
| Val   | 1,001  |
| Test  | 1,002  |

---

## Results

> Evaluated on HAM10000 test set with Test-Time Augmentation (TTA)

| Metric | Score |
|--------|-------|
| Dice   | TBD   |
| IoU    | TBD   |
| HD95   | TBD   |

*Results will be updated after full training completes.*

---

## Project Structure