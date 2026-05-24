# GhostMTFormer

> Lightweight dual-encoder network for skin lesion segmentation on HAM10000

---

## Architecture
![GhostMTFormer Architecture](docs/GhostMTFormer_model_architecture.png)


---

## Dataset

[HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) — 10,015 dermoscopic images with expert segmentation masks

| Split | Images |
|-------|--------|
| Train | 8,012  |
| Val   | 1,001  |
| Test  | 1,002  |

---

## Results

> Evaluated on HAM10000 test set (1,002 images) with Test-Time Augmentation (TTA)

| Metric | Score |
|--------|-------|
| Dice   | 93.998% ± 8.65% |
| IoU    | 89.63% ± 11.89% |
| HD95   | 3.24 px |

*Results will be updated after full training completes.*

---

## Project Structure
GhostMTFormer/
├── configs/
│   └── default.yaml          # all hyperparameters
├── src/
│   ├── dataset.py            # HAM10000 data pipeline
│   ├── losses.py             # Dice + BCE + Tversky + Focal + Boundary
│   ├── metrics.py            # Dice, IoU, HD95
│   ├── train.py              # training loop
│   ├── evaluate.py           # test evaluation + TTA
│   └── model/
│       ├── ghost_encoder.py  # GhostNet local encoder
│       ├── global_encoder.py # CNN global encoder
│       ├── cfca.py           # cross-feature attention + XFF bottleneck
│       ├── decoder.py        # boundary-refined decoder
│       └── ghostmtformer.py  # full model assembly
├── notebooks/
│   └── gradcam_analysis.ipynb
└── requirements.txt

---

## Setup

```bash
git clone https://github.com/adhavan1801/GhostMTFormer.git
cd GhostMTFormer

python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

---

## Training

```bash
python -m src.train
```

---

## Evaluation

```bash
python -m src.evaluate
```

---

## Environment

- Python 3.11
- PyTorch 2.7 + CUDA 12.8
- RTX 5060 8GB VRAM

---

## References

- [GhostNet](https://arxiv.org/abs/1911.11907) — Han et al., CVPR 2020
- [CFFormer](https://www.sciencedirect.com/science/article/pii/S0957417425003702) — Zhang et al., Expert Systems 2025
- [ECA-Net](https://arxiv.org/abs/1910.03151) — Wang et al., CVPR 2020
- [HAM10000](https://arxiv.org/abs/1803.10417) — Tschandl et al., 2018