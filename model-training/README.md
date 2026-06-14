# Data Challenge Telecom Paris — Face Occlusion Prediction

Predict the percentage of face occlusion (`FaceOcclusion ∈ [0, 1]`) for 224×224 face images, evaluated on a gender-balanced weighted MSE metric.

## Evaluation Metric

```
w_i = 1/30 + GT_i
Err(gender) = Σ w_i(p_i − GT_i)² / Σ w_i
Score = (ErrF + ErrM) / 2 + |ErrF − ErrM|    ← minimise
```

High-occlusion samples have up to 31× more weight. The gender disparity penalty `|ErrF − ErrM|` requires both subsets to be calibrated equally.

## Project Structure

```
├── src/
│   ├── dataset.py      # Dataset, transforms, weighted sampler, stratified splits
│   ├── model.py        # timm backbone + FC regression head → Sigmoid
│   ├── loss.py         # Weighted MSE loss matching official metric
│   ├── train.py        # Training loop with logging and checkpointing
│   ├── predict.py      # Inference with optional Test-Time Augmentation (TTA)
│   └── ensemble.py     # Weighted average of multiple model predictions
│   └── ensemble_csv.py # Weighted average of prediction CSVs
├── configs/            # Configs for the seven models of the final ensemble
├── scripts/            # SLURM batch scripts for the Telecom Paris cluster
└── experiments/        # All other approaches explored (see EXPERIMENTS.md)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision timm pandas numpy scikit-learn pyyaml tqdm
```

## Training

```bash
# Quick pipeline check (500 samples, 1 epoch)
python src/train.py --config configs/config_convnext_base.yaml --debug

# Full training
python src/train.py --config configs/config_convnext_base.yaml
```

Checkpoints are saved to `checkpoints/<exp>_best.pth`. Logs are written to `logs/<exp>.csv`.

## Inference

```bash
# Single model with TTA (15 passes)
python src/predict.py --config configs/config_convnext_large.yaml \
    --checkpoint checkpoints/convnext_large_v1_best.pth --tta 15
```

## Reproducing the Final Result (interim score 0.00112)

The final submission is a weighted blend of two ensembles built from seven models.

### 1. Train the seven models

```bash
python src/train.py --config configs/config_convnextv2_base.yaml      # convnextv2_base_v1
python src/train.py --config configs/config_convnext_large.yaml       # convnext_large_v1  (seed 123)
python src/train.py --config configs/config_convnext_large_s999.yaml  # convnext_large_s999
python src/train.py --config configs/config_convnext_base.yaml        # convnext_base_v1   (seed 42)
python src/train.py --config configs/config_convnext_base_v2.yaml     # convnext_base_v2   (seed 123)
python src/train.py --config configs/config_convnext_base_s456.yaml   # convnext_base_s456 (seed 456)
python src/train.py --config configs/config_b4_v2.yaml                # b4_v2_s123
```

### 2. Generate per-model predictions (TTA ×15)

```bash
python src/predict.py --config configs/<config>.yaml \
    --checkpoint checkpoints/<exp>_best.pth --tta 15
```

### 3. Blend into the final submission

```bash
# Strong ConvNeXt ensemble (ENS4)
python src/ensemble_csv.py --out submissions/ens4.csv --csvs \
    submissions/submission_convnextv2_base_v1.csv \
    submissions/submission_convnext_large_v1.csv \
    submissions/submission_convnext_base_s456.csv \
    submissions/submission_convnext_large_s999.csv

# Diverse ensemble (s6)
python src/ensemble_csv.py --out submissions/s6.csv --csvs \
    submissions/submission_convnext_base_v1.csv \
    submissions/submission_convnext_base_v2.csv \
    submissions/submission_b4_v2_s123.csv

# Final blend: 0.55 * ENS4 + 0.45 * s6
python src/ensemble_csv.py --weights 0.55 0.45 \
    --csvs submissions/ens4.csv submissions/s6.csv \
    --out submissions/submission_final.csv
```

### Per-model validation scores

| Config | Backbone | Seed | Best val score |
|---|---|---|---|
| `config_convnext_large_s999.yaml` | ConvNeXt-Large | 999 | 0.0011 |
| `config_convnext_large.yaml` | ConvNeXt-Large | 123 | 0.0011 |
| `config_convnextv2_base.yaml` | ConvNeXtV2-Base | 123 | 0.0014 |
| `config_convnext_base_v2.yaml` | ConvNeXt-Base | 123 | 0.0016 |
| `config_convnext_base.yaml` | ConvNeXt-Base | 42 | 0.0016 |
| `config_convnext_base_s456.yaml` | ConvNeXt-Base | 456 | 0.0018 |
| `config_b4_v2.yaml` | EfficientNet-B4 | 123 | 0.0017 |
| **Final blend (0.55·ENS4 + 0.45·s6)** | — | — | **0.00112 (interim)** |

Other approaches that were explored (synthetic degradation, ViT, Swin, EfficientNetV2,
global calibration) are documented in [`experiments/EXPERIMENTS.md`](experiments/EXPERIMENTS.md).

## Key Design Decisions

**Weighted sampler** — Sample weights `(1/30 + GT) × gender_factor × male_boost` correct both the occlusion imbalance (69% of samples have GT < 0.1) and the gender imbalance (68% male).

**Loss = metric** — `challenge_loss()` directly optimises the official score, so validation monitoring and training are aligned.

**Stratified split** — 80/20 split stratified on `gender × occlusion_bucket` (seed=42) prevents ordering bias.

**TTA** — First inference pass is clean (no augmentation), remaining passes use training augmentations. Predictions are averaged.
