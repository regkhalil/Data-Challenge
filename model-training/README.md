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
├── configs/            # One YAML config per experiment
├── scripts/            # SLURM batch scripts for the Telecom Paris cluster
└── train-free/         # Training-free baseline (separate track)
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
python src/train.py --config configs/config_b4_base.yaml --debug

# Full training
python src/train.py --config configs/config_b4_base.yaml
```

Checkpoints are saved to `checkpoints/<exp>_best.pth`. Logs are written to `logs/<exp>.csv`.

## Inference

```bash
# Single model, no TTA
python src/predict.py --config configs/config_b4_base.yaml \
    --checkpoint checkpoints/run_b4_best.pth

# With TTA (5 passes)
python src/predict.py --config configs/config_b4_base.yaml \
    --checkpoint checkpoints/run_b4_best.pth --tta 5
```

## Ensemble

```bash
python src/ensemble.py \
    --models configs/config_b4_base.yaml:checkpoints/run_b4_best.pth \
             configs/config_convnext.yaml:checkpoints/convnext_small_v1_best.pth \
    --tta 5 \
    --weights 0.5 0.5 \
    --out submissions/submission_ensemble.csv
```

## Reproducing Results

| Experiment | Config | Best val score |
|---|---|---|
| EfficientNet-B4 (seed=42) | `config_b4_base.yaml` | 0.001656 |
| EfficientNet-B4 (seed=123, boost=3.0) | `config_b4_v2.yaml` | 0.001555 |
| ConvNeXt-Small (boost=1.5) | `config_convnext.yaml` | 0.001700 |
| ConvNeXt-Base (boost=2.0) | `config_convnext_base.yaml` | in progress |
| ViT-Base/16 (boost=2.0) | `config_vit_base.yaml` | in progress |
| **Ensemble B4 + ConvNeXt-Small** | — | **0.00123 (interim)** |

## Key Design Decisions

**Weighted sampler** — Sample weights `(1/30 + GT) × gender_factor × male_boost` correct both the occlusion imbalance (69% of samples have GT < 0.1) and the gender imbalance (68% male).

**Loss = metric** — `challenge_loss()` directly optimises the official score, so validation monitoring and training are aligned.

**Stratified split** — 80/20 split stratified on `gender × occlusion_bucket` (seed=42) prevents ordering bias.

**TTA** — First inference pass is clean (no augmentation), remaining passes use training augmentations. Predictions are averaged.
