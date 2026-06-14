"""
Calibration post-hoc par genre.

Étape 1 : Run le meilleur modèle sur le val set pour obtenir des prédictions réelles.
Étape 2 : Optimise alpha_F, beta_F, alpha_M, beta_M avec scipy pour minimiser le Score val.
Étape 3 : Applique les coefficients au CSV de soumission test.

Usage :
  python src/calibrate_gender.py \
      --config configs/config_convnext_base.yaml \
      --checkpoint checkpoints/convnext_base_v1_best.pth \
      --submission submissions/submission_mega_ensemble.csv \
      --out submissions/submission_mega_calibrated.csv
"""

import argparse
import sys
import os
import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.optimize import minimize
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from dataset import OcclusionDataset, make_transforms, load_splits
from model import OcclusionModel
from loss import compute_score


# ─── Inférence val ────────────────────────────────────────────────────────────

@torch.no_grad()
def get_val_predictions(cfg: dict, checkpoint: str, device: torch.device):
    """Retourne (preds, gt, gender) sur le split de validation."""
    _, df_val, _ = load_splits(cfg)

    tf = make_transforms(train=False, cfg=cfg)
    ds = OcclusionDataset(df_val, cfg["data"]["image_root"], tf, has_label=True)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    model_cfg = cfg["model"]
    model = OcclusionModel(
        backbone=model_cfg["backbone"],
        pretrained=False,
        dropout=model_cfg["dropout"],
        head_dims=model_cfg.get("head_dims", [512, 128]),
    ).to(device)

    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint chargé : {checkpoint}  (val_score={ckpt.get('score', '?'):.5f})")
    print(f"Val set : {len(df_val)} samples")

    all_preds, all_gt, all_gender = [], [], []
    for imgs, gt, gender in tqdm(loader, desc="Inférence val"):
        preds = model(imgs.to(device)).cpu().squeeze()
        all_preds.append(preds)
        all_gt.append(gt)
        all_gender.append(gender)

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_gt).float().numpy(),
        torch.cat(all_gender).float().numpy(),
    )


# ─── Objectif d'optimisation ──────────────────────────────────────────────────

def calibration_score(params, preds, gt, gender):
    """Score officiel après calibration linéaire par genre."""
    alpha_f, beta_f, alpha_m, beta_m = params

    cal = preds.copy()
    mask_f = gender == 0.0
    mask_m = gender == 1.0
    cal[mask_f] = np.clip(alpha_f * preds[mask_f] + beta_f, 0.0, 1.0)
    cal[mask_m] = np.clip(alpha_m * preds[mask_m] + beta_m, 0.0, 1.0)

    _, _, score = compute_score(
        torch.tensor(cal, dtype=torch.float32),
        torch.tensor(gt,  dtype=torch.float32),
        torch.tensor(gender, dtype=torch.float32),
    )
    return score


# ─── Application au test ──────────────────────────────────────────────────────

def apply_calibration(submission_path: str, out_path: str, params, pred_col: str):
    alpha_f, beta_f, alpha_m, beta_m = params
    df = pd.read_csv(submission_path)

    mask_f = df["gender"] == 0.0
    mask_m = df["gender"] == 1.0

    df.loc[mask_f, pred_col] = np.clip(
        alpha_f * df.loc[mask_f, pred_col].values + beta_f, 0.0, 1.0
    )
    df.loc[mask_m, pred_col] = np.clip(
        alpha_m * df.loc[mask_m, pred_col].values + beta_m, 0.0, 1.0
    )

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSoumission calibrée sauvegardée : {out_path} ({len(df)} lignes)")
    print(f"Prédictions — min={df[pred_col].min():.4f}  mean={df[pred_col].mean():.4f}  max={df[pred_col].max():.4f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--submission", required=True, help="CSV test à calibrer")
    parser.add_argument("--out",        default="submissions/submission_mega_calibrated.csv")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    # 1. Prédictions sur val
    val_preds, val_gt, val_gender = get_val_predictions(cfg, args.checkpoint, device)

    # 2. Score de référence (sans calibration)
    err_f0, err_m0, score0 = compute_score(
        torch.tensor(val_preds), torch.tensor(val_gt), torch.tensor(val_gender)
    )
    print(f"\n--- Avant calibration ---")
    print(f"  ErrF = {err_f0:.6f}  |  ErrM = {err_m0:.6f}  |  Score = {score0:.6f}")
    print(f"  |ErrF - ErrM| = {abs(err_f0 - err_m0):.6f}  ({abs(err_f0-err_m0)/score0*100:.1f}% du score)")

    # 3. Optimisation
    print("\nOptimisation en cours...")
    x0 = [1.0, 0.0, 1.0, 0.0]  # alpha_f, beta_f, alpha_m, beta_m

    # Bornes : alpha ∈ [0.5, 2.0], beta ∈ [-0.2, 0.2]
    bounds = [(0.5, 2.0), (-0.2, 0.2), (0.5, 2.0), (-0.2, 0.2)]

    result = minimize(
        calibration_score,
        x0,
        args=(val_preds, val_gt, val_gender),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 1000, "ftol": 1e-10},
    )

    alpha_f, beta_f, alpha_m, beta_m = result.x
    score_cal = result.fun

    err_f1, err_m1, _ = compute_score(
        torch.tensor(np.clip(
            np.where(val_gender == 0.0,
                     alpha_f * val_preds + beta_f,
                     alpha_m * val_preds + beta_m), 0, 1)),
        torch.tensor(val_gt),
        torch.tensor(val_gender),
    )

    print(f"\n--- Après calibration ---")
    print(f"  alpha_F = {alpha_f:.5f}  beta_F = {beta_f:.5f}")
    print(f"  alpha_M = {alpha_m:.5f}  beta_M = {beta_m:.5f}")
    print(f"  ErrF = {err_f1:.6f}  |  ErrM = {err_m1:.6f}  |  Score = {score_cal:.6f}")
    print(f"  |ErrF - ErrM| = {abs(err_f1 - err_m1):.6f}")
    print(f"\n  Gain val : {score0:.6f} → {score_cal:.6f}  ({(score0-score_cal)/score0*100:.2f}% d'amélioration)")

    # 4. Application au test
    pred_col = "FaceOcclusion" if "FaceOcclusion" in pd.read_csv(args.submission).columns else "prediction"
    apply_calibration(args.submission, args.out, result.x, pred_col)


if __name__ == "__main__":
    main()
