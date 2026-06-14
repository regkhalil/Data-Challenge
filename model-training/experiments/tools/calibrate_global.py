"""
Calibration GLOBALE post-hoc (pas par genre — le genre du test est caché = "x").

Optimise UN seul couple (alpha, beta) sur le val pour minimiser le Score officiel,
puis l'applique à un CSV de soumission : pred_cal = clip(alpha * pred + beta, 0, 1).

C'est le seul post-hoc honnête possible ici : il corrige un biais systématique
de sur/sous-prédiction du modèle, sans avoir besoin du genre par échantillon.

Usage :
  python src/calibrate_global.py \
      --config configs/config_convnext_base_v2.yaml \
      --checkpoint checkpoints/convnext_base_v2_best.pth \
      --submission submissions/submission_mega_ensemble.csv \
      --out submissions/submission_mega_global_cal.csv
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


@torch.no_grad()
def get_val_predictions(cfg, checkpoint, device):
    _, df_val, _ = load_splits(cfg)
    tf = make_transforms(train=False, cfg=cfg)
    ds = OcclusionDataset(df_val, cfg["data"]["image_root"], tf, has_label=True)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)

    mc = cfg["model"]
    model = OcclusionModel(mc["backbone"], pretrained=False,
                           dropout=mc["dropout"], head_dims=mc.get("head_dims", [512, 128])).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Checkpoint : {checkpoint} (val_score={ckpt.get('score','?')})")
    print(f"Val : {len(df_val)} samples")

    preds, gts, genders = [], [], []
    for imgs, gt, gender in tqdm(loader, desc="Inférence val"):
        preds.append(model(imgs.to(device)).cpu().squeeze())
        gts.append(gt); genders.append(gender)
    return (torch.cat(preds).numpy(),
            torch.cat(gts).float().numpy(),
            torch.cat(genders).float().numpy())


def global_score(params, preds, gt, gender):
    a, b = params
    cal = np.clip(a * preds + b, 0.0, 1.0)
    _, _, s = compute_score(torch.tensor(cal, dtype=torch.float32),
                            torch.tensor(gt, dtype=torch.float32),
                            torch.tensor(gender, dtype=torch.float32))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}\n")

    vp, vg, vgen = get_val_predictions(cfg, args.checkpoint, device)

    ef0, em0, s0 = compute_score(torch.tensor(vp), torch.tensor(vg), torch.tensor(vgen))
    print(f"\n--- Avant ---  ErrF={ef0:.6f} ErrM={em0:.6f} Score={s0:.6f}")

    res = minimize(global_score, [1.0, 0.0], args=(vp, vg, vgen),
                   method="L-BFGS-B", bounds=[(0.5, 1.5), (-0.1, 0.1)],
                   options={"maxiter": 1000, "ftol": 1e-12})
    a, b = res.x
    s1 = res.fun
    print(f"--- Après --- alpha={a:.5f} beta={b:.5f} Score={s1:.6f}")
    print(f"Gain val : {s0:.6f} → {s1:.6f}  ({(s0-s1)/s0*100:.2f}%)")

    if s1 >= s0 - 1e-9:
        print("\n⚠️  Gain négligeable/nul — calibration globale inutile ici (modèle déjà bien calibré).")

    # Application au CSV
    df = pd.read_csv(args.submission)
    col = "FaceOcclusion" if "FaceOcclusion" in df.columns else "prediction"
    df[col] = np.clip(a * df[col].values + b, 0.0, 1.0)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nSauvegardé : {args.out} ({len(df)} lignes) "
          f"min={df[col].min():.4f} mean={df[col].mean():.4f} max={df[col].max():.4f}")


if __name__ == "__main__":
    main()
