"""
Décomposition de l'erreur pondérée sur le split de validation.

Répond empiriquement à : "QUELS samples tuent notre score ?"
Pas par l'œil — par la mesure exacte de w_i·(pred_i − GT_i)².

Usage :
  python src/analyze_val_error.py \
      --config configs/config_convnext_base_v2.yaml \
      --checkpoint checkpoints/convnext_base_v2_best.pth
"""

import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

mp.set_sharing_strategy("file_system")

from dataset import OcclusionDataset, make_transforms, load_splits
from model import OcclusionModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="logs/val_error_decomp.csv")
    parser.add_argument("--topk", type=int, default=100)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Même split que l'entraînement de ce modèle (seed dans le config)
    _, df_val, _ = load_splits(cfg)
    image_root = cfg["data"]["image_root"]
    val_tf = make_transforms(train=False, cfg=cfg)
    ds = OcclusionDataset(df_val, image_root, val_tf)  # pas de dégradation
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"],
                        shuffle=False, num_workers=0)

    m_cfg = cfg["model"]
    model = OcclusionModel(m_cfg["backbone"], pretrained=False,
                           dropout=m_cfg["dropout"], head_dims=m_cfg["head_dims"]).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    preds, gts, genders = [], [], []
    with torch.no_grad():
        for imgs, gt, g in loader:
            imgs = imgs.to(device)
            p = model(imgs).cpu().numpy()
            preds.append(p)
            gts.append(gt.numpy())
            genders.append(g.numpy())

    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    genders = np.concatenate(genders)

    w = 1.0 / 30.0 + gts
    err_contrib = w * (preds - gts) ** 2          # numérateur par sample

    # --- Score officiel reconstitué ---
    def werr(mask):
        if mask.sum() == 0:
            return 0.0
        return (w[mask] * (preds[mask] - gts[mask]) ** 2).sum() / w[mask].sum()
    errF = werr(genders == 0.0)
    errM = werr(genders == 1.0)
    score = (errF + errM) / 2 + abs(errF - errM)
    print(f"\n=== SCORE VAL : {score:.6f}  (ErrF={errF:.6f}  ErrM={errM:.6f}  |diff|={abs(errF-errM):.6f}) ===\n")

    # --- Agrégat par bucket GT : où se concentre l'erreur ? ---
    print("=== Contribution à l'erreur par bucket GT ===")
    print(f"{'bucket':>12} {'n':>6} {'%err_tot':>9} {'mean_pred':>10} {'mean_gt':>8} {'mean_|err|':>10}")
    edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.01]
    tot = err_contrib.sum()
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (gts >= lo) & (gts < hi)
        if mask.sum() == 0:
            continue
        share = 100 * err_contrib[mask].sum() / tot
        print(f"  [{lo:.1f}-{hi:.1f}] {mask.sum():>6} {share:>8.1f}% "
              f"{preds[mask].mean():>10.3f} {gts[mask].mean():>8.3f} "
              f"{np.abs(preds[mask]-gts[mask]).mean():>10.3f}")

    # --- Agrégat par genre × bucket : source du |ErrF-ErrM| ---
    print("\n=== Contribution par genre ===")
    for gname, gval in [("Femme", 0.0), ("Homme", 1.0)]:
        gm = genders == gval
        share = 100 * err_contrib[gm].sum() / tot
        print(f"  {gname:>6} : n={gm.sum():>6}  share_err={share:>5.1f}%  "
              f"mean_pred={preds[gm].mean():.3f}  mean_gt={gts[gm].mean():.3f}")

    # --- Top-K pires contributeurs : à inspecter visuellement ---
    order = np.argsort(-err_contrib)[:args.topk]
    rows = []
    for i in order:
        rows.append({
            "filename": df_val.iloc[i]["filename"],
            "gt": round(float(gts[i]), 4),
            "pred": round(float(preds[i]), 4),
            "gender": float(genders[i]),
            "err_contrib": round(float(err_contrib[i]), 6),
            "pct_of_total": round(100 * float(err_contrib[i]) / tot, 3),
        })
    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out, index=False)
    topshare = 100 * err_contrib[order].sum() / tot
    print(f"\n=== TOP-{args.topk} pires samples = {topshare:.1f}% de l'erreur totale ===")
    print(out_df.head(25).to_string(index=False))
    print(f"\nSauvegardé : {args.out}")


if __name__ == "__main__":
    main()
