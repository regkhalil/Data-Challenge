"""
Mega-Ensemble : moyenne pondérée de fichiers CSV de prédictions déjà calculés.

Usage :
  python src/ensemble_csv.py \
      --csvs submissions/submission_b4_v2_s123.csv \
              submissions/submission_convnext_base_v1.csv \
              submissions/submission_vit_base_v1.csv \
      --weights 0.3 0.4 0.3 \
      --out submissions/submission_mega_ensemble.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csvs", nargs="+", required=True, help="CSV de prédictions à fusionner")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Poids par modèle (défaut: égaux)")
    parser.add_argument("--out", default="submissions/submission_mega_ensemble.csv")
    args = parser.parse_args()

    weights = args.weights if args.weights else [1.0] * len(args.csvs)
    if len(weights) != len(args.csvs):
        raise ValueError("--weights doit avoir autant de valeurs que --csvs")
    weights = np.array(weights, dtype=float) / sum(weights)

    base_df = None
    ensemble_preds = np.zeros(0)

    for csv_path, w in zip(args.csvs, weights):
        df = pd.read_csv(csv_path)
        pred_col = "FaceOcclusion" if "FaceOcclusion" in df.columns else "prediction"
        preds = df[pred_col].values.astype(float)

        if base_df is None:
            base_df = df.copy()
            ensemble_preds = np.zeros(len(preds))

        if len(preds) != len(ensemble_preds):
            raise ValueError(f"Taille incohérente : {csv_path} ({len(preds)}) vs référence ({len(ensemble_preds)})")

        ensemble_preds += w * preds
        print(f"  {os.path.basename(csv_path)} (poids={w:.3f}) — mean={preds.mean():.4f} min={preds.min():.4f} max={preds.max():.4f}")

    final_preds = np.clip(ensemble_preds, 0.0, 1.0)

    pred_col = "FaceOcclusion" if "FaceOcclusion" in base_df.columns else "prediction"
    base_df[pred_col] = final_preds
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    base_df.to_csv(args.out, index=False)

    print(f"\nEnsemble sauvegardé : {args.out} ({len(base_df)} lignes)")
    print(f"Prédictions finales — min={final_preds.min():.4f}  mean={final_preds.mean():.4f}  max={final_preds.max():.4f}")


if __name__ == "__main__":
    main()
