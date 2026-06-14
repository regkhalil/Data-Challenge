"""
Ensemble max(specialist, generalist) — sample-par-sample.

Le spécialiste (dégradation-aware) peut seulement POUSSER vers le haut.
Le généraliste garde le contrôle sur le bulk (99.7% des samples).

Usage :
  python src/ensemble_max.py \
      --specialist submissions/submission_s9_new3_ensemble.csv \
      --generalist submissions/submission_s6_cnx_v1v2_b4.csv \
      --out submissions/submission_s10_max.csv
"""

import argparse
import os
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialist", required=True, help="CSV du spécialiste dégradation (nouveau)")
    parser.add_argument("--generalist", required=True, help="CSV du généraliste (ancien ensemble)")
    parser.add_argument("--out", default="submissions/submission_ensemble_max.csv")
    args = parser.parse_args()

    def load(path):
        df = pd.read_csv(path)
        col = "FaceOcclusion" if "FaceOcclusion" in df.columns else "prediction"
        return df, col

    spec_df, spec_col = load(args.specialist)
    gen_df, gen_col = load(args.generalist)

    if len(spec_df) != len(gen_df):
        raise ValueError(f"Tailles incompatibles : {len(spec_df)} vs {len(gen_df)}")

    spec_preds = spec_df[spec_col].values.astype(float)
    gen_preds = gen_df[gen_col].values.astype(float)

    final_preds = np.maximum(spec_preds, gen_preds)
    final_preds = np.clip(final_preds, 0.0, 1.0)

    pushed_up = (final_preds > gen_preds + 0.01).sum()
    print(f"Spécialiste   — mean={spec_preds.mean():.4f}  max={spec_preds.max():.4f}")
    print(f"Généraliste   — mean={gen_preds.mean():.4f}   max={gen_preds.max():.4f}")
    print(f"Max ensemble  — mean={final_preds.mean():.4f}  max={final_preds.max():.4f}")
    print(f"Samples poussés à la hausse par le spécialiste : {pushed_up} / {len(final_preds)}")

    out_df = gen_df.copy()
    out_df[gen_col] = final_preds
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nSauvegardé : {args.out} ({len(out_df)} lignes)")


if __name__ == "__main__":
    main()
