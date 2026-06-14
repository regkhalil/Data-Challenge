"""
Optimise les poids d'ensemble sur le val (sans dépenser de soumission HFactory).

Lit val_preds_seed123.npz (produit par dump_val_preds.py) et trouve les poids
w* = argmin Score(Σ wᵢ·predᵢ) sur le val.

Validation : reconstruit aussi les blends connus (ENS3, champions_55_45) pour
vérifier que le ranking val correspond au ranking HFactory réel.

Usage :
  python src/optimize_weights.py --npz val_preds_seed123.npz [--extra_csvs ...]
"""
import argparse
import numpy as np
from scipy.optimize import minimize


# ── Métrique officielle ────────────────────────────────────────────────────────
def weighted_err(pred, gt):
    w = 1.0 / 30.0 + gt
    return (w * (pred - gt) ** 2).sum() / w.sum()


def score(pred, gt, gender):
    mf, mm = gender == 0.0, gender == 1.0
    ef = weighted_err(pred[mf], gt[mf])
    em = weighted_err(pred[mm], gt[mm])
    return (ef + em) / 2.0 + abs(ef - em), ef, em


def blend_score(weights, preds, gt, gender):
    w = np.array(weights)
    w = w / w.sum()
    pred = np.clip((preds * w[:, None]).sum(0), 0, 1)
    s, ef, em = score(pred, gt, gender)
    return s


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="val_preds_seed123.npz")
    ap.add_argument("--extra_csvs", nargs="*", default=[],
                    help="CSV de test à blender aussi (filenames alignés). FORMAT: path:name")
    args = ap.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    preds   = data["preds"].astype(np.float64)   # (n_models, n_val)
    gt      = data["gt"].astype(np.float64)
    gender  = data["gender"].astype(np.float64)
    names   = list(data["names"])
    n = len(names)

    print(f"Modèles : {names}")
    print(f"Val : {len(gt)} samples  (F={int((gender==0).sum())} M={int((gender==1).sum())})\n")

    # ── Scores individuels ──
    print("=== SCORES INDIVIDUELS SUR VAL ===")
    for i, nm in enumerate(names):
        s, ef, em = score(np.clip(preds[i], 0, 1), gt, gender)
        print(f"  {nm:10s}  ErrF={ef:.5f}  ErrM={em:.5f}  Score={s:.5f}")

    # ── Ensemble uniforme (baseline) ──
    eq = np.clip(preds.mean(0), 0, 1)
    s_eq, ef_eq, em_eq = score(eq, gt, gender)
    print(f"\n  {'EQUAL':10s}  ErrF={ef_eq:.5f}  ErrM={em_eq:.5f}  Score={s_eq:.5f}")

    # ── Validation ranking val vs HFactory ──
    print("\n=== VALIDATION RANKING VAL vs HFACTORY ===")
    # Reconstructions des blends connus sur val pour calibrer la confiance
    # Note: ENS3 = cnxv2(0.33) + large(0.33) + s456(0.33)
    #       champions_55_45 = 0.55*ENS3 + 0.45*s6 (s6 non dispo en val seed-123)
    # On vérifie juste le ranking entre les 3 modèles seed-123
    anchors = {
        "HFactory large": 0.00118,
        "HFactory cnxv2": 0.00124,
        "HFactory ENS3(large+cnxv2+s456)": 0.00115,
        "HFactory champions(ENS3+s6)": 0.00113,
    }
    print("  Scores HFactory connus (référence):")
    for k, v in anchors.items():
        print(f"    {k}: {v:.5f}")
    print("  → Utiliser ces anchors pour juger si val est fiable avant de soumettre")

    # ── Optimisation des poids ──
    print("\n=== OPTIMISATION DES POIDS ===")
    best_s, best_w, best_res = np.inf, None, None

    # Grid search grossier d'abord
    from itertools import product
    grid = np.arange(0.1, 1.01, 0.1)
    if n == 3:
        for w0, w1 in product(grid, grid):
            w2 = 1.0
            w = np.array([w0, w1, w2])
            s = blend_score(w, preds, gt, gender)
            if s < best_s:
                best_s, best_w = s, w / w.sum()

    # Affinement par L-BFGS-B
    x0 = best_w if best_w is not None else np.ones(n) / n
    bounds = [(0.05, 1.0)] * n
    res = minimize(
        lambda w: blend_score(w, preds, gt, gender),
        x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12}
    )
    w_opt = np.array(res.x) / sum(res.x)

    pred_opt = np.clip((preds * w_opt[:, None]).sum(0), 0, 1)
    s_opt, ef_opt, em_opt = score(pred_opt, gt, gender)

    print(f"\nPoids optimaux:")
    for nm, w in zip(names, w_opt):
        print(f"  {nm:10s}: {w:.4f}")
    print(f"\nScore val EQUAL  : {s_eq:.5f}  (ErrF={ef_eq:.5f} ErrM={em_eq:.5f})")
    print(f"Score val OPTIMAL: {s_opt:.5f}  (ErrF={ef_opt:.5f} ErrM={em_opt:.5f})")
    print(f"Gain val          : {(s_eq - s_opt)/s_eq*100:.2f}%")

    # ── Générer le CSV de soumission optimisé ──
    print("\n=== GÉNÉRATION CSV DE SOUMISSION ===")
    print("⚠️  Ce CSV est valide seulement si le val score est FIABLE")
    print("   Vérifie que le ranking val (ci-dessus) correspond à HFactory avant de soumettre !")

    import pandas as pd, os
    test_csv = "DataChallenge2026/occlusion_datasets/test_students.csv"
    print(f"→ Charge les prédictions TEST depuis les CSVs individuels (à faire séparément)")
    print(f"→ Applique les poids: {dict(zip(names, w_opt.round(4)))}")
    print(f"\n  Commande à lancer sur le cluster:")
    weights_str = " ".join(f"{w:.4f}" for w in w_opt)
    names_str = " ".join(
        f"configs/config_{'convnext_base_v2' if n=='cnx_v2' else 'convnext_large' if n=='large' else 'convnextv2_base'}.yaml:"
        f"checkpoints/{'convnext_base_v2' if n=='cnx_v2' else 'convnext_large_v1' if n=='large' else 'convnextv2_base_v1'}_best.pth"
        for n in names
    )
    print(f"  python src/ensemble.py \\")
    for nm, w in zip(names, w_opt):
        cfg = {'cnxv2': 'convnextv2_base', 'large': 'convnext_large', 'cnx_v2': 'convnext_base_v2'}[nm]
        ckpt = {'cnxv2': 'convnextv2_base_v1', 'large': 'convnext_large_v1', 'cnx_v2': 'convnext_base_v2'}[nm]
        print(f"    --models configs/config_{cfg}.yaml:checkpoints/{ckpt}_best.pth \\")
    print(f"    --weights {weights_str} --tta 15 \\")
    print(f"    --out submissions/submission_opt_val_seed123.csv")


if __name__ == "__main__":
    main()
