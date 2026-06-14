"""
Dump les prédictions VAL de plusieurs modèles partageant le MÊME seed (donc le
même split val → zéro fuite de données). Sauvegarde un .npz avec :
  - preds : matrice (n_models, n_val) des prédictions
  - gt    : ground truth (n_val,)
  - gender: genre (n_val,)  [0=F, 1=M]
  - names : noms des modèles

Sert ensuite à optimiser localement les poids d'ensemble pour minimiser le
Score officiel (ErrF+ErrM)/2 + |ErrF-ErrM|, SANS dépenser de soumission HFactory.

Usage :
  python src/dump_val_preds.py \
    --models configs/config_convnextv2_base.yaml:checkpoints/convnextv2_base_v1_best.pth:cnxv2 \
             configs/config_convnext_large.yaml:checkpoints/convnext_large_v1_best.pth:large \
             configs/config_convnext_base_v2.yaml:checkpoints/convnext_base_v2_best.pth:cnx_v2 \
    --tta 8 --out val_preds_seed123.npz
"""
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OcclusionDataset, make_transforms, load_splits
from model import OcclusionModel


@torch.no_grad()
def predict_val(cfg, checkpoint, tta, device):
    m = cfg["model"]
    model = OcclusionModel(
        backbone=m["backbone"], pretrained=False,
        dropout=m["dropout"], head_dims=m.get("head_dims", [512, 128]),
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, df_val, _ = load_splits(cfg)
    image_root = cfg["data"]["image_root"]

    gt = df_val["FaceOcclusion"].values.astype(np.float32)
    gender = df_val["gender"].values.astype(np.float32)

    passes = []
    for i in range(tta):
        tf = make_transforms(train=(i > 0), cfg=cfg)
        ds = OcclusionDataset(df_val, image_root, tf)  # returns (img, gt, gender)
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        preds = []
        for imgs, _, _ in tqdm(loader, desc=f"  TTA {i+1}/{tta}", leave=False):
            preds.append(model(imgs.to(device)).cpu())
        passes.append(torch.cat(preds).numpy())
    return np.stack(passes).mean(0), gt, gender


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    metavar="CONFIG:CHECKPOINT:NAME")
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--out", default="val_preds.npz")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    all_preds, names = [], []
    ref_gt = ref_gender = ref_seed = None
    for spec in args.models:
        cfg_path, ckpt_path, name = spec.split(":")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        seed = cfg["data"]["seed"]
        if ref_seed is None:
            ref_seed = seed
        elif seed != ref_seed:
            raise ValueError(f"{name} a seed={seed} ≠ {ref_seed} → val sets différents, fuite !")
        print(f"\n{name} (backbone={cfg['model']['backbone']}, seed={seed})")
        preds, gt, gender = predict_val(cfg, ckpt_path, args.tta, device)
        if ref_gt is None:
            ref_gt, ref_gender = gt, gender
        else:
            assert np.allclose(ref_gt, gt), f"{name} : GT désaligné !"
        all_preds.append(preds)
        names.append(name)

    preds = np.stack(all_preds)
    np.savez(args.out, preds=preds, gt=ref_gt, gender=ref_gender, names=np.array(names))
    print(f"\n✅ Sauvegardé : {args.out}  (preds {preds.shape}, val n={len(ref_gt)})")

    # Score individuel rapide
    from loss import compute_score
    for i, nm in enumerate(names):
        ef, em, sc = compute_score(
            torch.tensor(preds[i]), torch.tensor(ref_gt), torch.tensor(ref_gender))
        print(f"  {nm:8s} ErrF={ef:.5f} ErrM={em:.5f} Score={sc:.5f}")


if __name__ == "__main__":
    main()
