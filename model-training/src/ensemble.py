import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OcclusionDataset, make_transforms
from model import OcclusionModel


@torch.no_grad()
def predict_one(cfg: dict, checkpoint: str, tta: int, device: torch.device) -> np.ndarray:
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
    print(f"  Chargé : {checkpoint} (backbone={model_cfg['backbone']}, score={ckpt.get('score', '?'):.4f})")

    df_test = pd.read_csv(cfg["data"]["test_csv"])
    image_root = cfg["data"]["image_root"]

    all_passes = []
    for i in range(tta):
        tf = make_transforms(train=(i > 0), cfg=cfg)
        ds = OcclusionDataset(df_test, image_root, tf, has_label=False)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
        preds = []
        for imgs, _ in tqdm(loader, desc=f"  TTA {i+1}/{tta}", leave=False):
            preds.append(model(imgs.to(device)).cpu())
        all_passes.append(torch.cat(preds).numpy())

    return np.stack(all_passes).mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", required=True,
        metavar="CONFIG:CHECKPOINT",
        help="Paires config:checkpoint, ex: configs/config_b4_base.yaml:checkpoints/run_b4_best.pth",
    )
    parser.add_argument("--tta", type=int, default=1, help="Passes TTA par modèle")
    parser.add_argument("--weights", nargs="+", type=float, default=None,
                        help="Poids par modèle (défaut: égaux). Ex: --weights 0.6 0.4")
    parser.add_argument("--out", default=None, help="Chemin de sortie CSV")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    weights = args.weights
    if weights is None:
        weights = [1.0] * len(args.models)
    if len(weights) != len(args.models):
        raise ValueError("--weights doit avoir autant de valeurs que --models")
    weights = np.array(weights) / sum(weights)

    all_preds = []
    test_csv = None
    for pair, w in zip(args.models, weights):
        cfg_path, ckpt_path = pair.split(":")
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        test_csv = cfg["data"]["test_csv"]
        print(f"\nModèle (poids={w:.2f}) : {cfg['model']['backbone']}")
        preds = predict_one(cfg, ckpt_path, args.tta, device)
        all_preds.append(preds * w)

    final_preds = np.clip(sum(all_preds), 0, 1)

    df_test = pd.read_csv(test_csv)
    os.makedirs("submissions", exist_ok=True)
    out_path = args.out or "submissions/submission_ensemble.csv"
    pd.DataFrame({
        "filename": df_test["filename"],
        "FaceOcclusion": final_preds,
        "gender": df_test["gender"] if "gender" in df_test.columns else "x",
    }).to_csv(out_path, index=False)
    print(f"\nEnsemble sauvegardé : {out_path} ({len(df_test)} lignes)")
    print(f"Prédictions — min={final_preds.min():.4f} mean={final_preds.mean():.4f} max={final_preds.max():.4f}")


if __name__ == "__main__":
    main()
