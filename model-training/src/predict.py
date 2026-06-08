import os
import argparse
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import OcclusionDataset, make_transforms
from model import OcclusionModel


@torch.no_grad()
def predict(cfg_path: str, checkpoint: str, tta: int = 1):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Charger le modèle
    ckpt = torch.load(checkpoint, map_location=device)
    model_cfg = cfg["model"]
    model = OcclusionModel(
        backbone=model_cfg["backbone"],
        pretrained=False,
        dropout=model_cfg["dropout"],
        head_dims=model_cfg.get("head_dims", [512, 128]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    df_test = pd.read_csv(cfg["data"]["test_csv"])
    image_root = cfg["data"]["image_root"]

    # TTA : plusieurs passes avec augmentations, moyenne des prédictions
    all_preds = []
    for i in range(tta):
        use_aug = (i > 0)  # première passe sans augmentation
        tf = make_transforms(train=use_aug, cfg=cfg)
        ds = OcclusionDataset(df_test, image_root, tf, has_label=False)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)

        preds = []
        for imgs, _ in tqdm(loader, desc=f"TTA {i+1}/{tta}"):
            imgs = imgs.to(device)
            preds.append(model(imgs).cpu())
        all_preds.append(torch.cat(preds))

    final_preds = torch.stack(all_preds).mean(0).clamp(0, 1).numpy()

    os.makedirs(cfg["output"]["submission_dir"], exist_ok=True)
    exp = cfg["output"]["experiment_name"]
    out_path = os.path.join(cfg["output"]["submission_dir"], f"submission_{exp}.csv")

    df_out = pd.DataFrame({
        "filename": df_test["filename"],
        "FaceOcclusion": final_preds,
        "gender": df_test["gender"] if "gender" in df_test.columns else "x",
    })
    df_out.to_csv(out_path, index=False)
    print(f"Soumission sauvegardée : {out_path} ({len(df_out)} lignes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tta", type=int, default=1, help="Nombre de passes TTA")
    args = parser.parse_args()
    predict(args.config, args.checkpoint, args.tta)
