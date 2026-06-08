import os
import csv
import random
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from dataset import OcclusionDataset, make_transforms, make_weighted_sampler, load_splits
from model import OcclusionModel
from loss import challenge_loss, compute_score


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for imgs, gt, gender in tqdm(loader, leave=False, desc="train"):
        imgs, gt, gender = imgs.to(device), gt.float().to(device), gender.float().to(device)
        optimizer.zero_grad()
        pred = model(imgs)
        loss = challenge_loss(pred, gt, gender)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(gt)
        n += len(gt)
    return total_loss / n


@torch.no_grad()
def val_epoch(model, loader, device):
    model.eval()
    preds, gts, genders = [], [], []
    for imgs, gt, gender in tqdm(loader, leave=False, desc="val"):
        imgs = imgs.to(device)
        pred = model(imgs)
        preds.append(pred.cpu())
        gts.append(gt.float())
        genders.append(gender.float())
    preds = torch.cat(preds)
    gts = torch.cat(gts)
    genders = torch.cat(genders)
    err_f, err_m, score = compute_score(preds, gts, genders)
    return err_f, err_m, score


def main(cfg_path: str, debug: bool = False, exp_name: str = None, resume: str = None):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    if exp_name:
        cfg["output"]["experiment_name"] = exp_name

    set_seed(cfg["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df_train, df_val, _ = load_splits(cfg)
    if debug:
        df_train = df_train.head(500)
        df_val = df_val.head(200)

    train_tf = make_transforms(train=True, cfg=cfg)
    val_tf = make_transforms(train=False, cfg=cfg)
    image_root = cfg["data"]["image_root"]

    train_ds = OcclusionDataset(df_train, image_root, train_tf)
    val_ds = OcclusionDataset(df_val, image_root, val_tf)

    sampler = None
    shuffle = True
    if cfg["training"].get("use_weighted_sampler"):
        sampler = make_weighted_sampler(
            df_train,
            cfg["training"].get("gender_balance", True),
            cfg["training"].get("male_boost", 1.0),
        )
        shuffle = False

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        sampler=sampler, shuffle=shuffle, num_workers=4, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    model_cfg = cfg["model"]
    model = OcclusionModel(
        backbone=model_cfg["backbone"],
        pretrained=model_cfg["pretrained"],
        dropout=model_cfg["dropout"],
        head_dims=model_cfg.get("head_dims", [512, 128]),
    ).to(device)

    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"Reprise depuis {resume} (score={ckpt.get('score', '?'):.4f}, epoch={ckpt.get('epoch', '?')})")

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"],
                      weight_decay=cfg["training"]["weight_decay"])

    warmup_epochs = cfg["training"].get("warmup_epochs", 0)
    total_epochs = cfg["training"]["epochs"]
    if warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    os.makedirs(cfg["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["output"]["log_dir"], exist_ok=True)

    exp = cfg["output"]["experiment_name"]
    log_path = os.path.join(cfg["output"]["log_dir"], f"{exp}.csv")
    best_score, best_epoch = float("inf"), 0

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_err_f", "val_err_m", "val_score", "lr"])

        for epoch in range(1, cfg["training"]["epochs"] + 1):
            train_loss = train_epoch(model, train_loader, optimizer, device)
            err_f, err_m, score = val_epoch(model, val_loader, device)
            lr = optimizer.param_groups[0]["lr"]
            scheduler.step()

            print(f"Epoch {epoch:03d} | loss={train_loss:.4f} | "
                  f"ErrF={err_f:.4f} ErrM={err_m:.4f} Score={score:.4f} | lr={lr:.2e}")
            writer.writerow([epoch, train_loss, err_f, err_m, score, lr])
            f.flush()

            if score < best_score:
                best_score = score
                best_epoch = epoch
                ckpt_path = os.path.join(cfg["output"]["checkpoint_dir"], f"{exp}_best.pth")
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "score": score, "cfg": cfg}, ckpt_path)
                print(f"  → Nouveau meilleur checkpoint (Score={score:.4f})")

    print(f"\nMeilleur score : {best_score:.4f} à l'epoch {best_epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--debug", action="store_true", help="500 samples, 1 epoch")
    parser.add_argument("--exp-name", default=None, help="Override experiment_name from config")
    parser.add_argument("--resume", default=None, help="Chemin vers checkpoint pour fine-tuning")
    args = parser.parse_args()
    main(args.config, args.debug, args.exp_name, args.resume)
