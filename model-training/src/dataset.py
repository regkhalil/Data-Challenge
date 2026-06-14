import os
import random
import pandas as pd
import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


# ── Synthetic degradation augmentation ────────────────────────────────────────
# Simulates the visual patterns found in high-GT samples:
#   scan lines (GT≈1.0), overexposure (GT≈0.7), posterize/sketch (GT≈0.6-0.7)
# Applied at image level (PIL) before the normal torchvision transforms.
# The label floor is used to RAISE the GT, never to lower it.

_DEGRAD_FLOOR = {
    "scanlines": 0.85,   # interlacing → almost total visual occlusion (vrai GT≈1.0)
    "overexpose": 0.65,  # surexposition extrême → détails perdus (vrai GT≈0.78)
    "posterize": 0.60,   # posterisation/style poster → peu de valeurs (vrai GT≈0.68)
    "downscale": 0.55,   # pixelisation lourde → image très floue (vrai GT≈0.62)
    "erase_rect": 0.55,  # rectangle noir/gris simulant main ou objet → vrai GT≈0.57
}


def _apply_degradation(img: Image.Image, kind: str) -> Image.Image:
    """Apply one synthetic degradation to a PIL Image."""
    w, h = img.size
    if kind == "scanlines":
        img = img.copy()
        arr = np.array(img)
        # darken every other row (interlacing effect)
        arr[::2] = (arr[::2] * 0.35).astype(np.uint8)
        return Image.fromarray(arr)
    elif kind == "overexpose":
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(2.5, 4.0))
        return img
    elif kind == "posterize":
        img = img.convert("L").convert("RGB")  # desaturate
        img = ImageOps.posterize(img, bits=random.randint(2, 3))
        return img
    elif kind == "downscale":
        # downscale fort puis upscale → pixelisation/flou de basse résolution
        factor = random.choice([6, 8, 10])
        small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
        return small.resize((w, h), Image.NEAREST)
    elif kind == "erase_rect":
        # rectangle couvrant 20-50% du visage simulant main/objet physique
        img = img.copy()
        arr = np.array(img)
        # taille du rectangle : 20-50% de la dimension
        rh = random.randint(h // 5, h // 2)
        rw = random.randint(w // 5, w // 2)
        # position centrée sur le bas du visage (bouche/menton) avec bruit
        cy = random.randint(h // 2, int(h * 0.75))
        cx = random.randint(w // 4, int(w * 0.75))
        y1, y2 = max(0, cy - rh // 2), min(h, cy + rh // 2)
        x1, x2 = max(0, cx - rw // 2), min(w, cx + rw // 2)
        # couleur : gris peau foncé ou noir (simule main sombre ou objet)
        fill = random.randint(30, 120)
        arr[y1:y2, x1:x2] = fill
        return Image.fromarray(arr)
    return img


def apply_synthetic_degradation(img: Image.Image, gt: float,
                                  p: float = 0.10) -> tuple:
    """
    With probability p, apply a random degradation and raise GT to its floor.
    Returns (augmented_img, new_gt).
    """
    if random.random() >= p:
        return img, gt
    kind = random.choice(list(_DEGRAD_FLOOR.keys()))
    img = _apply_degradation(img, kind)
    new_gt = max(gt, _DEGRAD_FLOOR[kind])
    return img, new_gt


def make_transforms(train: bool, cfg: dict) -> transforms.Compose:
    aug = cfg.get("augmentation", {})
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if train:
        jitter = aug.get("color_jitter", {})
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=aug.get("horizontal_flip", 0.5)),
            transforms.ColorJitter(
                brightness=jitter.get("brightness", 0.2),
                contrast=jitter.get("contrast", 0.2),
                saturation=jitter.get("saturation", 0.1),
            ),
            transforms.RandomRotation(degrees=aug.get("rotation_degrees", 15)),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3)],
                p=aug.get("gaussian_blur_p", 0.3),
            ),
            transforms.ToTensor(),
            norm,
        ])
    return transforms.Compose([transforms.ToTensor(), norm])


class OcclusionDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_root: str, transform=None,
                 has_label: bool = True, synth_degrad_p: float = 0.0):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.has_label = has_label
        self.synth_degrad_p = synth_degrad_p  # 0 = disabled (val/test)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_root, row["filename"])
        img = Image.open(path).convert("RGB")
        if self.has_label and self.synth_degrad_p > 0:
            gt = float(row["FaceOcclusion"])
            img, gt = apply_synthetic_degradation(img, gt, p=self.synth_degrad_p)
        else:
            gt = float(row["FaceOcclusion"]) if self.has_label else None
        if self.transform:
            img = self.transform(img)
        if self.has_label:
            return img, gt, float(row["gender"])
        gender = float(row["gender"]) if "gender" in row.index else 0.0
        return img, gender


def make_weighted_sampler(df: pd.DataFrame, gender_balance: bool = True,
                          male_boost: float = 1.0,
                          tail_k: float = 0.0, tail_p: float = 2.0,
                          male_boost_gt_scaled: bool = False) -> WeightedRandomSampler:
    """
    Poids = (1/30 + GT) × (1 + tail_k·GT^tail_p) × facteur_genre × boost_homme
    tail_k > 0 sur-pondère les fortes occlusions pour casser le plafonnement
    des prédictions (le modèle apprend à prédire au-dessus de ~0.5).

    boost_homme :
      - male_boost_gt_scaled=False : male_boost uniforme sur TOUS les hommes
        (gaspille le boost sur les ~13k hommes faiblement occlus, déjà bien prédits)
      - male_boost_gt_scaled=True  : boost = 1 + (male_boost-1)·GT
        → concentré sur les hommes à FORTE occlusion. Corrige la vraie défaillance
          mesurée : ErrM ≫ ErrF car le modèle a appris "homme = faible occlusion"
          et s'effondre sur les rares hommes réellement occlus.
    """
    occlusion = df["FaceOcclusion"].values
    sample_weights = (1 / 30 + occlusion) * (1.0 + tail_k * occlusion ** tail_p)

    if gender_balance:
        genders = df["gender"].values
        counts = {g: (genders == g).sum() for g in np.unique(genders)}
        total = len(genders)
        gender_w = np.array([total / (len(counts) * counts[g]) for g in genders])
        if male_boost_gt_scaled:
            male_factor = 1.0 + (male_boost - 1.0) * occlusion
            boost = np.where(genders == 1.0, male_factor, 1.0)
        else:
            boost = np.where(genders == 1.0, male_boost, 1.0)
        sample_weights = sample_weights * gender_w * boost

    return WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def load_splits(cfg: dict):
    data_cfg = cfg["data"]
    df = pd.read_csv(data_cfg["train_csv"])
    df_test = pd.read_csv(data_cfg["test_csv"])

    # Stratification sur genre × bucket d'occlusion (5 buckets pour éviter les classes vides)
    df["occ_bucket"] = (df["FaceOcclusion"] * 5).clip(0, 4).astype(int)
    df["strat_key"] = df["gender"].astype(str) + "_" + df["occ_bucket"].astype(str)

    # Fusionner les classes trop rares (< 2 membres) avec le bucket précédent
    counts = df["strat_key"].value_counts()
    rare = set(counts[counts < 2].index)
    if rare:
        df["occ_bucket"] = df["occ_bucket"].clip(0, 3)
        df["strat_key"] = df["gender"].astype(str) + "_" + df["occ_bucket"].astype(str)

    df_train, df_val = train_test_split(
        df,
        test_size=data_cfg["val_split"],
        random_state=data_cfg["seed"],
        stratify=df["strat_key"],
    )
    df_train = df_train.drop(columns=["occ_bucket", "strat_key"])
    df_val = df_val.drop(columns=["occ_bucket", "strat_key"])

    return df_train, df_val, df_test
