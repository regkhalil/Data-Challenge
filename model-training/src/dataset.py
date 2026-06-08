import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split


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
    def __init__(self, df: pd.DataFrame, image_root: str, transform=None, has_label: bool = True):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.image_root, row["filename"])
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.has_label:
            return img, float(row["FaceOcclusion"]), float(row["gender"])
        gender = float(row["gender"]) if "gender" in row.index else 0.0
        return img, gender


def make_weighted_sampler(df: pd.DataFrame, gender_balance: bool = True,
                          male_boost: float = 1.0) -> WeightedRandomSampler:
    """
    Poids = (1/30 + GT) × facteur_genre × male_boost (si homme)
    male_boost > 1.0 force le modèle à mieux apprendre sur les hommes.
    """
    occlusion = df["FaceOcclusion"].values
    sample_weights = 1 / 30 + occlusion

    if gender_balance:
        genders = df["gender"].values
        counts = {g: (genders == g).sum() for g in np.unique(genders)}
        total = len(genders)
        gender_w = np.array([total / (len(counts) * counts[g]) for g in genders])
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
