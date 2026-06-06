"""Training-free baseline submission for the Idemia Face Occlusion challenge.

Iterates over the test CSV, verifies each image crop exists on disk, estimates
the occluded fraction of the face using a simple HSV skin-color heuristic
restricted to a central circular region of interest, and writes a submission
CSV in the exact format expected by the evaluation platform
(``filename, FaceOcclusion, gender``), where ``gender`` is the required dummy
``'x'`` column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

FAILSAFE_OCCLUSION = 0.10
DEFAULT_TEST_CSV = Path("occlusion_datasets/test_students.csv")
DEFAULT_IMAGE_DIR = Path("crops/Crop_224_5fp_100K")
DEFAULT_OUTPUT_CSV = Path("test_predictions.csv")

# HSV skin-color thresholds (generic, training-free heuristic).
SKIN_HSV_LOWER = np.array([0, 20, 70], dtype=np.uint8)
SKIN_HSV_UPPER = np.array([20, 255, 255], dtype=np.uint8)

# Central circular ROI for the 224x224 crops (ignore background corners).
ROI_SHAPE: tuple[int, int] = (224, 224)
ROI_CENTER: tuple[int, int] = (112, 112)
ROI_RADIUS: int = 100


def _build_roi_mask(
    shape: tuple[int, int] = ROI_SHAPE,
    center: tuple[int, int] = ROI_CENTER,
    radius: int = ROI_RADIUS,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    return mask


_ROI_MASK: np.ndarray = _build_roi_mask()
_ROI_PIXEL_COUNT: int = int(np.count_nonzero(_ROI_MASK))


def estimate_occlusion(image_path: Path) -> float:
    """Estimate the occluded face fraction inside a central circular ROI.

    Returns a value in ``[0.0, 1.0]`` rounded to 4 decimals; falls back to
    :data:`FAILSAFE_OCCLUSION` when the image cannot be read.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return FAILSAFE_OCCLUSION

    if image.shape[:2] == ROI_SHAPE:
        roi_mask = _ROI_MASK
        roi_pixel_count = _ROI_PIXEL_COUNT
    else:
        h, w = image.shape[:2]
        roi_mask = _build_roi_mask(
            shape=(h, w),
            center=(w // 2, h // 2),
            radius=max(1, min(h, w) // 2 - 12),
        )
        roi_pixel_count = int(np.count_nonzero(roi_mask))

    if roi_pixel_count == 0:
        return FAILSAFE_OCCLUSION

    # Step 1: Normalize lighting locally using CLAHE in YCrCb space
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    image_normalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Convert normalized image to HSV
    hsv = cv2.cvtColor(image_normalized, cv2.COLOR_BGR2HSV)
    
    # Step 2: Check if the original image is monochrome/grayscale
    original_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mean_saturation = np.mean(original_hsv[:, :, 1])
    
    if mean_saturation < 12:
        # --- GRAYSCALE / MONOCHROME LOGIC ---
        v_channel = hsv[:, :, 2]
        skin_mask = cv2.inRange(v_channel, 45, 235)
    else:
        # --- NORMAL COLOR LOGIC ---
        lower1 = np.array([0, 8, 35], dtype=np.uint8)
        upper1 = np.array([28, 255, 255], dtype=np.uint8)
        
        lower2 = np.array([160, 8, 35], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)
        
        mask1 = cv2.inRange(hsv, lower1, upper1)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        skin_mask = cv2.bitwise_or(mask1, mask2)
        
    # Apply the circular Region of Interest (ROI) mask
    skin_in_roi = cv2.bitwise_and(skin_mask, skin_mask, mask=roi_mask)
    skin_pixels = int(np.count_nonzero(skin_in_roi))

    # --- SAFETY NET FOR HEURISTIC FAILURE ---
    # If the mask finds absolutely 0 skin pixels, the color filter completely lost the face.
    # Instead of confidently predicting 1.0 (100% occlusion) and getting penalized heavily,
    # we guess a conservative, safe value close to the dataset baseline.
    if skin_pixels == 0:
        return 0.2000

    # Calculate final occlusion fraction
    occlusion = 1.0 - (skin_pixels / roi_pixel_count)
    
    # Defensive Clamping: Even if a face looks highly occluded to our basic script,
    # we cap the maximum prediction at 0.60 to protect against heavy mathematical penalties.
    occlusion = float(np.clip(occlusion, 0.0, 0.60))
    
    return round(occlusion, 4)



def build_submission(
    test_csv: Path,
    image_dir: Path,
    output_csv: Path,
) -> pd.DataFrame:
    df_test = pd.read_csv(test_csv, delimiter=",").dropna(subset=["filename"])

    records: list[dict[str, object]] = []
    missing: list[str] = []

    for filename in tqdm(df_test["filename"].tolist(), desc="Predicting", unit="img"):
        image_path = image_dir / filename
        if not image_path.is_file():
            missing.append(filename)
            continue

        occlusion = estimate_occlusion(image_path)
        records.append(
            {
                "filename": filename,
                "FaceOcclusion": occlusion,
                "gender": "x",
            }
        )

    if missing:
        print(
            f"[warn] {len(missing)} image(s) listed in {test_csv} were not found "
            f"under {image_dir}. First few: {missing[:5]}",
            file=sys.stderr,
        )

    submission = pd.DataFrame(records, columns=["filename", "FaceOcclusion", "gender"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, sep=",", index=False)
    print(f"Wrote {len(submission)} predictions to {output_csv}")
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, default=DEFAULT_TEST_CSV)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_submission(
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
