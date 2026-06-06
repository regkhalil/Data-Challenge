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

FAILSAFE_OCCLUSION = 0.05
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


def _build_anatomy_mask(h: int, w: int) -> np.ndarray:
    """Build a fixed-geometry exclusion mask for eyes and mouth.

    Based on standard 5-point face alignment (MTCNN/RetinaFace convention) for
    224x224 crops.  Pixels inside these ellipses are excluded from the Laplacian
    texture filter so that natural eyelash / lip edges are never misclassified as
    occlusions.
    """
    sx, sy = w / 224.0, h / 224.0
    mask = np.zeros((h, w), dtype=np.uint8)
    # Left eye (center ~75, 80 in 224-space)
    cv2.ellipse(mask, (int(75 * sx), int(80 * sy)),
                (int(24 * sx), int(15 * sy)), 0, 0, 360, 255, -1)
    # Right eye (center ~149, 80)
    cv2.ellipse(mask, (int(149 * sx), int(80 * sy)),
                (int(24 * sx), int(15 * sy)), 0, 0, 360, 255, -1)
    # Mouth (center ~112, 152)
    cv2.ellipse(mask, (int(112 * sx), int(152 * sy)),
                (int(32 * sx), int(18 * sy)), 0, 0, 360, 255, -1)
    return mask


# Pre-built anatomy mask for the standard 224x224 case.
_ANATOMY_MASK: np.ndarray = _build_anatomy_mask(*ROI_SHAPE)

# Candidate sampling patches for adaptive skin color (in 224x224 coordinates).
# Each entry is (row_start, row_end, col_start, col_end).
# Left cheek, right cheek, forehead, nose/centre — all away from eye/mouth anatomy.
_SAMPLE_PATCHES: list[tuple[int, int, int, int]] = [
    ( 72, 108,  48,  88),  # left cheek
    ( 72, 108, 136, 176),  # right cheek
    ( 32,  62,  88, 136),  # forehead
    (100, 130,  96, 128),  # nose / centre  ← always on-face for aligned crops
]

# Morphological kernels (built once).
_MORPH_CLOSE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_MORPH_OPEN_K  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def estimate_occlusion(image_path: Path) -> float:
    """Estimate the occluded face fraction inside a central circular ROI.

    Two-channel skin detection strategy:
    - **YCrCb primary** (Cr∈[133,173], Cb∈[77,127]): illumination-invariant
      chrominance ranges covering the full skin-tone gamut with no per-image
      calibration — robust to dark, blurry, and low-contrast faces.
    - **Adaptive HSV secondary**: reference patch is chosen by the *fraction
      of YCrCb-confirmed skin pixels* inside each candidate window, so
      colourful backgrounds (red stage lights, blue backdrops) that have high
      saturation but no skin chrominance are never mistaken for skin.
    - **Union** of both detectors gives permissive coverage.
    - Laplacian texture filter uses a 5×5 pre-blur (vs 3×3) and a raised
      threshold of 60 (vs 45) to avoid false edges in blurry/noisy images.
    - Safety cap lowered to 0.50; blind-pipeline failsafe returns
      :data:`FAILSAFE_OCCLUSION` (0.05).

    Returns a value in ``[0.0, 0.50]`` rounded to 4 decimals.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return FAILSAFE_OCCLUSION

    h, w = image.shape[:2]

    # ------------------------------------------------------------------
    # 1. ROI and anatomy masks
    # ------------------------------------------------------------------
    if (h, w) == ROI_SHAPE:
        roi_mask        = _ROI_MASK
        roi_pixel_count = _ROI_PIXEL_COUNT
        anatomy_mask    = _ANATOMY_MASK
    else:
        roi_mask = _build_roi_mask(
            shape=(h, w),
            center=(w // 2, h // 2),
            radius=max(1, min(h, w) // 2 - 12),
        )
        roi_pixel_count = int(np.count_nonzero(roi_mask))
        anatomy_mask = _build_anatomy_mask(h, w)

    if roi_pixel_count == 0:
        return FAILSAFE_OCCLUSION

    # ------------------------------------------------------------------
    # 2. Lighting normalisation — CLAHE on the Y channel (YCrCb)
    # ------------------------------------------------------------------
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    image_norm = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    ycrcb_norm = cv2.cvtColor(image_norm, cv2.COLOR_BGR2YCrCb)
    hsv_norm   = cv2.cvtColor(image_norm, cv2.COLOR_BGR2HSV)

    # ------------------------------------------------------------------
    # 3a. Primary skin mask — YCrCb chrominance
    #     Cr ∈ [133, 173], Cb ∈ [77, 127] covers the skin-tone gamut and
    #     is largely insensitive to illumination after CLAHE.
    # ------------------------------------------------------------------
    ycrcb_skin = cv2.inRange(
        ycrcb_norm,
        np.array([0,   133,  77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8),
    )

    # ------------------------------------------------------------------
    # 3b. Secondary skin mask — adaptive HSV
    #     Select the patch whose pixels are most confirmed by the YCrCb
    #     detector; this makes patch selection background-resistant.
    # ------------------------------------------------------------------
    mean_saturation = float(np.mean(hsv_norm[:, :, 1]))

    if mean_saturation < 12:
        # Monochrome / near-IR image — luminance-only path
        hsv_skin = cv2.inRange(ycrcb_norm[:, :, 0],
                               45, 235)
    else:
        sx, sy = w / 224.0, h / 224.0
        best_patch_hsv: np.ndarray | None = None
        best_skin_frac = -1.0

        for r0, r1, c0, c1 in _SAMPLE_PATCHES:
            pr0, pr1 = int(r0 * sy), int(r1 * sy)
            pc0, pc1 = int(c0 * sx), int(c1 * sx)
            patch_ycrcb = ycrcb_norm[pr0:pr1, pc0:pc1]
            patch_hsv   = hsv_norm[pr0:pr1, pc0:pc1]
            if patch_ycrcb.size == 0:
                continue
            # Fraction of pixels inside this patch confirmed as skin by YCrCb
            skin_frac = float(np.mean(
                cv2.inRange(patch_ycrcb,
                            np.array([0,   133,  77], dtype=np.uint8),
                            np.array([255, 173, 127], dtype=np.uint8)) > 0
            ))
            if skin_frac > best_skin_frac:
                best_skin_frac = skin_frac
                best_patch_hsv = patch_hsv

        if best_patch_hsv is not None and best_skin_frac > 0.25:
            med_h = float(np.median(best_patch_hsv[:, :, 0]))
            med_s = float(np.median(best_patch_hsv[:, :, 1]))
            med_v = float(np.median(best_patch_hsv[:, :, 2]))

            h_tol, s_tol, v_tol = 22, 80, 90
            h_lo = max(0,   med_h - h_tol)
            h_hi = min(180, med_h + h_tol)
            s_lo = max(5,   med_s - s_tol)
            v_lo = max(20,  med_v - v_tol)

            hsv_skin = cv2.inRange(
                hsv_norm,
                np.array([h_lo, s_lo, v_lo], dtype=np.uint8),
                np.array([h_hi, 255,  255],   dtype=np.uint8),
            )
            # Wrap-around for reddish / dark skin tones near H = 0
            if h_lo < 5:
                wrap_lo = max(0, 175 - int(h_tol))
                hsv_skin = cv2.bitwise_or(
                    hsv_skin,
                    cv2.inRange(
                        hsv_norm,
                        np.array([wrap_lo, s_lo, v_lo], dtype=np.uint8),
                        np.array([180,     255,  255],   dtype=np.uint8),
                    ),
                )
        else:
            # Wide fallback — loosened floor values for dark / pale faces
            hsv_skin = cv2.bitwise_or(
                cv2.inRange(hsv_norm,
                            np.array([0,   5, 20], dtype=np.uint8),
                            np.array([30, 255, 255], dtype=np.uint8)),
                cv2.inRange(hsv_norm,
                            np.array([155, 5, 20], dtype=np.uint8),
                            np.array([180, 255, 255], dtype=np.uint8)),
            )

    # Union: a pixel counts as skin when either detector fires
    skin_mask = cv2.bitwise_or(ycrcb_skin, hsv_skin)

    # ------------------------------------------------------------------
    # 4. Morphological closing — fills small intra-skin holes
    # ------------------------------------------------------------------
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, _MORPH_CLOSE_K)

    # ------------------------------------------------------------------
    # 5. Texture filter — Laplacian
    #    5×5 pre-blur (vs 3×3) and threshold 60 (vs 45) avoid stripping
    #    skin pixels from blurry or noisy images while still catching
    #    dense occlusion textures (mask fabric, hands).
    # ------------------------------------------------------------------
    gray    = cv2.cvtColor(image_norm, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    abs_lap = cv2.convertScaleAbs(cv2.Laplacian(blurred, cv2.CV_64F))
    _, edge_mask = cv2.threshold(abs_lap, 60, 255, cv2.THRESH_BINARY)
    edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_OPEN, _MORPH_OPEN_K)

    # Suppress anatomy zones so natural eyelash / lip edges are not penalised
    edge_mask_exterior = cv2.bitwise_and(edge_mask,
                                         cv2.bitwise_not(anatomy_mask))
    skin_mask = cv2.bitwise_and(skin_mask,
                                cv2.bitwise_not(edge_mask_exterior))

    # ------------------------------------------------------------------
    # 6. Aggregate within ROI
    # ------------------------------------------------------------------
    skin_pixels = int(np.count_nonzero(
        cv2.bitwise_and(skin_mask, skin_mask, mask=roi_mask)
    ))

    # --- Low-coverage rescue ---
    # If the Laplacian filter reduced skin coverage to < 20% of the ROI but
    # the YCrCb detector alone (unaffected by the Laplacian) sees ≥ 30%,
    # the texture filter was over-aggressive (blurry/noisy/dark image).
    # Substitute the YCrCb-only count so the pipeline doesn't falsely
    # report high occlusion on a clear face.
    ycrcb_in_roi = int(np.count_nonzero(
        cv2.bitwise_and(ycrcb_skin, ycrcb_skin, mask=roi_mask)
    ))
    if (skin_pixels < int(0.20 * roi_pixel_count)
            and ycrcb_in_roi >= int(0.30 * roi_pixel_count)):
        skin_pixels = ycrcb_in_roi

    if skin_pixels == 0:
        return FAILSAFE_OCCLUSION

    occlusion = 1.0 - (skin_pixels / roi_pixel_count)
    occlusion = float(np.clip(occlusion, 0.0, 0.45))
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
