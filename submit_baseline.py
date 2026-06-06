"""Training-free baseline submission for the Idemia Face Occlusion challenge.

Iterates over the test CSV, verifies each image crop exists on disk, assigns a
constant dummy occlusion value to every sample, and writes a submission CSV in
the exact format expected by the evaluation platform (filename, FaceOcclusion,
gender), where ``gender`` is the required dummy ``'x'`` column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DUMMY_OCCLUSION = 0.15
DEFAULT_TEST_CSV = Path("occlusion_datasets/test_students.csv")
DEFAULT_IMAGE_DIR = Path("crops/Crop_224_5fp_100K")
DEFAULT_OUTPUT_CSV = Path("test_predictions.csv")


def build_submission(
    test_csv: Path,
    image_dir: Path,
    output_csv: Path,
    dummy_value: float = DUMMY_OCCLUSION,
) -> pd.DataFrame:
    df_test = pd.read_csv(test_csv, delimiter=",").dropna(subset=["filename"])

    records: list[dict[str, object]] = []
    missing: list[str] = []

    for filename in tqdm(df_test["filename"].tolist(), desc="Predicting", unit="img"):
        image_path = image_dir / filename
        if not image_path.is_file():
            missing.append(filename)
            continue

        records.append(
            {
                "filename": filename,
                "FaceOcclusion": dummy_value,
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
    parser.add_argument("--value", type=float, default=DUMMY_OCCLUSION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_submission(
        test_csv=args.test_csv,
        image_dir=args.image_dir,
        output_csv=args.output_csv,
        dummy_value=args.value,
    )


if __name__ == "__main__":
    main()
