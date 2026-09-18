#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from ALIGNMENT.alignment import AlignmentEstimator


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = APP_DIR / "output"

COEF_X = 0.001
COEF_Y = 0.001

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise FileNotFoundError(f"Input path not found: {input_path}")


def make_json_safe(result: dict) -> dict:
    output = {}

    for key, value in result.items():
        if key in {"debug_image", "mask"}:
            continue

        if hasattr(value, "item"):
            output[key] = value.item()
        elif isinstance(value, Path):
            output[key] = str(value)
        else:
            output[key] = value

    return output


def process_image(
    image_path: Path,
    output_dir: Path,
    estimator: AlignmentEstimator,
) -> bool:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"[ERROR] Cannot load image: {image_path}")
        return False

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        result = estimator.predict(rgb)
    except Exception as error:
        print(f"[ERROR] Inference failed for {image_path.name}: {error}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_path = output_dir / image_path.name
    json_path = output_dir / f"{image_path.stem}.json"

    if not cv2.imwrite(str(debug_path), result["debug_image"]):
        print(f"[ERROR] Failed to save debug image: {debug_path}")
        return False

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            make_json_safe(result),
            f,
            indent=4,
            ensure_ascii=False,
        )

    mask_area = int(result["mask"].sum()) if result["mask"] is not None else 0
    score = result["sam_score"]

    print(f"[INPUT]  {image_path}")
    print(f"[OUTPUT] Image: {debug_path}")
    print(f"[OUTPUT] JSON : {json_path}")
    print(f"[RESULT] Present: {result['present']}")
    print(
        f"         BBox={result['bbox']} | "
        f"SAM={score:.3f} | Area={mask_area}"
    )

    alignment = result.get("alignment")

    if alignment is not None:
        print(
            f"         Rotation={alignment['angle_deg']:.2f} deg | "
            f"DX={alignment['x_px']:.2f} px | "
            f"DY={alignment['y_px']:.2f} px"
        )
        print(
            f"         DX={alignment['x_m']:.5f} m | "
            f"DY={alignment['y_m']:.5f} m"
        )

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SAM2 Alignment inference"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or directory",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output directory",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    images = find_images(input_path)

    if not images:
        raise RuntimeError(f"No images found in: {input_path}")

    print("=" * 60)
    print("ALIGNMENT INFERENCE")
    print("=" * 60)
    print(f"[INPUT]  {input_path}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[INFO] Found {len(images)} image(s)")
    print(f"[INFO] coef_x = {COEF_X}")
    print(f"[INFO] coef_y = {COEF_Y}")
    print("=" * 60)

    print("[INFO] Initializing AlignmentEstimator...")

    estimator = AlignmentEstimator(
        coef_x=COEF_X,
        coef_y=COEF_Y,
    )

    print("[INFO] AlignmentEstimator ready.")
    print("-" * 60)

    success = 0

    for image_path in images:
        if process_image(image_path, output_dir, estimator):
            success += 1

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"[INFO] Processed: {success}/{len(images)}")
    print(f"[INFO] Output   : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
