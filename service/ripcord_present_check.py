#!/usr/bin/env python3
"""Ripcord Present Check Inference Script."""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from RIPCORD_CHECK.ripcord_esitimator import DINOv3Estimator as RipcordCheckEstimator

DEFAULT_OUTPUT_DIR = APP_DIR / "output_ripcord_check"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

SIM_THRESHOLD = 0.75
COVERAGE_THRESHOLD = 0.5


def load_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")
    return image


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise RuntimeError(f"Unsupported image format: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        images = sorted(
            p
            for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise RuntimeError(f"No supported images found in: {input_path}")

        return images

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _json_default(obj):
    # pass_mask / similarity_map 是 numpy 数组,需要转换
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Not JSON serializable: {type(obj).__name__}")


def save_json(result: dict, output_path: Path):
    json_result = {key: value for key, value in result.items() if key != "debug_image"}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            json_result,
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )


def process_image(
    estimator: RipcordCheckEstimator,
    image_path: Path,
    output_dir: Path,
):
    """Return (success, absent). absent is None on failure."""
    print("-" * 60)
    print(f"[INPUT] {image_path}")

    try:
        rgb = load_image(image_path)
        result = estimator.predict(rgb)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {image_path}")
        print(f"        {e}")
        return False, None

    debug_image = result.get("debug_image")

    if debug_image is None:
        print(f"[WARNING] No debug_image returned: {image_path}")
        return False, None

    image_output_path = output_dir / image_path.name

    if not cv2.imwrite(
        str(image_output_path),
        debug_image,
    ):
        print(f"[ERROR] Failed to save: {image_output_path}")
        return False, None

    json_output_path = output_dir / f"{image_path.stem}.json"

    try:
        save_json(result, json_output_path)
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {json_output_path}")
        print(f"        {e}")
        return False, None

    print(f"[OUTPUT] Image: {image_output_path}")
    print(f"[OUTPUT] JSON : {json_output_path}")

    absent = bool(result.get("ripcord"))
    item = result.get("ripcord_evaluation", {})

    print(f"  ripcord (absent): {absent}")
    print(
        f"  label={item.get('label')} | "
        f"coverage={item.get('coverage_percent', 0.0):.1f}% | "
        f"mean={item.get('mean_similarity', 0.0):.4f} | "
        f"p10={item.get('p10_similarity', 0.0):.4f} | "
        f"patch={item.get('pass_patch_count', 0)}/{item.get('patch_count', 0)}"
    )

    return True, absent


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Ripcord Present Check inference. "
            "Input can be a single image or a folder."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input image or folder containing images.",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "Output directory for debug images and JSON files. "
            f"Default: {DEFAULT_OUTPUT_DIR}"
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    print("=" * 60)
    print("RIPCORD PRESENT CHECK INFERENCE")
    print("=" * 60)
    print(f"[INPUT]  {input_path}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[CONFIG] SIM_THRESHOLD      = {SIM_THRESHOLD}")
    print(f"[CONFIG] COVERAGE_THRESHOLD = {COVERAGE_THRESHOLD}")

    images = collect_images(input_path)
    print(f"[INFO] Found {len(images)} image(s)")

    print("[INFO] Initializing RipcordCheckEstimator...")

    estimator = RipcordCheckEstimator(
        sim_threshold=SIM_THRESHOLD,
        coverage_threshold=COVERAGE_THRESHOLD,
    )

    print("[INFO] RipcordCheckEstimator ready.")

    output_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    failed_count = 0
    present_count = 0
    absent_count = 0

    for image_path in images:
        success, absent = process_image(
            estimator=estimator,
            image_path=image_path,
            output_dir=output_dir,
        )

        if success:
            success_count += 1
            if absent:
                absent_count += 1
            else:
                present_count += 1
        else:
            failed_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total   : {len(images)}")
    print(f"Success : {success_count}")
    print(f"Failed  : {failed_count}")
    print(f"PRESENT : {present_count}")
    print(f"ABSENT  : {absent_count}")
    print(f"Output  : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
