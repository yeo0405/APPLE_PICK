#!/usr/bin/env python3
"""Box Check Inference Script."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from FLAP_CHECK.ear_esitimator import DINOv3Estimator as EarEstimator

DEFAULT_OUTPUT_DIR = APP_DIR / "output"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 这些键不写入 JSON，也不打印（图像 / mask / 相似度图）
EXCLUDE_KEYS = {
    "debug_image",
    "mask",
    "roi_mask",
    "pass_mask",
    "similarity_map",
}

_SKIP = object()


def _sanitize(obj: Any) -> Any:
    """递归清理：去掉 mask / 数组，numpy 标量转原生类型，使结果可被 json 序列化。"""
    if isinstance(obj, dict):
        cleaned = {}
        for key, value in obj.items():
            if key in EXCLUDE_KEYS:
                continue
            value = _sanitize(value)
            if value is _SKIP:
                continue
            cleaned[str(key)] = value
        return cleaned

    if isinstance(obj, (list, tuple)):
        items = [_sanitize(v) for v in obj]
        return [v for v in items if v is not _SKIP]

    # numpy 数组（含 bool mask）一律丢弃
    if isinstance(obj, np.ndarray):
        return _SKIP

    # torch tensor 同样丢弃（不强制 import torch）
    if type(obj).__module__.startswith("torch"):
        return _SKIP

    if isinstance(obj, np.generic):
        return obj.item()

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    # 其他无法序列化的类型转成字符串，避免写 JSON 时报错
    return str(obj)


def sanitize_result(result: dict) -> dict:
    cleaned = _sanitize(result)
    return cleaned if cleaned is not _SKIP else {}


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


def save_json(clean_result: dict, output_path: Path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            clean_result,
            f,
            indent=2,
            ensure_ascii=False,
        )


def process_image(
    estimator: EarEstimator,
    image_path: Path,
    output_dir: Path,
) -> bool:
    print("-" * 60)
    print(f"[INPUT] {image_path}")

    try:
        rgb = load_image(image_path)
        result = estimator.predict(rgb)
    except Exception as e:
        print(f"[ERROR] Prediction failed: {image_path}")
        print(f"        {e}")
        return False

    debug_image = result.get("debug_image")

    if debug_image is None:
        print(f"[WARNING] No debug_image returned: {image_path}")
        return False

    image_output_path = output_dir / image_path.name

    if not cv2.imwrite(
        str(image_output_path),
        debug_image,
    ):
        print(f"[ERROR] Failed to save: {image_output_path}")
        return False

    # 清理后的结果：不含 debug_image / mask / 数组
    clean_result = sanitize_result(result)

    json_output_path = output_dir / f"{image_path.stem}.json"

    try:
        save_json(clean_result, json_output_path)
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {json_output_path}")
        print(f"        {e}")
        return False

    print(f"[OUTPUT] Image: {image_output_path}")
    print(f"[OUTPUT] JSON : {json_output_path}")

    for key, value in clean_result.items():
        print(f"  {key}: {value}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description=("Box Check inference. " "Input can be a single image or a folder.")
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
    print("BOX CHECK INFERENCE")
    print("=" * 60)
    print(f"[INPUT]  {input_path}")
    print(f"[OUTPUT] {output_dir}")

    images = collect_images(input_path)
    print(f"[INFO] Found {len(images)} image(s)")

    print("[INFO] Initializing EarEstimator...")

    estimator = EarEstimator()

    print("[INFO] EarEstimator ready.")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    success_count = 0
    failed_count = 0

    for image_path in images:
        success = process_image(
            estimator=estimator,
            image_path=image_path,
            output_dir=output_dir,
        )

        if success:
            success_count += 1
        else:
            failed_count += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total   : {len(images)}")
    print(f"Success : {success_count}")
    print(f"Failed  : {failed_count}")
    print(f"Output  : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
