#!/usr/bin/env python3
"""LIP Check Offline Inference Script."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from RACE_6D.core import PoseEstimator
from LIP_CHECK.flap_presence_helper import FlapPresenceHelper

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "assets" / "lip_check"

DEFAULT_K = np.array(
    [
        [1070.7897900136718, 0.0, 962.3093872070312],
        [0.0, 1070.5269785615235, 512.0372314453125],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080

STATUS_PRESENT = "PRESENT"
STATUS_ABSENT = "ABSENT"


# ----------------------------------------------------------------------
# Args / config
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="LIP Check offline inference.")

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Folder containing *_color.png and *_depth.png pairs.",
    )

    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: ./lip_check_test_out",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=("cuda", "cpu"),
        help="Inference device.",
    )

    return parser.parse_args()


def load_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {CONFIG_PATH}")

    return config


def resolve_path(path: str | Path) -> Path:
    path = Path(path).expanduser()

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path.resolve()


# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------
def write_json(path: Path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ----------------------------------------------------------------------
# Input / IO
# ----------------------------------------------------------------------
def collect_pairs(
    input_path: str | Path,
) -> list[tuple[Path, Path]]:
    input_path = Path(input_path).expanduser().resolve()

    if input_path.is_file():
        if input_path.name.endswith("_color.png"):
            color_path = input_path
            depth_path = input_path.with_name(
                input_path.name.replace(
                    "_color.png",
                    "_depth.png",
                )
            )

            if depth_path.is_file():
                return [(color_path, depth_path)]

        raise FileNotFoundError(f"Cannot find matching color/depth pair: {input_path}")

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    pairs = []

    for color_path in sorted(input_path.glob("*_color.png")):
        depth_path = color_path.with_name(
            color_path.name.replace(
                "_color.png",
                "_depth.png",
            )
        )

        if depth_path.is_file():
            pairs.append((color_path, depth_path))
        else:
            print(f"[WARNING] Missing depth image: {depth_path}")

    if not pairs:
        raise FileNotFoundError(
            f"No matching *_color.png / *_depth.png pairs found: {input_path}"
        )

    return pairs


def read_color(path: Path):
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(f"Cannot read color image: {path}")

    if image.shape[:2] != (
        INPUT_HEIGHT,
        INPUT_WIDTH,
    ):
        raise RuntimeError(
            f"Color image must be "
            f"{INPUT_WIDTH}x{INPUT_HEIGHT}, "
            f"got {image.shape[1]}x{image.shape[0]}: "
            f"{path.name}"
        )

    return image


def read_depth(
    path: Path,
    color_shape,
):
    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        raise RuntimeError(f"Cannot read depth image: {path}")

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype != np.uint16:
        print(
            f"[WARN] {path.name}: depth dtype is "
            f"{depth.dtype}, expected uint16 (mm)"
        )

    if depth.shape != tuple(color_shape[:2]):
        raise RuntimeError(
            f"RGB/depth size mismatch: "
            f"RGB={tuple(color_shape[:2])}, "
            f"Depth={depth.shape}"
        )

    return depth


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------
def make_estimator(
    config: dict,
    device: str,
):
    race_cfg = config.get("RACE_6D", {})

    if not isinstance(race_cfg, dict):
        raise ValueError("Invalid RACE_6D configuration.")

    model_config = resolve_path(race_cfg["MODEL_CONFIG"])

    model_path = resolve_path(race_cfg["MODEL_PATH"])

    if not model_config.is_file():
        raise FileNotFoundError(f"RACE-6D model config not found: {model_config}")

    if not model_path.is_file():
        raise FileNotFoundError(f"RACE-6D model weights not found: {model_path}")

    label_names = race_cfg.get("LABEL_NAMES")

    if not isinstance(label_names, (dict, list)):
        raise ValueError("RACE_6D.LABEL_NAMES must be a dict or list.")

    pose_estimator = PoseEstimator(
        model_path=str(model_path),
        config_path=str(model_config),
        device=device,
        score_threshold=float(race_cfg.get("SCORE_THRESHOLD", 0.25)),
        max_per_class=int(race_cfg.get("MAX_PER_CLASS", 1)),
        max_detections=race_cfg.get("MAX_DETECTIONS"),
        depth_z_max_mm=race_cfg.get("DEPTH_Z_MAX_MM"),
        invalid_depth_value=int(race_cfg.get("INVALID_DEPTH_VALUE", 65535)),
        class_id=None,
        refine_config=config.get("RGB_REFINE"),
    )

    helper = FlapPresenceHelper(
        pose_estimator=pose_estimator,
    )

    return helper, pose_estimator, device


# ----------------------------------------------------------------------
# Debug image
# ----------------------------------------------------------------------
def save_debug_image(
    helper: FlapPresenceHelper,
    debug_path: Path,
):
    debug_image = helper.debug_image

    if debug_image is None:
        return None

    if not isinstance(debug_image, np.ndarray):
        raise RuntimeError(f"Invalid debug image type: {type(debug_image)}")

    if debug_image.size == 0:
        raise RuntimeError("Debug image is empty.")

    debug_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    success = cv2.imwrite(
        str(debug_path),
        debug_image,
    )

    if not success:
        raise RuntimeError(f"Failed to save debug image: {debug_path}")

    return str(debug_path)


# ----------------------------------------------------------------------
# Result processing
# ----------------------------------------------------------------------
def process_image(
    stem,
    color_path,
    depth_path,
    helper,
    camera_matrix,
    out_dir,
):
    debug_path = out_dir / "debug" / f"{stem}_debug.png"

    image_record = {
        "schema_version": 1,
        "image": stem,
        "mode": "lip_check",
        "inputs": {
            "color": str(color_path),
            "depth": str(depth_path),
        },
        "success": False,
        "status": None,
        "predict_ms": None,
        "debug_image": None,
        "error": None,
    }

    try:
        rgb = read_color(color_path)

        depth = read_depth(
            depth_path,
            rgb.shape,
        )

        helper.debug_image = None

        t0 = time.perf_counter()

        try:
            status = helper.predict(
                rgb=rgb,
                depth=depth,
                intrinsic=camera_matrix,
            )
        finally:
            image_record["debug_image"] = save_debug_image(
                helper=helper,
                debug_path=debug_path,
            )

        elapsed = time.perf_counter() - t0

        if status not in (
            STATUS_PRESENT,
            STATUS_ABSENT,
        ):
            raise RuntimeError(f"Unexpected prediction result: {status}")

        image_record["success"] = True
        image_record["status"] = status
        image_record["predict_ms"] = elapsed * 1000.0

        print(f"[RESULT] {status} " f"({elapsed * 1000.0:.1f} ms)")

        if image_record["debug_image"] is not None:
            print(f"[DEBUG] Saved: " f"{image_record['debug_image']}")
        else:
            print("[WARNING] No debug image generated.")

    except Exception as e:
        image_record["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }

        print(f"[ERROR] {stem}: " f"{type(e).__name__}: {e}")

        if image_record["debug_image"] is not None:
            print(f"[DEBUG] Saved: " f"{image_record['debug_image']}")

    write_json(
        out_dir / f"{stem}_lip_check.json",
        image_record,
    )

    return image_record


# ----------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------
def write_summary(
    pairs,
    records,
    device,
    out_dir,
):
    successful = [r for r in records if r["success"]]

    failed = [r for r in records if not r["success"]]

    present_count = sum(r["status"] == STATUS_PRESENT for r in successful)

    absent_count = sum(r["status"] == STATUS_ABSENT for r in successful)

    timings = np.asarray(
        [r["predict_ms"] for r in successful if r["predict_ms"] is not None],
        dtype=np.float64,
    )

    summary = {
        "schema_version": 1,
        "mode": "lip_check",
        "input": str(Path(pairs[0][0]).parent) if pairs else None,
        "output": str(out_dir),
        "device": device,
        "images": {
            "total": len(pairs),
            "processed": len(successful),
            "failed": len(failed),
        },
        "results": {
            "present": present_count,
            "absent": absent_count,
        },
        "metrics": {
            "predict_mean_ms": (float(np.mean(timings)) if timings.size else None),
            "predict_median_ms": (float(np.median(timings)) if timings.size else None),
        },
        "failed_images": [
            {
                "image": r["image"],
                "error": r["error"],
                "debug_image": r["debug_image"],
            }
            for r in failed
        ],
        "image_records": records,
    }

    write_json(
        out_dir / "lip_check_summary.json",
        summary,
    )

    return summary


def print_summary(summary):
    images = summary["images"]
    results = summary["results"]
    metrics = summary["metrics"]

    print("\n" + "=" * 60)
    print("LIP CHECK SUMMARY")
    print("=" * 60)

    print(f"Images  : {images['processed']}/{images['total']}")

    print(f"Failed  : {images['failed']}")

    print(f"PRESENT : {results['present']}")

    print(f"ABSENT  : {results['absent']}")

    if metrics["predict_mean_ms"] is not None:
        print(
            f"Predict : mean={metrics['predict_mean_ms']:.1f} ms "
            f"median={metrics['predict_median_ms']:.1f} ms"
        )

    print("=" * 60)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    args = parse_args()

    pairs = collect_pairs(args.input)

    if not pairs:
        raise SystemExit(f"No color/depth pair found: {args.input}")

    config = load_config()

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path.cwd() / "lip_check_test_out"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera_matrix = DEFAULT_K.copy()

    print("=" * 60)
    print("LIP CHECK OFFLINE TEST")
    print("=" * 60)

    print(f"Input  : {Path(args.input).resolve()}")

    print(f"Pairs  : {len(pairs)}")

    print(f"Output : {out_dir}")

    print(f"Debug  : {out_dir / 'debug'}")

    print(f"Device : {args.device}")

    print("[INFO] Initializing RACE-6D PoseEstimator...")

    helper, pose_estimator, device = make_estimator(
        config=config,
        device=args.device,
    )

    print("[INFO] PoseEstimator ready.")

    print("[INFO] Initializing LIP Check helper...")

    print("[INFO] LIP Check ready. " f"Device: {device}")

    records = []

    for color_path, depth_path in pairs:
        stem = color_path.name.removesuffix("_color.png")

        print(f"\n--- {stem} ---")

        record = process_image(
            stem=stem,
            color_path=color_path,
            depth_path=depth_path,
            helper=helper,
            camera_matrix=camera_matrix,
            out_dir=out_dir,
        )

        records.append(record)

    summary = write_summary(
        pairs=pairs,
        records=records,
        device=device,
        out_dir=out_dir,
    )

    print_summary(summary)

    print(f"\nOutput       : {out_dir}")

    print(f"Debug Images : {out_dir / 'debug'}")

    print(f"Summary JSON : {out_dir / 'lip_check_summary.json'}")


if __name__ == "__main__":
    main()
