#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from RACE_6D.core import PoseEstimator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
DEFAULT_INPUT = PROJECT_ROOT / "assets" / "race_6d"

INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080
CROP_X = 240
CROP_WIDTH = 1440
AXIS_LENGTH_MM = 50.0

COLOR_SUFFIX = "_color.png"
DEPTH_SUFFIX = "_depth.png"

FONT = cv2.FONT_HERSHEY_SIMPLEX
NAN = float("nan")

DEFAULT_K = np.array(
    [
        [1070.7897900136718, 0.0, 962.3093872070312],
        [0.0, 1070.5269785615235, 512.0372314453125],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

CSV_FIELDS = [
    "image",
    "label",
    "name",
    "confidence",
    "refined",
    "method",
    "fallback",
    "reason",
    "edge_reason",
    "tx_raw",
    "ty_raw",
    "tz_raw",
    "tx",
    "ty",
    "tz",
    "dt_mm",
    "dang_deg",
    "edge_shift_0",
    "edge_shift_1",
    "edge_shift_2",
    "edge_shift_3",
    "corner_shift_px",
    "area_ratio",
    "n_ref_kp",
    "n_patch_kp",
    "n_matches",
    "n_gated",
    "n_h_inliers",
    "n_inliers",
    "shift_dx_mm",
    "shift_dy_mm",
    "h_scale",
    "h_rot_deg",
    "coverage",
    "reproj_px",
    "depth_med_mm",
    "z_err_raw_mm",
    "z_err_ref_mm",
]


# ----------------------------------------------------------------------
# Args / config
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="RACE-6D EDGE-only offline refinement test."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Folder containing *_color.png / *_depth.png pairs, "
        "or a single *_color.png file.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Output image scale.",
    )
    parser.add_argument(
        "--out-dir",
        default="",
        help="Output directory. Default: ./race6d_edge_test_out",
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


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_race6d_config():
    config = load_config()
    race_config = config.get("RACE_6D")

    if not isinstance(race_config, dict):
        raise ValueError(f"RACE_6D section is missing in {CONFIG_PATH}")

    label_names = race_config.get("LABEL_NAMES", {})

    if not isinstance(label_names, dict):
        raise ValueError("RACE_6D.LABEL_NAMES must be a mapping.")

    return {
        "model_path": resolve_project_path(race_config["MODEL_PATH"]),
        "model_config": resolve_project_path(race_config["MODEL_CONFIG"]),
        "score_threshold": float(race_config.get("SCORE_THRESHOLD", 0.25)),
        "max_per_class": int(race_config.get("MAX_PER_CLASS", 1)),
        "max_detections": race_config.get("MAX_DETECTIONS"),
        "class_id": race_config.get("CLASS_ID"),
        "depth_z_max_mm": float(race_config.get("DEPTH_Z_MAX_MM", 2000.0)),
        "invalid_depth_value": int(race_config.get("INVALID_DEPTH_VALUE", 65535)),
        "label_names": {int(k): str(v) for k, v in label_names.items()},
        "refine_config": config.get("RGB_REFINE"),
    }


# ----------------------------------------------------------------------
# JSON
# ----------------------------------------------------------------------
def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]

    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, (str, int, bool)) or value is None:
        return value

    return str(value)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            json_safe(data),
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_detection_json(det, label_names, use_refine=True):
    label = int(det.get("label", -1))
    name = label_names.get(label, f"class_{label}")

    info = det.get("refine_info") or {}
    depth_info = info.get("depth") or det.get("depth_refine_info") or {}

    rgb_ok = bool(info.get("ok", det.get("refined", False)))
    depth_ok = bool(depth_info.get("ok", False))
    depth_applied = bool(info.get("depth_applied", det.get("depth_refined", False)))

    t_raw = det.get("translation_mm_raw", det.get("translation_mm"))
    t_rgb = det.get("translation_mm_rgb", det.get("translation_mm"))
    t_final = det.get("translation_mm")

    r_raw = det.get("rotation_raw", det.get("rotation"))
    r_final = det.get("rotation")

    if depth_ok and depth_applied:
        final_source = "depth_refined"
    elif rgb_ok:
        final_source = "rgb_refined"
    else:
        final_source = "race6d_raw"

    return {
        "label": label,
        "name": name,
        "confidence": det.get("confidence"),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "refined": bool(det.get("refined", rgb_ok)),
        "pose": {
            "raw": {
                "translation_mm": t_raw,
                "rotation": r_raw,
                "quaternion": det.get("quat_raw"),
            },
            "rgb_refined": {
                "translation_mm": t_rgb,
                "rotation": det.get("rotation_rgb"),
            },
            "final": {
                "translation_mm": t_final,
                "rotation": r_final,
                "quaternion": det.get("quat"),
            },
        },
        "refinement": {
            "rgb": {
                "enabled": bool(use_refine),
                "ok": rgb_ok if use_refine else False,
                "method": info.get("method"),
                "fallback": bool(info.get("fallback", False)),
                "edge_reason": info.get("edge_reason"),
                "reason": info.get(
                    "reason",
                    "refine_info_missing" if use_refine else "disabled",
                ),
                "details": info,
            },
            "depth": {
                "enabled": bool(use_refine),
                "ok": depth_ok,
                "applied": depth_applied,
                "reason": depth_info.get(
                    "reason",
                    "depth_info_missing" if use_refine else "disabled",
                ),
                "details": depth_info,
            },
            "final_source": final_source,
        },
        "raw_detection": det,
    }


# ----------------------------------------------------------------------
# Input / IO
# ----------------------------------------------------------------------
def collect_pairs(input_value: str):
    path = Path(input_value).expanduser().resolve()

    if path.is_file():
        colors = [path]
    elif path.is_dir():
        colors = sorted(path.glob(f"*{COLOR_SUFFIX}"))
    else:
        raise FileNotFoundError(f"Input not found: {path}")

    pairs = []

    for color in colors:
        if not color.name.endswith(COLOR_SUFFIX):
            print(f"[WARN] Skip: {color.name}")
            continue

        stem = color.name[: -len(COLOR_SUFFIX)]
        depth = color.with_name(stem + DEPTH_SUFFIX)

        if not depth.is_file():
            print(f"[WARN] Skip {color.name}: missing {depth.name}")
            continue

        pairs.append((stem, color, depth))

    return pairs


def read_color(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Cannot read color image: {path}")

    if img.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
        raise RuntimeError(
            f"Color image must be {INPUT_WIDTH}x{INPUT_HEIGHT}, "
            f"got {img.shape[1]}x{img.shape[0]}: {path.name}"
        )

    return img


def read_depth(path: Path, color_shape):
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Cannot read depth image: {path}")

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype != np.uint16:
        print(
            f"[WARN] {path.name}: depth dtype is {depth.dtype}, " "expected uint16 (mm)"
        )

    if depth.shape != tuple(color_shape[:2]):
        raise RuntimeError(
            f"RGB/depth size mismatch: RGB={tuple(color_shape[:2])}, "
            f"Depth={depth.shape} ({path.name})"
        )

    return depth


# ----------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------
def quaternion_to_rotation_matrix(q):
    q = np.asarray(q, dtype=np.float64).reshape(-1)

    if q.size != 4:
        raise ValueError(f"Invalid quaternion: {q}")

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero.")

    qw, qx, qy, qz = q / norm

    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def project_point(point, K):
    x, y, z = point

    if not np.isfinite(z) or z <= 1e-6:
        return None

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    return int(round(u)), int(round(v))


def rotation_angle_deg(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def put_text(img, text, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(
        img,
        text,
        org,
        FONT,
        scale,
        (0, 0, 0),
        thick + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        text,
        org,
        FONT,
        scale,
        color,
        thick,
        cv2.LINE_AA,
    )


def draw_axis(image, translation_mm, quaternion, K, axis_length_mm):
    t = np.asarray(translation_mm, dtype=np.float64).reshape(-1)

    if t.size != 3 or not np.all(np.isfinite(t)):
        return False

    try:
        R = quaternion_to_rotation_matrix(quaternion)
    except Exception:
        return False

    points = np.array(
        [
            t,
            t + R[:, 0] * axis_length_mm,
            t + R[:, 1] * axis_length_mm,
            t + R[:, 2] * axis_length_mm,
        ],
        dtype=np.float64,
    )

    projected = [project_point(p, K) for p in points]

    if projected[0] is None:
        return False

    origin = projected[0]
    h, w = image.shape[:2]

    def valid(point):
        if point is None:
            return False
        x, y = point
        return -w <= x <= 2 * w and -h <= y <= 2 * h

    for name, endpoint, color in (
        ("X", projected[1], (0, 0, 255)),
        ("Y", projected[2], (0, 255, 0)),
        ("Z", projected[3], (255, 0, 0)),
    ):
        if not valid(endpoint):
            continue

        cv2.arrowedLine(
            image,
            origin,
            endpoint,
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.15,
        )
        cv2.putText(
            image,
            name,
            (endpoint[0] + 8, endpoint[1]),
            FONT,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return True


def draw_origins(image, det, K):
    t_raw = det.get("translation_mm_raw")
    t = det.get("translation_mm")

    for tt, color, radius in (
        (t_raw, (0, 255, 255), 9),
        (t, (255, 255, 0), 5),
    ):
        if tt is None:
            continue

        p = project_point(np.asarray(tt, dtype=np.float64), K)

        if p is not None:
            cv2.circle(image, p, radius, color, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Detection analysis
# ----------------------------------------------------------------------
def num(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else NAN
    except (TypeError, ValueError):
        return NAN


def depth_median_mm(depth, bbox, invalid, z_max, ratio=0.2):
    x0, y0, x1, y1 = bbox

    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    hw = (x1 - x0) * ratio
    hh = (y1 - y0) * ratio

    patch = depth[
        max(0, int(cy - hh)) : min(depth.shape[0], int(cy + hh)),
        max(0, int(cx - hw)) : min(depth.shape[1], int(cx + hw)),
    ].astype(np.float32)

    patch = patch[(patch > 0) & (patch != invalid) & (patch <= z_max)]

    return float(np.median(patch)) if patch.size >= 50 else None


def half_thickness_mm(refine_cfg, label):
    try:
        return float(refine_cfg["LABELS"][int(label)]["SIZE_M"]["Z"]) * 500.0
    except Exception:
        return None


def analyse_detection(det, image_name, depth, cfg, label_names):
    label = int(det.get("label", -1))
    name = label_names.get(label, f"class_{label}")

    t = np.asarray(det["translation_mm"], dtype=np.float64)
    t_raw = np.asarray(
        det.get("translation_mm_raw", t),
        dtype=np.float64,
    )

    R = np.asarray(det["rotation"], dtype=np.float64)
    R_raw = np.asarray(
        det.get("rotation_raw", R),
        dtype=np.float64,
    )

    info = det.get("refine_info") or {}

    refined = bool(det.get("refined", False))
    method = info.get("method", "-")
    reason = info.get("reason", "n/a")
    fallback = bool(info.get("fallback", False))
    edge_reason = info.get("edge_reason", "") or ""

    einfo = info.get("edge") or info

    shifts = [num(v) for v in (einfo.get("edge_shift_px") or [])]
    shifts = shifts[:4] + [NAN] * max(0, 4 - len(shifts))

    dep = depth_median_mm(
        depth,
        det["bbox_xyxy"],
        cfg["invalid_depth_value"],
        cfg["depth_z_max_mm"],
    )

    half = half_thickness_mm(cfg["refine_config"], label)

    z_err_raw = z_err_ref = NAN

    if dep is not None and half is not None:
        z_err_raw = float(t_raw[2] - dep - half)
        z_err_ref = float(t[2] - dep - half)

    row = {
        "image": image_name,
        "label": label,
        "name": name,
        "confidence": float(det.get("confidence", NAN)),
        "refined": int(refined),
        "method": method,
        "fallback": int(fallback),
        "reason": reason,
        "edge_reason": edge_reason,
        "tx_raw": t_raw[0],
        "ty_raw": t_raw[1],
        "tz_raw": t_raw[2],
        "tx": t[0],
        "ty": t[1],
        "tz": t[2],
        "dt_mm": float(np.linalg.norm(t - t_raw)),
        "dang_deg": rotation_angle_deg(R, R_raw),
        "edge_shift_0": shifts[0],
        "edge_shift_1": shifts[1],
        "edge_shift_2": shifts[2],
        "edge_shift_3": shifts[3],
        "corner_shift_px": num(einfo.get("corner_shift_px")),
        "area_ratio": num(einfo.get("area_ratio")),
        "n_ref_kp": NAN,
        "n_patch_kp": NAN,
        "n_matches": NAN,
        "n_gated": NAN,
        "n_h_inliers": NAN,
        "n_inliers": NAN,
        "shift_dx_mm": NAN,
        "shift_dy_mm": NAN,
        "h_scale": NAN,
        "h_rot_deg": NAN,
        "coverage": NAN,
        "reproj_px": num(info.get("reproj_px", einfo.get("reproj_px"))),
        "depth_med_mm": dep if dep is not None else NAN,
        "z_err_raw_mm": z_err_raw,
        "z_err_ref_mm": z_err_ref,
    }

    def f(v, nd=1):
        return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"

    tag = "REFINED" if refined else "RAW"

    lines = [
        f"{name} {row['confidence']:.2f} [{tag}/{method}] {reason}",
        (
            f"T raw=({t_raw[0]:.0f},{t_raw[1]:.0f},{t_raw[2]:.0f}) "
            f"ref=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f}) "
            f"dt={f(row['dt_mm'])}mm dR={f(row['dang_deg'])}deg"
        ),
        (
            f"edge shift(px)=[{', '.join(f(v) for v in shifts)}] "
            f"corner_shift={f(row['corner_shift_px'])} "
            f"area_ratio={f(row['area_ratio'], 2)} "
            f"reproj={f(row['reproj_px'], 2)}"
        ),
        (
            f"depth={f(row['depth_med_mm'], 0)}mm "
            f"z_err raw={f(z_err_raw)} ref={f(z_err_ref)}"
        ),
    ]

    return row, lines


# ----------------------------------------------------------------------
# Result processing
# ----------------------------------------------------------------------
def process_result(
    stem,
    color_path,
    depth_path,
    rgb,
    depth,
    result,
    estimator,
    cfg,
    label_names,
    out_dir,
    scale,
):
    image_record = {
        "schema_version": 4,
        "image": stem,
        "mode": "edge_only",
        "inputs": {
            "color": str(color_path),
            "depth": str(depth_path),
        },
        "success": False,
        "detection_count": 0,
        "detections": [],
        "summary": {},
        "error": None,
    }

    rows = []

    try:
        detections = result.get("detections", [])

        if not isinstance(detections, list):
            detections = []

        debug_full = result.get("debug_image")
        debug_full = rgb.copy() if debug_full is None else debug_full.copy()

        text_blocks = []
        image_detections = []

        for det in detections:
            if not isinstance(det, dict):
                continue

            image_detections.append(build_detection_json(det, label_names, True))

            try:
                draw_axis(
                    debug_full,
                    det.get("translation_mm"),
                    det.get("quat"),
                    DEFAULT_K,
                    AXIS_LENGTH_MM,
                )
                draw_origins(debug_full, det, DEFAULT_K)
            except Exception as e:
                print(f"[WARN] Draw failed: {type(e).__name__}: {e}")

            row, lines = analyse_detection(
                det,
                stem,
                depth,
                cfg,
                label_names,
            )

            rows.append(row)
            text_blocks.append(lines)

        image_record["success"] = True
        image_record["detection_count"] = len(image_detections)
        image_record["detections"] = image_detections

        image_record["summary"] = {
            "rgb_refine_success": sum(
                d["refinement"]["rgb"]["ok"] for d in image_detections
            ),
            "rgb_refine_failure": sum(
                d["refinement"]["rgb"]["enabled"] and not d["refinement"]["rgb"]["ok"]
                for d in image_detections
            ),
            "rgb_methods": dict(
                Counter(
                    d["refinement"]["rgb"]["method"]
                    for d in image_detections
                    if d["refinement"]["rgb"]["ok"]
                )
            ),
        }

        if not text_blocks:
            text_blocks = [["No detection"]]

        view = debug_full[:, CROP_X : CROP_X + CROP_WIDTH].copy()

        y = 32

        for lines in text_blocks:
            for i, line in enumerate(lines):
                put_text(
                    view,
                    line,
                    (15, y),
                    scale=0.6 if i else 0.75,
                    color=(255, 255, 255) if i else (0, 255, 255),
                )
                y += 26
            y += 10

        put_text(
            view,
            f"{stem} [EDGE ONLY]",
            (15, view.shape[0] - 15),
            scale=0.7,
        )

        put_text(
            view,
            "yellow=RACE top  cyan=refined  RGB arrows=final pose",
            (15, view.shape[0] - 45),
            scale=0.5,
            color=(200, 200, 200),
            thick=1,
        )

        out_w = int(round(CROP_WIDTH * scale))
        out_h = int(round(INPUT_HEIGHT * scale))

        if (out_w, out_h) != (view.shape[1], view.shape[0]):
            view = cv2.resize(
                view,
                (out_w, out_h),
                interpolation=cv2.INTER_AREA,
            )

        cv2.imwrite(
            str(out_dir / f"{stem}_race6d_edge.png"),
            view,
        )

    except Exception as e:
        image_record["error"] = {
            "type": type(e).__name__,
            "message": str(e),
        }
        print(f"[ERROR] {stem}: {type(e).__name__}: {e}")

    return image_record, rows


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def median_value(rows, key):
    values = np.asarray(
        [r[key] for r in rows],
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]

    return float(np.median(values)) if values.size else None


def build_summary(
    pairs,
    image_records,
    rows,
    timings,
    eligible,
    edge_cfg,
    device,
    out_dir,
):
    refined_rows = [r for r in rows if r["refined"]]
    failed_rows = [r for r in rows if not r["refined"]]

    reasons = Counter(r["reason"] for r in failed_rows)
    edge_reasons = Counter(r["edge_reason"] or r["reason"] for r in failed_rows)

    methods = Counter(r["method"] for r in rows)

    timing_ms = np.asarray(timings, dtype=np.float64) * 1000.0

    return {
        "schema_version": 1,
        "mode": "edge_only",
        "input": str(Path(pairs[0][1]).parent) if pairs else None,
        "output": str(out_dir),
        "device": device,
        "configuration": {
            "texture_enabled": False,
            "eligible_labels": sorted(eligible),
            "params": edge_cfg.get("PARAMS", {}),
            "labels": edge_cfg.get("LABELS", {}),
        },
        "images": {
            "total": len(pairs),
            "processed": sum(bool(r.get("success")) for r in image_records),
            "failed": sum(not bool(r.get("success")) for r in image_records),
        },
        "detections": {
            "total": len(rows),
            "refined": len(refined_rows),
            "failed": len(failed_rows),
            "success_rate": (len(refined_rows) / len(rows) if rows else None),
            "methods": dict(methods),
            "failure_reasons": dict(reasons),
            "edge_reasons": dict(edge_reasons),
        },
        "metrics": {
            "predict_mean_ms": (float(np.mean(timing_ms)) if timing_ms.size else None),
            "predict_median_ms": (
                float(np.median(timing_ms)) if timing_ms.size else None
            ),
            "dt_median_mm": median_value(refined_rows, "dt_mm"),
            "dang_median_deg": median_value(refined_rows, "dang_deg"),
            "reproj_median_px": median_value(refined_rows, "reproj_px"),
            "corner_shift_median_px": median_value(refined_rows, "corner_shift_px"),
            "area_ratio_median": median_value(refined_rows, "area_ratio"),
            "edge_shift_median_px": [
                median_value(
                    [{"value": r[f"edge_shift_{i}"]} for r in rows],
                    "value",
                )
                for i in range(4)
            ],
        },
        "failed_detections": [
            {
                "image": r["image"],
                "label": r["label"],
                "name": r["name"],
                "reason": r["reason"],
                "edge_reason": r["edge_reason"],
                "edge_shift_px": [r[f"edge_shift_{i}"] for i in range(4)],
                "corner_shift_px": r["corner_shift_px"],
                "area_ratio": r["area_ratio"],
                "reproj_px": r["reproj_px"],
            }
            for r in failed_rows
        ],
        "image_records": image_records,
    }


def print_summary(summary):
    images = summary["images"]
    detections = summary["detections"]
    metrics = summary["metrics"]

    print("\n" + "=" * 76)
    print("EDGE REFINEMENT SUMMARY")
    print("=" * 76)

    print(f"Images       : {images['processed']}/{images['total']}")
    print(f"Detections   : {detections['total']}")

    if detections["total"]:
        print(
            f"Refined      : {detections['refined']}/"
            f"{detections['total']} "
            f"({detections['success_rate'] * 100:.1f}%)"
        )

    if metrics["predict_mean_ms"] is not None:
        print(
            f"Predict      : mean={metrics['predict_mean_ms']:.1f} ms "
            f"median={metrics['predict_median_ms']:.1f} ms"
        )

    for key, label, suffix in (
        ("dt_median_mm", "dt median", "mm"),
        ("dang_median_deg", "dR median", "deg"),
        ("reproj_median_px", "reproj median", "px"),
        ("corner_shift_median_px", "corner shift median", "px"),
        ("area_ratio_median", "area ratio median", ""),
    ):
        value = metrics.get(key)

        if value is not None:
            print(f"{label:<20}: {value:.3f} {suffix}")

    if detections["failure_reasons"]:
        print(
            "Failure reasons: "
            + ", ".join(
                f"{k}={v}"
                for k, v in Counter(detections["failure_reasons"]).most_common()
            )
        )

    if detections["edge_reasons"]:
        print(
            "EDGE reasons    : "
            + ", ".join(
                f"{k}={v}" for k, v in Counter(detections["edge_reasons"]).most_common()
            )
        )


# ----------------------------------------------------------------------
# Estimator
# ----------------------------------------------------------------------
def make_estimator(cfg, refine_cfg):
    estimator = PoseEstimator(
        model_path=str(cfg["model_path"]),
        config_path=str(cfg["model_config"]),
        device="cuda" if torch.cuda.is_available() else "cpu",
        score_threshold=cfg["score_threshold"],
        max_per_class=cfg["max_per_class"],
        max_detections=cfg["max_detections"],
        depth_z_max_mm=cfg["depth_z_max_mm"],
        invalid_depth_value=cfg["invalid_depth_value"],
        class_id=cfg["class_id"],
        refine_config=refine_cfg,
    )

    if estimator.refiner is None:
        raise RuntimeError("EDGE refiner failed to initialize")

    eligible = {int(k) for k in estimator.refiner.boxes}

    if not eligible:
        raise RuntimeError("No EDGE refinement labels configured")

    return estimator, eligible


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
        raise SystemExit(
            f"No *{COLOR_SUFFIX} / *{DEPTH_SUFFIX} pairs found: {args.input}"
        )

    cfg = load_race6d_config()

    if not cfg["model_path"].is_file():
        raise FileNotFoundError(f"RACE-6D model not found: {cfg['model_path']}")

    if not cfg["model_config"].is_file():
        raise FileNotFoundError(f"RACE-6D config not found: {cfg['model_config']}")

    base_refine = cfg["refine_config"]

    if not isinstance(base_refine, dict):
        raise SystemExit("[ERROR] RGB_REFINE section is missing")

    if not base_refine.get("ENABLED", False):
        raise SystemExit("[ERROR] RGB_REFINE.ENABLED must be true")

    edge_cfg = copy.deepcopy(base_refine)

    texture_cfg = copy.deepcopy(edge_cfg.get("TEXTURE") or {})
    texture_cfg["ENABLED"] = False
    edge_cfg["TEXTURE"] = texture_cfg

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path.cwd() / "race6d_edge_test_out"
    )

    edge_dir = out_dir / "edge"
    edge_dir.mkdir(parents=True, exist_ok=True)

    camera_matrix = DEFAULT_K.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 76)
    print("RACE-6D EDGE-ONLY OFFLINE TEST")
    print("=" * 76)
    print(f"Input       : {Path(args.input).resolve()}")
    print(f"Pairs       : {len(pairs)}")
    print(f"Output      : {out_dir}")
    print(f"Device      : {device}")
    print(f"Edge labels : " f"{sorted(int(k) for k in (edge_cfg.get('LABELS') or {}))}")
    print("Texture     : disabled")

    print("[INFO] Initializing EDGE estimator...")

    estimator, eligible = make_estimator(cfg, edge_cfg)

    print(f"[INFO] EDGE labels: {sorted(eligible)}")

    image_records = []
    all_rows = []
    timings = []
    n_done = 0

    for stem, color_path, depth_path in pairs:
        print(f"\n--- {stem} ---")

        try:
            rgb = read_color(color_path)
            depth = read_depth(depth_path, rgb.shape)

            if hasattr(estimator.refiner, "last_match_image"):
                estimator.refiner.last_match_image = None

            t0 = time.perf_counter()

            result = estimator.predict(
                rgb=rgb,
                depth=depth,
                intrinsic=camera_matrix,
            )

            elapsed = time.perf_counter() - t0
            timings.append(elapsed)

            image_record, rows = process_result(
                stem=stem,
                color_path=color_path,
                depth_path=depth_path,
                rgb=rgb,
                depth=depth,
                result=result,
                estimator=estimator,
                cfg=cfg,
                label_names=cfg["label_names"],
                out_dir=edge_dir,
                scale=args.scale,
            )

            image_record["predict_seconds"] = elapsed
            image_record["predict_ms"] = elapsed * 1000.0

            image_record["configuration"] = {
                "method": "edge",
                "texture_enabled": False,
                "refine_enabled": True,
                "eligible_labels": sorted(eligible),
            }

            image_records.append(image_record)
            all_rows.extend(rows)

            write_json(
                edge_dir / f"{stem}_race6d_edge.json",
                image_record,
            )

            print(
                f"EDGE: {image_record['detection_count']} detections, "
                f"{elapsed * 1000.0:.1f} ms"
            )

            for row in rows:
                print(
                    f"  {row['name']} label={row['label']} "
                    f"refined={bool(row['refined'])} "
                    f"reason={row['reason']} "
                    f"dt={row['dt_mm']:.2f} mm "
                    f"dR={row['dang_deg']:.2f} deg "
                    f"reproj={row['reproj_px']:.2f} px"
                )

            n_done += 1

        except Exception as e:
            print(f"[ERROR] {stem}: {type(e).__name__}: {e}")

            image_record = {
                "schema_version": 4,
                "image": stem,
                "mode": "edge_only",
                "success": False,
                "error": {
                    "type": type(e).__name__,
                    "message": str(e),
                },
            }

            image_records.append(image_record)

            write_json(
                edge_dir / f"{stem}_race6d_edge.json",
                image_record,
            )

    # CSV
    csv_path = out_dir / "race6d_edge_results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_FIELDS,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary JSON
    summary = build_summary(
        pairs=pairs,
        image_records=image_records,
        rows=all_rows,
        timings=timings,
        eligible=eligible,
        edge_cfg=edge_cfg,
        device=device,
        out_dir=out_dir,
    )

    summary["images"]["processed"] = n_done
    summary["images"]["failed"] = len(pairs) - n_done

    write_json(
        out_dir / "race6d_edge_summary.json",
        summary,
    )

    print_summary(summary)

    print(f"\nOutput       : {out_dir}")
    print(f"Summary JSON : {out_dir / 'race6d_edge_summary.json'}")
    print(f"Results CSV  : {csv_path}")


if __name__ == "__main__":
    main()
