#!/usr/bin/env python3
"""RACE-6D offline comparison: independent EDGE-only and TEXTURE-only refinement.

Usage
    python race6d_check.py --input ../assets/race_6d/
    python race6d_check.py --input ../assets/race_6d/xxx_color.png        # single image
The script runs two independent estimators on every input:
- EDGE: texture fallback disabled.
- TEXTURE: MIN_VALID_EDGES=5 forces the edge stage to fail, then texture refines
  directly from the original RACE-6D pose.

Input folder: pairs of  <name>_color.png  and  <name>_depth.png  (16-bit depth, mm).

Output (in --out-dir, default ./race6d_compare_out)
    edge/<name>_race6d_edge.png and texture/<name>_race6d_texture.png
    <name>_comparison.json     : both result records and per-detection differences
    race6d_refine_comparison.csv : tabular comparison

Overlay legend
    yellow polygon : RACE top-face projection (before refine)
    edge method    : orange = RACE bottom face, white dots / magenta lines = detected edges,
                     green dots = line intersections
    texture method : green dots = inlier matches, red dots = rejected matches
    cyan polygon   : refined top face (only when a refine method was accepted)
    yellow ring    : RACE origin,  cyan ring : final origin
    RGB arrows     : final pose axes (X red, Y green, Z blue)
"""

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
# args / config
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare independent EDGE-only and TEXTURE-only RACE-6D refinement."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Folder with <name>_color.png / <name>_depth.png pairs, "
        "or a single *_color.png file.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor for the output image (default 1.0).",
    )
    parser.add_argument(
        "--out-dir", default="", help="Output directory (default: ./race6d_check_out)."
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
        # RGB_REFINE is a top-level section (None if missing)
        "refine_config": config.get("RGB_REFINE"),
    }


# ----------------------------------------------------------------------
# json
# ----------------------------------------------------------------------
def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    return str(value)


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(data), f, ensure_ascii=False, indent=2)


def build_detection_json(det, label_names, use_refine):
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
                "method": info.get("method"),  # edge | texture | none
                "fallback": bool(info.get("fallback", False)),
                "edge_reason": info.get("edge_reason"),
                "reason": info.get(
                    "reason", "refine_info_missing" if use_refine else "disabled"
                ),
                "details": info,
            },
            "depth": {
                "enabled": bool(use_refine),
                "ok": depth_ok,
                "applied": depth_applied,
                "reason": depth_info.get(
                    "reason", "depth_info_missing" if use_refine else "disabled"
                ),
                "details": depth_info,
            },
            "final_source": final_source,
        },
        "raw_detection": det,
    }


# ----------------------------------------------------------------------
# input pairing / io
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
            print(f"[WARN] Skip (not a *{COLOR_SUFFIX} file): {color.name}")
            continue

        stem = color.name[: -len(COLOR_SUFFIX)]
        depth = color.with_name(stem + DEPTH_SUFFIX)

        if not depth.is_file():
            print(f"[WARN] Skip {color.name}: missing {depth.name}")
            continue

        pairs.append((stem, color, depth))

    return pairs


def read_color(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Cannot read color image: {path}")

    if img.shape[:2] != (INPUT_HEIGHT, INPUT_WIDTH):
        raise RuntimeError(
            f"Color image must be {INPUT_WIDTH}x{INPUT_HEIGHT}, "
            f"got {img.shape[1]}x{img.shape[0]}: {path.name}"
        )

    return img


def read_depth(path: Path, color_shape) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Cannot read depth image: {path}")

    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if depth.dtype != np.uint16:
        print(f"[WARN] {path.name}: depth dtype is {depth.dtype}, expected uint16 (mm)")

    if depth.shape != tuple(color_shape[:2]):
        raise RuntimeError(
            f"RGB/depth size mismatch: RGB={tuple(color_shape[:2])}, "
            f"Depth={depth.shape} ({path.name})"
        )

    return depth


# ----------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------
def quaternion_to_rotation_matrix(q) -> np.ndarray:
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


def project_point(point: np.ndarray, K: np.ndarray):
    x, y, z = point

    if not np.isfinite(z) or z <= 1e-6:
        return None

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    return int(round(u)), int(round(v))


def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


# ----------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------
def put_text(img, text, org, color=(255, 255, 255), scale=0.7, thick=2):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_axis(image, translation_mm, quaternion, K, axis_length_mm) -> bool:
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

        cv2.arrowedLine(image, origin, endpoint, color, 3, cv2.LINE_AA, tipLength=0.15)
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


def draw_origins(image, det, K) -> None:
    """Yellow ring = RACE origin, cyan ring = final origin."""
    t_raw = det.get("translation_mm_raw")
    t = det.get("translation_mm")

    for tt, color, radius in ((t_raw, (0, 255, 255), 9), (t, (255, 255, 0), 5)):
        if tt is None:
            continue
        p = project_point(np.asarray(tt, dtype=np.float64), K)
        if p is not None:
            cv2.circle(image, p, radius, color, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------
# per-detection analysis
# ----------------------------------------------------------------------
def num(v) -> float:
    try:
        v = float(v)
        return v if np.isfinite(v) else NAN
    except (TypeError, ValueError):
        return NAN


def depth_median_mm(depth, bbox, invalid, z_max, ratio=0.2):
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) * ratio, (y1 - y0) * ratio

    patch = depth[
        max(0, int(cy - hh)) : int(cy + hh),
        max(0, int(cx - hw)) : int(cx + hw),
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
    t_raw = np.asarray(det.get("translation_mm_raw", t), dtype=np.float64)
    R = np.asarray(det["rotation"], dtype=np.float64)
    R_raw = np.asarray(det.get("rotation_raw", R), dtype=np.float64)

    info = det.get("refine_info") or {}
    refined = bool(det.get("refined", False))
    method = info.get("method", "-")
    reason = info.get("reason", "n/a")
    fallback = bool(info.get("fallback", False))
    edge_reason = info.get("edge_reason", "") or ""

    # edge-refiner info: nested when the texture fallback ran, otherwise the top level
    einfo = info.get("edge") or info
    # texture-refiner info: top level only when the fallback ran
    tinfo = info if fallback else {}

    shifts = [num(v) for v in (einfo.get("edge_shift_px") or [])]
    shifts = shifts + [NAN] * (4 - len(shifts))
    shift = tinfo.get("shift_mm") or [NAN, NAN]
    orient = tinfo.get("orient") or {}

    dep = depth_median_mm(
        depth, det["bbox_xyxy"], cfg["invalid_depth_value"], cfg["depth_z_max_mm"]
    )
    half = half_thickness_mm(cfg["refine_config"], label)

    # ideal: z_center - depth_surface == half thickness -> err == 0 (rough for tilted boxes)
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
        "n_ref_kp": num(tinfo.get("n_ref_kp")),
        "n_patch_kp": num(tinfo.get("n_patch_kp")),
        "n_matches": num(tinfo.get("n_matches")),
        "n_gated": num(tinfo.get("n_gated")),
        "n_h_inliers": num(tinfo.get("n_h_inliers")),
        "n_inliers": num(tinfo.get("n_inliers")),
        "shift_dx_mm": num(shift[0]),
        "shift_dy_mm": num(shift[1]),
        "h_scale": num(tinfo.get("h_scale")),
        "h_rot_deg": num(tinfo.get("h_rot_deg")),
        "coverage": num(tinfo.get("coverage")),
        "reproj_px": num(info.get("reproj_px")),
        "depth_med_mm": dep if dep is not None else NAN,
        "z_err_raw_mm": z_err_raw,
        "z_err_ref_mm": z_err_ref,
    }

    def f(v, nd=1):
        return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"

    tag = "REFINED" if refined else "RAW"
    lines = [
        f"{name} {row['confidence']:.2f}  [{tag}/{method}] {reason}",
        f"T raw=({t_raw[0]:.0f},{t_raw[1]:.0f},{t_raw[2]:.0f})  "
        f"ref=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})  dt={f(row['dt_mm'])}mm  dR={f(row['dang_deg'])}deg",
    ]

    shift_txt = "[" + ", ".join(f(v) for v in shifts) + "]"
    if fallback:
        e = f"edge failed: {edge_reason}"
        if np.isfinite(shifts).any():
            e += f"  shift(px)={shift_txt}"
        lines.append(e)
        lines.append(
            f"tex kp ref/patch={f(row['n_ref_kp'], 0)}/{f(row['n_patch_kp'], 0)}  "
            f"match all/gate/H/pnp={f(row['n_matches'], 0)}/{f(row['n_gated'], 0)}/"
            f"{f(row['n_h_inliers'], 0)}/{f(row['n_inliers'], 0)}"
        )
        lines.append(
            f"tex shift=({f(row['shift_dx_mm'])},{f(row['shift_dy_mm'])})mm  "
            f"h_scale={f(row['h_scale'], 3)}  h_rot={f(row['h_rot_deg'])}deg  "
            f"cov={f(row['coverage'], 2)}  reproj={f(row['reproj_px'], 2)}"
        )
        if orient:
            parts = " ".join(
                f"{k}={v.get('h_inliers', 0)}"
                for k, v in orient.items()
                if k != "_best"
            )
            lines.append(f"tex orient best={orient.get('_best')}  H-inliers: {parts}")
    elif method == "edge" or np.isfinite(shifts).any():
        lines.append(
            f"edge shift(px)={shift_txt}  corner_shift={f(row['corner_shift_px'])}  "
            f"area_ratio={f(row['area_ratio'], 2)}  reproj={f(row['reproj_px'], 2)}"
        )

    lines.append(
        f"depth={f(row['depth_med_mm'], 0)}mm  z_err raw={f(z_err_raw)}  ref={f(z_err_ref)}"
    )

    return row, lines


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------
def _med(rows, key):
    a = np.array([r[key] for r in rows], dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else None


def print_summary(rows, timings, use_refine, n_images, eligible):
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Images         : {n_images}")

    if timings:
        arr = np.asarray(timings) * 1000.0
        print(
            f"predict() time : mean={arr.mean():.1f} ms  "
            f"median={np.median(arr):.1f} ms  max={arr.max():.1f} ms"
        )

    if not rows:
        print("No detections.")
        return

    print(f"Detections     : {len(rows)}")

    def z_line(label, rs, key):
        z = np.array([r[key] for r in rs], dtype=float)
        z = z[np.isfinite(z)]
        if z.size:
            print(
                f"{label:<15}: median={np.median(z):+.1f} mm  mean|.|={np.abs(z).mean():.1f} mm"
            )

    if not use_refine:
        z_line("z_err (raw)", rows, "z_err_raw_mm")
        return

    elig = [r for r in rows if r["label"] in eligible]
    print(
        f"Refine labels  : {sorted(eligible)}  "
        f"(detections: {len(elig)}, other labels: {len(rows) - len(elig)})"
    )
    if not elig:
        print("No detection of a refine-enabled label.")
        return

    ok_rows = [r for r in elig if r["refined"]]
    print(
        f"Refined        : {len(ok_rows)}/{len(elig)} ({100.0 * len(ok_rows) / len(elig):.1f}%)"
    )
    print(
        "Methods        : "
        + ", ".join(
            f"{k}={v}" for k, v in Counter(r["method"] for r in elig).most_common()
        )
    )

    fb = [r for r in elig if r["fallback"]]
    if fb:
        fb_ok = [r for r in fb if r["refined"]]
        print(f"Texture tried  : {len(fb)}  succeeded: {len(fb_ok)}")
        print(
            "Edge reasons   : "
            + ", ".join(
                f"{k}={v}"
                for k, v in Counter(r["edge_reason"] for r in fb).most_common()
            )
        )

    print(
        "Reasons        : "
        + ", ".join(
            f"{k}={v}" for k, v in Counter(r["reason"] for r in elig).most_common()
        )
    )

    # edge shifts exist even for failed edge attempts
    sh = np.array([[r[f"edge_shift_{i}"] for i in range(4)] for r in elig], dtype=float)
    if np.isfinite(sh).any():
        med = np.nanmedian(sh, axis=0)
        print(
            "edge_shift med : [" + ", ".join(f"{v:+.1f}" for v in med) + "] px "
            "(+ = real edge outside prediction -> RACE face too small)"
        )

    if fb:
        for key, label in (
            ("n_ref_kp", "ref keypoints"),
            ("n_patch_kp", "patch keypoints"),
            ("n_matches", "matches"),
            ("n_gated", "after gate"),
            ("n_h_inliers", "H inliers"),
        ):
            v = _med(fb, key)
            if v is not None:
                print(f"tex {label:<11}: median={v:.0f}")
        for key, label, nd in (
            ("h_scale", "h_scale", 3),
            ("shift_dx_mm", "shift dx mm", 1),
            ("shift_dy_mm", "shift dy mm", 1),
            ("h_rot_deg", "h_rot deg", 1),
        ):
            v = _med(fb, key)
            if v is not None:
                print(f"tex {label:<11}: median={v:+.{nd}f}")

    for method in ("edge", "texture"):
        rs = [r for r in ok_rows if r["method"] == method]
        if not rs:
            continue
        print(
            f"[{method}] n={len(rs)}  dt_mm med={_med(rs, 'dt_mm'):.1f}  "
            f"dang_deg med={_med(rs, 'dang_deg'):.1f}  reproj_px med={_med(rs, 'reproj_px'):.2f}"
        )
        z_line(f"  z_err raw", rs, "z_err_raw_mm")
        z_line(f"  z_err refined", rs, "z_err_ref_mm")

    print(
        "(z_err = z_center - depth_median - half_thickness; ideal ~ 0; rough when the box is tilted)"
    )


# ----------------------------------------------------------------------
def make_estimator(cfg, refine_cfg, mode):
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
        raise RuntimeError(f"{mode}: RGB_REFINE failed to initialize")
    eligible = {int(k) for k in estimator.refiner.boxes}
    if mode == "texture" and getattr(estimator.refiner, "texture", None) is None:
        raise RuntimeError(
            "Texture refiner failed to initialize; check RGB_REFINE.TEXTURE, "
            "REF_IMAGE paths and reference features."
        )
    return estimator, eligible


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
    mode,
    out_dir,
    scale,
):
    image_record = {
        "schema_version": 3,
        "image": stem,
        "mode": mode,
        "inputs": {"color": str(color_path), "depth": str(depth_path)},
        "success": False,
        "detection_count": 0,
        "detections": [],
        "summary": {},
        "error": None,
    }
    rows = []
    try:
        m_img = getattr(estimator.refiner, "last_match_image", None)
        if m_img is not None:
            m_img = cv2.resize(
                m_img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST
            )
            cv2.imwrite(str(out_dir / f"{stem}_{mode}_matches.png"), m_img)

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
                print(f"[WARN] {mode} draw failed: {type(e).__name__}: {e}")
            row, lines = analyse_detection(det, stem, depth, cfg, label_names)
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
        put_text(view, f"{stem}  [{mode.upper()}]", (15, view.shape[0] - 15), scale=0.7)
        put_text(
            view,
            "yellow=RACE top  green=accepted  red=rejected  cyan=refined",
            (15, view.shape[0] - 45),
            scale=0.5,
            color=(200, 200, 200),
            thick=1,
        )
        out_w = int(round(CROP_WIDTH * scale))
        out_h = int(round(INPUT_HEIGHT * scale))
        if (out_w, out_h) != (view.shape[1], view.shape[0]):
            view = cv2.resize(view, (out_w, out_h), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{stem}_race6d_{mode}.png"), view)
    except Exception as e:
        image_record["error"] = {"type": type(e).__name__, "message": str(e)}
        print(f"[ERROR] {stem} [{mode}]: {type(e).__name__}: {e}")
    return image_record, rows


def _det_key(det):
    bbox = det.get("bbox_xyxy")
    if bbox is None:
        bbox = []
    return (
        int(det.get("label", -1)),
        tuple(round(float(x), 1) for x in np.asarray(bbox).reshape(-1)),
    )


def compare_detection_records(edge_record, texture_record):
    edge_dets = edge_record.get("detections", [])
    tex_dets = texture_record.get("detections", [])
    tex_map = {_det_key(d): d for d in tex_dets}
    comparisons = []
    for edge in edge_dets:
        key = _det_key(edge)
        tex = tex_map.get(key)
        if tex is None:
            continue
        ep = edge.get("pose", {})
        tp = tex.get("pose", {})
        er = ep.get("raw") or {}
        et = ep.get("final") or {}
        tr = tp.get("raw") or {}
        tt = tp.get("final") or {}
        epos = np.asarray(et.get("translation_mm") or [NAN] * 3, dtype=float)
        tpos = np.asarray(tt.get("translation_mm") or [NAN] * 3, dtype=float)
        raw = np.asarray(er.get("translation_mm") or [NAN] * 3, dtype=float)
        try:
            eR = np.asarray(et["rotation"], dtype=float)
            tR = np.asarray(tt["rotation"], dtype=float)
            dR = rotation_angle_deg(eR, tR)
        except Exception:
            dR = NAN
        comparisons.append(
            {
                "label": key[0],
                "name": edge.get("name"),
                "bbox_xyxy": list(key[1]),
                "edge_ok": bool(edge.get("refinement", {}).get("rgb", {}).get("ok")),
                "edge_method": edge.get("refinement", {}).get("rgb", {}).get("method"),
                "texture_ok": bool(tex.get("refinement", {}).get("rgb", {}).get("ok")),
                "texture_method": tex.get("refinement", {})
                .get("rgb", {})
                .get("method"),
                "raw_translation_mm": raw.tolist(),
                "edge_translation_mm": epos.tolist(),
                "texture_translation_mm": tpos.tolist(),
                "edge_vs_raw_dt_mm": float(np.linalg.norm(epos - raw)),
                "texture_vs_raw_dt_mm": float(np.linalg.norm(tpos - raw)),
                "edge_texture_dt_mm": float(np.linalg.norm(epos - tpos)),
                "edge_texture_dang_deg": dR,
                "edge_reason": edge.get("refinement", {}).get("rgb", {}).get("reason"),
                "texture_reason": tex.get("refinement", {})
                .get("rgb", {})
                .get("reason"),
            }
        )
    return comparisons


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    pairs = collect_pairs(args.input)
    if not pairs:
        raise SystemExit(
            f"No *{COLOR_SUFFIX} / *{DEPTH_SUFFIX} pairs found in: {args.input}"
        )

    cfg = load_race6d_config()
    if not cfg["model_path"].is_file():
        raise FileNotFoundError(f"RACE-6D model not found: {cfg['model_path']}")
    if not cfg["model_config"].is_file():
        raise FileNotFoundError(f"RACE-6D config not found: {cfg['model_config']}")
    base_refine = cfg["refine_config"]
    if not isinstance(base_refine, dict) or not base_refine.get("ENABLED", False):
        raise SystemExit("[ERROR] RGB_REFINE.ENABLED must be true in config.yaml")
    if not (base_refine.get("TEXTURE") or {}).get("ENABLED", False):
        raise SystemExit("[ERROR] RGB_REFINE.TEXTURE.ENABLED must be true")

    edge_cfg = copy.deepcopy(base_refine)
    edge_texture_cfg = copy.deepcopy(edge_cfg.get("TEXTURE") or {})
    edge_texture_cfg["ENABLED"] = False
    edge_cfg["TEXTURE"] = edge_texture_cfg

    texture_cfg = copy.deepcopy(base_refine)
    texture_params = dict(texture_cfg.get("PARAMS") or {})
    texture_params["MIN_VALID_EDGES"] = 5
    texture_cfg["PARAMS"] = texture_params

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path.cwd() / "race6d_compare_out"
    )
    edge_dir = out_dir / "edge"
    texture_dir = out_dir / "texture"
    edge_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)
    camera_matrix = DEFAULT_K.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 76)
    print("RACE-6D OFFLINE REFINE COMPARISON: EDGE vs TEXTURE")
    print("=" * 76)
    print(f"Input       : {Path(args.input).resolve()} ({len(pairs)} pairs)")
    print(f"Output      : {out_dir}")
    print(f"Device      : {device}")
    print(f"Edge labels : {sorted(int(k) for k in (edge_cfg.get('LABELS') or {}))}")
    print("Edge path   : texture fallback disabled")
    print("Texture path: MIN_VALID_EDGES=5; texture runs from original RACE pose")

    print("[INFO] Initializing EDGE estimator...")
    edge_estimator, eligible = make_estimator(cfg, edge_cfg, "edge")
    print("[INFO] Initializing TEXTURE estimator...")
    texture_estimator, texture_eligible = make_estimator(cfg, texture_cfg, "texture")
    if eligible != texture_eligible:
        print(
            f"[WARN] Refine label mismatch: edge={sorted(eligible)}, texture={sorted(texture_eligible)}"
        )

    all_comparisons = []
    edge_rows, texture_rows = [], []
    edge_times, texture_times = [], []
    n_done = 0
    for stem, color_path, depth_path in pairs:
        print(f"\n--- {stem} ---")
        try:
            rgb = read_color(color_path)
            depth = read_depth(depth_path, rgb.shape)
            edge_estimator.refiner.last_match_image = None
            texture_estimator.refiner.last_match_image = None

            t0 = time.perf_counter()
            edge_result = edge_estimator.predict(
                rgb=rgb, depth=depth, intrinsic=camera_matrix
            )
            edge_seconds = time.perf_counter() - t0
            edge_times.append(edge_seconds)

            t0 = time.perf_counter()
            texture_result = texture_estimator.predict(
                rgb=rgb, depth=depth, intrinsic=camera_matrix
            )
            texture_seconds = time.perf_counter() - t0
            texture_times.append(texture_seconds)

            edge_record, erows = process_result(
                stem,
                color_path,
                depth_path,
                rgb,
                depth,
                edge_result,
                edge_estimator,
                cfg,
                cfg["label_names"],
                "edge",
                edge_dir,
                args.scale,
            )
            texture_record, trows = process_result(
                stem,
                color_path,
                depth_path,
                rgb,
                depth,
                texture_result,
                texture_estimator,
                cfg,
                cfg["label_names"],
                "texture",
                texture_dir,
                args.scale,
            )
            edge_record["predict_seconds"] = edge_seconds
            texture_record["predict_seconds"] = texture_seconds
            edge_rows.extend(erows)
            texture_rows.extend(trows)

            comparisons = compare_detection_records(edge_record, texture_record)
            all_comparisons.extend({"image": stem, **c} for c in comparisons)
            compare_record = {
                "schema_version": 1,
                "image": stem,
                "edge": edge_record,
                "texture": texture_record,
                "comparison": comparisons,
            }
            write_json(out_dir / f"{stem}_comparison.json", compare_record)

            print(
                f"EDGE    : {len(edge_record['detections'])} detections, {edge_seconds * 1000:.1f} ms"
            )
            print(
                f"TEXTURE : {len(texture_record['detections'])} detections, {texture_seconds * 1000:.1f} ms"
            )
            for c in comparisons:
                print(
                    f"  {c['name']} label={c['label']} "
                    f"edge={c['edge_ok']}({c['edge_method']}) "
                    f"texture={c['texture_ok']}({c['texture_method']}) "
                    f"dT(edge-texture)={c['edge_texture_dt_mm']:.2f} mm "
                    f"dR={c['edge_texture_dang_deg']:.2f} deg"
                )
            write_json(edge_dir / f"{stem}_race6d_edge.json", edge_record)
            write_json(texture_dir / f"{stem}_race6d_texture.json", texture_record)
            n_done += 1
        except Exception as e:
            print(f"[ERROR] {stem}: {type(e).__name__}: {e}")

    comparison_fields = [
        "image",
        "label",
        "name",
        "edge_ok",
        "edge_method",
        "texture_ok",
        "texture_method",
        "edge_vs_raw_dt_mm",
        "texture_vs_raw_dt_mm",
        "edge_texture_dt_mm",
        "edge_texture_dang_deg",
        "edge_reason",
        "texture_reason",
    ]
    if all_comparisons:
        with (out_dir / "race6d_refine_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=comparison_fields)
            writer.writeheader()
            writer.writerows(
                {k: r.get(k, "") for k in comparison_fields} for r in all_comparisons
            )

    def report(label, rows, times):
        ok = [r for r in rows if r["refined"]]
        print(
            f"{label:<10}: success={len(ok)}/{len(rows)} "
            f"({100 * len(ok) / len(rows):.1f}%)"
            if rows
            else f"{label:<10}: no detections"
        )
        if times:
            a = np.asarray(times) * 1000
            print(
                f"{'':<10}  predict mean={a.mean():.1f} ms median={np.median(a):.1f} ms"
            )
        if ok:
            print(
                f"{'':<10}  dt median={_med(ok, 'dt_mm'):.2f} mm, "
                f"dR median={_med(ok, 'dang_deg'):.2f} deg, "
                f"reproj median={_med(ok, 'reproj_px'):.2f} px"
            )
        fails = [r for r in rows if not r["refined"]]
        if fails:
            print(
                f"{'':<10}  failures: "
                + ", ".join(
                    f"{k}={v}"
                    for k, v in Counter(r["reason"] for r in fails).most_common()
                )
            )

    print("\n" + "=" * 76)
    print("COMPARISON SUMMARY")
    print("=" * 76)
    print(f"Images processed: {n_done}/{len(pairs)}")
    report("EDGE", edge_rows, edge_times)
    report("TEXTURE", texture_rows, texture_times)
    if all_comparisons:
        print(f"Matched detections: {len(all_comparisons)}")
        print(
            f"Median edge-texture translation difference: "
            f"{np.median([x['edge_texture_dt_mm'] for x in all_comparisons]):.2f} mm"
        )
        valid_rot = [
            x["edge_texture_dang_deg"]
            for x in all_comparisons
            if np.isfinite(x["edge_texture_dang_deg"])
        ]
        if valid_rot:
            print(
                f"Median edge-texture rotation difference: {np.median(valid_rot):.2f} deg"
            )
    print(f"Output: {out_dir}")
    print(f"Comparison CSV: {out_dir / 'race6d_refine_comparison.csv'}")


if __name__ == "__main__":
    main()
