#!/usr/bin/env python3
"""RACE-6D image check (with RGB_REFINE diagnostics).

Usage
    python race6d_check.py --input ../assets/race_6d/
    python race6d_check.py --input ../assets/race_6d/ --no-refine     # baseline
    python race6d_check.py --input ../assets/race_6d/xxx_color.png    # single image
    python race6d_check.py --input ../assets/race_6d/ --csv           # also write a CSV

Input folder: pairs of  <name>_color.png  and  <name>_depth.png  (16-bit depth, mm).

Output (in --out-dir, default ./race6d_check_out)
    <name>_race6d_{refine|raw}.png : ONE debug image per input (cropped x=240..1680, full resolution)

Overlay legend
    yellow polygon : RACE top-face projection (before refine)
    green dots     : detected top-face corners
    cyan polygon   : refined top face (only when refine accepted)
    yellow ring    : RACE origin,  cyan ring : final origin
    RGB arrows     : final pose axes (X red, Y green, Z blue)
"""

from __future__ import annotations

import argparse
import csv
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
    "reason",
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
    "reproj_px",
    "area_ratio",
    "depth_med_mm",
    "z_err_raw_mm",
    "z_err_ref_mm",
]


# ----------------------------------------------------------------------
# args / config
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="RACE-6D image check (with RGB_REFINE diagnostics)."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Folder with <name>_color.png / <name>_depth.png pairs, "
        "or a single *_color.png file.",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Disable RGB_REFINE (baseline run for comparison).",
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
    parser.add_argument(
        "--csv", action="store_true", help="Also write a per-detection CSV."
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
    reason = info.get("reason", "n/a")
    shifts = list(info.get("edge_shift_px") or [])
    shifts = shifts + [np.nan] * (4 - len(shifts))

    dep = depth_median_mm(
        depth, det["bbox_xyxy"], cfg["invalid_depth_value"], cfg["depth_z_max_mm"]
    )
    half = half_thickness_mm(cfg["refine_config"], label)

    # ideal: z_center - depth_surface == half thickness -> err == 0 (rough for tilted boxes)
    z_err_raw = z_err_ref = np.nan
    if dep is not None and half is not None:
        z_err_raw = float(t_raw[2] - dep - half)
        z_err_ref = float(t[2] - dep - half)

    row = {
        "image": image_name,
        "label": label,
        "name": name,
        "confidence": float(det.get("confidence", np.nan)),
        "refined": int(refined),
        "reason": reason,
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
        "corner_shift_px": info.get("corner_shift_px", np.nan),
        "reproj_px": info.get("reproj_px", np.nan),
        "area_ratio": info.get("area_ratio", np.nan),
        "depth_med_mm": dep if dep is not None else np.nan,
        "z_err_raw_mm": z_err_raw,
        "z_err_ref_mm": z_err_ref,
    }

    def f(v, nd=1):
        return "nan" if v is None or not np.isfinite(v) else f"{v:.{nd}f}"

    tag = "REFINED" if refined else "RAW"
    lines = [
        f"{name} {row['confidence']:.2f}  [{tag}] {reason}",
        f"T raw=({t_raw[0]:.0f},{t_raw[1]:.0f},{t_raw[2]:.0f})  "
        f"ref=({t[0]:.0f},{t[1]:.0f},{t[2]:.0f})  dt={f(row['dt_mm'])}mm  dR={f(row['dang_deg'])}deg",
        "edge_shift(px)=[" + ", ".join(f(v) for v in shifts) + "]  "
        f"corner_shift={f(row['corner_shift_px'])}  reproj={f(row['reproj_px'], 2)}",
        f"depth={f(row['depth_med_mm'], 0)}mm  z_err raw={f(z_err_raw)}  ref={f(z_err_ref)}",
    ]

    return row, lines


# ----------------------------------------------------------------------
# summary
# ----------------------------------------------------------------------
def print_summary(rows, timings, use_refine, n_images):
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

    zr = np.array([r["z_err_raw_mm"] for r in rows], dtype=float)
    zf = np.array([r["z_err_ref_mm"] for r in rows], dtype=float)

    if not use_refine:
        z = zr[np.isfinite(zr)]
        if z.size:
            print(
                f"z_err (raw)    : median={np.median(z):+.1f} mm  mean|.|={np.abs(z).mean():.1f} mm"
            )
        return

    reasons = Counter(r["reason"] for r in rows)
    ok_rows = [r for r in rows if r["refined"]]
    print(
        f"Refined        : {len(ok_rows)}/{len(rows)} ({100.0 * len(ok_rows) / len(rows):.1f}%)"
    )
    print("Reasons        : " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))

    # edge shifts exist even for rejected detections -> use all rows
    sh = np.array([[r[f"edge_shift_{i}"] for i in range(4)] for r in rows], dtype=float)
    if np.isfinite(sh).any():
        med = np.nanmedian(sh, axis=0)
        print(
            "edge_shift med : [" + ", ".join(f"{v:+.1f}" for v in med) + "] px "
            "(+ = real edge outside prediction -> RACE face too small)"
        )

    if ok_rows:

        def med(key):
            return float(np.nanmedian([r[key] for r in ok_rows]))

        print(f"dt_mm          : median={med('dt_mm'):.1f}")
        print(f"dang_deg       : median={med('dang_deg'):.1f}")
        print(f"reproj_px      : median={med('reproj_px'):.2f}")
        print(f"corner_shift   : median={med('corner_shift_px'):.1f} px")

    m = np.isfinite(zr) & np.isfinite(zf)
    if m.any():
        print(
            f"z_err raw      : median={np.median(zr[m]):+.1f} mm  mean|.|={np.abs(zr[m]).mean():.1f} mm"
        )
        print(
            f"z_err refined  : median={np.median(zf[m]):+.1f} mm  mean|.|={np.abs(zf[m]).mean():.1f} mm"
        )
        print(
            "(z_err = z_center - depth_median - half_thickness; ideal ~ 0; rough when the box is tilted)"
        )


# ----------------------------------------------------------------------
def main():
    args = parse_args()

    pairs = collect_pairs(args.input)

    if not pairs:
        raise SystemExit(
            f"No *{COLOR_SUFFIX} / *{DEPTH_SUFFIX} pairs found in: {args.input}"
        )

    cfg = load_race6d_config()
    model_path = cfg["model_path"]
    model_config = cfg["model_config"]
    label_names = cfg["label_names"]

    if not model_path.is_file():
        raise FileNotFoundError(f"RACE-6D model not found: {model_path}")

    if not model_config.is_file():
        raise FileNotFoundError(f"RACE-6D config not found: {model_config}")

    refine_cfg = None if args.no_refine else cfg["refine_config"]
    use_refine = bool(refine_cfg and refine_cfg.get("ENABLED", False))
    mode = "refine" if use_refine else "raw"

    out_dir = (
        Path(args.out_dir).resolve()
        if args.out_dir
        else Path.cwd() / "race6d_check_out"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"race6d_check_{mode}.csv"

    camera_matrix = DEFAULT_K.copy()

    out_w = int(round(CROP_WIDTH * args.scale))
    out_h = int(round(INPUT_HEIGHT * args.scale))

    print("=" * 70)
    print("RACE-6D IMAGE CHECK (RGB_REFINE test)")
    print("=" * 70)
    print(f"Input      : {Path(args.input).resolve()}  ({len(pairs)} pairs)")
    print(f"Output dir : {out_dir}")
    print(f"Model      : {model_path}")
    print(
        f"Mode       : {mode.upper()}"
        + ("" if use_refine else "  (RGB_REFINE disabled or missing)")
    )
    if use_refine:
        print(
            f"Refine lbl : {sorted(int(k) for k in (refine_cfg.get('LABELS') or {}))}"
        )
        print(f"Refine prm : {refine_cfg.get('PARAMS')}")
    print("=" * 70)

    print("[INFO] Initializing RACE-6D...")

    estimator = PoseEstimator(
        model_path=str(model_path),
        config_path=str(model_config),
        device="cuda" if torch.cuda.is_available() else "cpu",
        score_threshold=cfg["score_threshold"],
        max_per_class=cfg["max_per_class"],
        max_detections=cfg["max_detections"],
        depth_z_max_mm=cfg["depth_z_max_mm"],
        invalid_depth_value=cfg["invalid_depth_value"],
        class_id=cfg["class_id"],
        refine_config=refine_cfg if use_refine else None,
    )

    print("[INFO] RACE-6D ready.")

    rows: list[dict] = []
    timings: list[float] = []
    warmed_up = False
    n_done = 0

    for stem, color_path, depth_path in pairs:
        print(f"\n--- {stem} ---")

        try:
            rgb = read_color(color_path)
            depth = read_depth(depth_path, rgb.shape)

            if not warmed_up:  # CUDA warm-up, excluded from timing
                estimator.predict(rgb=rgb, depth=depth, intrinsic=camera_matrix)
                warmed_up = True

            t0 = time.perf_counter()
            result = estimator.predict(rgb=rgb, depth=depth, intrinsic=camera_matrix)
            timings.append(time.perf_counter() - t0)

            detections = result.get("detections", [])
            if not isinstance(detections, list):
                detections = []

            # predict() returns the FULL-resolution debug image
            debug_full = result.get("debug_image")
            debug_full = rgb.copy() if debug_full is None else debug_full.copy()

            text_blocks: list[list[str]] = []
            n_kept = 0

            for det in detections:
                if not isinstance(det, dict):
                    continue

                if not draw_axis(
                    debug_full,
                    det.get("translation_mm"),
                    det.get("quat"),
                    camera_matrix,
                    AXIS_LENGTH_MM,
                ):
                    continue

                draw_origins(debug_full, det, camera_matrix)

                row, lines = analyse_detection(det, stem, depth, cfg, label_names)
                rows.append(row)
                text_blocks.append(lines)
                n_kept += 1

                print("\n".join(lines))

            if not n_kept:
                print("No detection.")
                text_blocks = [["No detection"]]

            # ---- the single debug image (cropped, full resolution) ----
            view = debug_full[:, CROP_X : CROP_X + CROP_WIDTH].copy()

            y = 32
            for lines in text_blocks:
                for i, line in enumerate(lines):
                    put_text(
                        view,
                        line,
                        (15, y),
                        scale=0.65 if i else 0.75,
                        color=(255, 255, 255) if i else (0, 255, 255),
                    )
                    y += 28
                y += 10

            put_text(view, f"{stem}  [{mode}]", (15, view.shape[0] - 15), scale=0.7)
            put_text(
                view,
                "yellow=RACE face  green=corners  cyan=refined",
                (15, view.shape[0] - 45),
                scale=0.55,
                color=(200, 200, 200),
                thick=1,
            )

            if (out_w, out_h) != (view.shape[1], view.shape[0]):
                view = cv2.resize(view, (out_w, out_h), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(out_dir / f"{stem}_race6d_{mode}.png"), view)
            n_done += 1

        except Exception as error:
            print(f"[ERROR] {stem}: {type(error).__name__}: {error}")

    if args.csv and rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in CSV_FIELDS})

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Images      : {n_done}/{len(pairs)}")
    print(f"Detections  : {len(rows)}")
    print(f"Output dir  : {out_dir}")
    if args.csv:
        print(f"CSV         : {csv_path if rows else '(no detections, not written)'}")
    print_summary(rows, timings, use_refine, n_done)


if __name__ == "__main__":
    main()
