#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ALIGNMENT.alignment import AlignmentEstimator, GT_PATH

APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = APP_DIR / "ALIGNMENT" / "output"

HEIGHT = 0.07
WIDTH = 0.2
ROI_SCALE = 1.0
DEVICE = "cuda"

TARGET_XYZ = [
    0.0209420735844708,
    0.00019818218611398144,
    0.25,
]

TARGET_QUATERNION = [
    -0.05618882243543772,
    0.0024422489260742986,
    0.003846210085275473,
    0.9984097647366446,
]

K = np.array(
    [
        [339.97906, 0.0, 318.17065],
        [0.0, 339.77747, 241.6358],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def find_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if not input_path.name.endswith("_color.jpg"):
            raise ValueError(f"Input file must end with '_color.jpg': {input_path}")
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p
            for p in input_path.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".jpg"
            and p.name.endswith("_color.jpg")
        )

    raise FileNotFoundError(f"Input path not found: {input_path}")


def get_depth_path(image_path: Path) -> Path:
    return image_path.with_name(image_path.name.replace("_color.jpg", "_depth.png"))


def make_json_safe(value, key=None):
    excluded = {
        "debug_image",
        "mask",
        "obb_mask",
        "plane_mask",
        "points",
    }

    if key in excluded:
        return None

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        result = {}
        for k, v in value.items():
            if k in excluded:
                continue
            converted = make_json_safe(v, k)
            if converted is not None:
                result[k] = converted
        return result

    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]

    return value


def save_mask(path: Path, mask) -> bool:
    if mask is None:
        return False

    mask = (np.asarray(mask) > 0).astype(np.uint8) * 255
    return bool(cv2.imwrite(str(path), mask))


def print_quaternion(title: str, quaternion):
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)

    if q.size != 4:
        print(f"{title}: invalid quaternion {q}")
        return

    print(f"{title}:")
    print(f"  qx: {q[0]:.6f}")
    print(f"  qy: {q[1]:.6f}")
    print(f"  qz: {q[2]:.6f}")
    print(f"  qw: {q[3]:.6f}")


def print_pose(title: str, pose):
    if pose is None:
        print(f"{title}: None")
        return

    location = pose.get("location")
    rotation = pose.get("rotation")

    if location is not None:
        xyz = np.asarray(location, dtype=np.float64).reshape(-1)

        if xyz.size == 3:
            print(f"{title} XYZ [m]:")
            print(f"  x: {xyz[0]:.6f}")
            print(f"  y: {xyz[1]:.6f}")
            print(f"  z: {xyz[2]:.6f}")
        else:
            print(f"{title} XYZ: invalid {xyz}")

    if rotation is not None:
        print_quaternion(f"{title} Quaternion", rotation)


def print_geometry_debug(pose):
    if pose is None:
        return

    obb = pose.get("obb")

    if obb is not None:
        print("-" * 60)
        print("INPUT OBB")
        print(f"  center: {obb['center']}")
        print(f"  width : {obb['width']:.3f}")
        print(f"  height: {obb['height']:.3f}")
        print(f"  angle : {obb['angle_deg']:.3f} deg")
        print(f"  area  : {obb.get('area', 0.0):.3f}")

    plane = pose.get("plane")

    if plane is not None:
        print("-" * 60)
        print("INPUT PLANE")
        print(f"  point_count: {plane['point_count']}")
        print(f"  rmse: {plane['rmse']:.6f} m")

        normal = np.asarray(
            plane["normal"],
            dtype=np.float64,
        ).reshape(-1)

        print(f"  normal: {normal.tolist()}")

        obb_mask = plane.get("obb_mask")
        plane_mask = plane.get("plane_mask")

        if obb_mask is not None:
            print(f"  OBB mask pixels: {int(np.count_nonzero(obb_mask))}")

        if plane_mask is not None:
            print(f"  Plane mask pixels: {int(np.count_nonzero(plane_mask))}")

        if plane.get("horizontal_inset") is not None:
            print(f"  H inset: {plane['horizontal_inset']}")

        if plane.get("vertical_inset") is not None:
            print(f"  V inset: {plane['vertical_inset']}")

    orientation = pose.get("orientation_debug")

    if orientation is None:
        return

    print("-" * 60)
    print("INPUT ORIENTATION")

    for key, label in (
        ("image_axis", "image_axis"),
        ("x_axis", "X axis"),
        ("y_axis", "Y axis"),
        ("z_axis", "Z axis"),
    ):
        axis = np.asarray(
            orientation[key],
            dtype=np.float64,
        ).reshape(-1)
        print(f"  {label}: {axis.tolist()}")

    rotation_matrix = np.asarray(
        orientation["rotation_matrix"],
        dtype=np.float64,
    )

    print("  Rotation matrix:")
    for row in rotation_matrix:
        print("    " + " ".join(f"{value: .6f}" for value in row))


def print_delta(delta):
    if delta is None:
        print("DELTA: None")
        return

    location = delta.get("location")
    rotation = delta.get("rotation")
    euler = delta.get("rotation_euler_deg")
    angle = delta.get("rotation_angle_deg")

    if location is not None:
        xyz = np.asarray(location, dtype=np.float64).reshape(-1)
        if xyz.size == 3:
            print("DELTA XYZ [m]:")
            print(f"  x: {xyz[0]:.6f}")
            print(f"  y: {xyz[1]:.6f}")
            print(f"  z: {xyz[2]:.6f}")

    if rotation is not None:
        print_quaternion("DELTA Quaternion", rotation)

    if angle is not None:
        print(f"DELTA Rotation Angle: {float(angle):.6f} deg")

    if euler is not None:
        values = np.asarray(euler, dtype=np.float64).reshape(-1)
        if values.size == 3:
            print("DELTA RPY [deg]:")
            print(f"  roll : {values[0]:.6f}")
            print(f"  pitch: {values[1]:.6f}")
            print(f"  yaw  : {values[2]:.6f}")


def process_image(
    image_path: Path,
    output_dir: Path,
    estimator: AlignmentEstimator,
    location: np.ndarray,
    rotation: np.ndarray,
) -> bool:
    depth_path = get_depth_path(image_path)

    if not depth_path.is_file():
        print(f"[ERROR] Depth not found: {depth_path}")
        return False

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"[ERROR] Cannot load RGB: {image_path}")
        return False

    if depth is None:
        print(f"[ERROR] Cannot load depth: {depth_path}")
        return False

    if depth.ndim != 2:
        print(f"[ERROR] Depth must be single-channel: {depth_path}")
        return False

    if image.shape[:2] != depth.shape[:2]:
        print(
            f"[ERROR] RGB/depth resolution mismatch: "
            f"{image.shape[:2]} vs {depth.shape[:2]}"
        )
        return False

    try:
        result = estimator.predict(
            image,
            depth,
            K,
            location.tolist(),
            rotation.tolist(),
        )
    except Exception as error:
        print(f"[ERROR] Inference failed for {image_path.name}: {error}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    debug_path = output_dir / image_path.name
    json_path = output_dir / f"{image_path.stem}.json"
    mask_dir = output_dir / "masks" / image_path.stem
    mask_dir.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(debug_path), result["debug_image"]):
        print(f"[ERROR] Failed to save debug image: {debug_path}")
        return False

    input_pose = result.get("input_pose")
    sam_mask = result.get("mask")

    if sam_mask is not None:
        sam_mask_path = mask_dir / "sam_mask.png"
        if not save_mask(sam_mask_path, sam_mask):
            print(f"[WARNING] Failed to save SAM2 mask: {sam_mask_path}")

    if input_pose is not None:
        plane = input_pose.get("plane")

        if plane is not None:
            for key, filename in (
                ("obb_mask", "obb_mask.png"),
                ("plane_mask", "plane_mask.png"),
            ):
                mask = plane.get(key)
                if mask is not None:
                    mask_path = mask_dir / filename
                    if not save_mask(mask_path, mask):
                        print(f"[WARNING] Failed to save mask: {mask_path}")

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            make_json_safe(result),
            file,
            indent=4,
            ensure_ascii=False,
        )

    mask_area = int(np.count_nonzero(sam_mask)) if sam_mask is not None else 0
    score = result.get("sam_score")

    print()
    print("=" * 60)
    print(f"[INPUT]  RGB  : {image_path}")
    print(f"[INPUT]  Depth: {depth_path}")
    print(f"[OUTPUT] Image: {debug_path}")
    print(f"[OUTPUT] JSON : {json_path}")
    print(f"[OUTPUT] Masks: {mask_dir}")
    print("=" * 60)
    print(f"[RESULT] Present: {result.get('present')}")
    print(
        f"         BBox={result.get('bbox')} | SAM={score:.3f}"
        if score is not None
        else f"         BBox={result.get('bbox')} | SAM=N/A"
    )
    print(f"         SAM mask area={mask_area}")
    print("-" * 60)

    print_pose("GT", result.get("gt_pose"))
    print("-" * 60)
    print_pose("INPUT", input_pose)
    print_geometry_debug(input_pose)
    print("-" * 60)
    print_delta(result.get("delta"))
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(description="Offline 6DoF Alignment inference")
    parser.add_argument(
        "--input",
        required=True,
        help="Directory or *_color.jpg image path",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output directory",
    )
    parser.add_argument(
        "--device",
        default=DEVICE,
        help="Inference device, e.g. cuda or cpu",
    )
    args = parser.parse_args()

    if HEIGHT <= 0 or WIDTH <= 0 or ROI_SCALE <= 0:
        parser.error("HEIGHT, WIDTH and ROI_SCALE must be > 0")

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not GT_PATH.is_file():
        raise FileNotFoundError(f"GT JSON not found: {GT_PATH}")

    images = find_images(input_path)

    if not images:
        raise RuntimeError(f"No *_color.jpg images found in: {input_path}")

    location = np.asarray(TARGET_XYZ, dtype=np.float64)
    rotation = np.asarray(TARGET_QUATERNION, dtype=np.float64)

    if location.shape != (3,) or not np.all(np.isfinite(location)):
        raise ValueError("TARGET_XYZ must contain three finite values")

    if rotation.shape != (4,) or not np.all(np.isfinite(rotation)):
        raise ValueError("TARGET_QUATERNION must contain four finite values")

    quaternion_norm = np.linalg.norm(rotation)
    if quaternion_norm < 1e-8:
        raise ValueError("TARGET_QUATERNION must not be a zero quaternion")

    rotation /= quaternion_norm

    print("=" * 60)
    print("6DoF ALIGNMENT OFFLINE INFERENCE")
    print("=" * 60)
    print(f"[INPUT]       {input_path}")
    print(f"[OUTPUT]      {output_dir}")
    print(f"[GT]          {GT_PATH}")
    print(f"[INFO]        Found {len(images)} RGB image(s)")
    print(f"[INFO]        Height: {HEIGHT:.6f} m")
    print(f"[INFO]        Width : {WIDTH:.6f} m")
    print(f"[INFO]        ROI scale: {ROI_SCALE:.3f}")
    print(f"[INFO]        Device: {args.device}")
    print(f"[INFO]        fx = {K[0, 0]:.6f}")
    print(f"[INFO]        fy = {K[1, 1]:.6f}")
    print(f"[INFO]        cx = {K[0, 2]:.6f}")
    print(f"[INFO]        cy = {K[1, 2]:.6f}")
    print("=" * 60)

    print_pose(
        "TARGET",
        {"location": location, "rotation": rotation},
    )

    print("[INFO] Initializing AlignmentEstimator...")

    estimator = AlignmentEstimator(
        height=HEIGHT,
        width=WIDTH,
        device=args.device,
        roi_scale=ROI_SCALE,
    )

    print("[INFO] AlignmentEstimator ready.")
    print("-" * 60)

    success = 0

    for image_path in images:
        if process_image(
            image_path,
            output_dir,
            estimator,
            location,
            rotation,
        ):
            success += 1

    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"[INFO] Processed: {success}/{len(images)}")
    print(f"[INFO] Output   : {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
