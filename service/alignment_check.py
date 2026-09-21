#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from ALIGNMENT.alignment import AlignmentEstimator


APP_DIR = Path(__file__).resolve().parent
DEFAULT_GT_JSON = APP_DIR / "ALIGNMENT" / "GT" / "gt.json"
DEFAULT_OUTPUT = APP_DIR / "ALIGNMENT" / "output"

K = np.array(
    [
        [1070.7897900136718, 0.0, 962.3093872070312],
        [0.0, 1070.5269785615235, 512.0372314453125],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def find_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if not input_path.name.endswith("_color.jpg"):
            raise ValueError(
                f"Input file must end with '_color.jpg': {input_path}"
            )
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p for p in input_path.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".jpg"
            and p.name.endswith("_color.jpg")
        )

    raise FileNotFoundError(f"Input path not found: {input_path}")


def get_depth_path(image_path: Path) -> Path:
    return image_path.with_name(
        image_path.name.replace("_color.jpg", "_depth.png")
    )


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
            print(
                f"  OBB mask pixels: "
                f"{int(np.count_nonzero(obb_mask))}"
            )

        if plane_mask is not None:
            print(
                f"  Plane mask pixels: "
                f"{int(np.count_nonzero(plane_mask))}"
            )

        if plane.get("horizontal_inset") is not None:
            print(
                f"  H inset: "
                f"{plane['horizontal_inset']}"
            )

        if plane.get("vertical_inset") is not None:
            print(
                f"  V inset: "
                f"{plane['vertical_inset']}"
            )

    orientation = pose.get("orientation_debug")

    if orientation is None:
        return

    print("-" * 60)
    print("INPUT ORIENTATION")

    image_axis = np.asarray(
        orientation["image_axis"],
        dtype=np.float64,
    )

    x_axis = np.asarray(
        orientation["x_axis"],
        dtype=np.float64,
    )

    y_axis = np.asarray(
        orientation["y_axis"],
        dtype=np.float64,
    )

    z_axis = np.asarray(
        orientation["z_axis"],
        dtype=np.float64,
    )

    print(f"  image_axis: {image_axis.tolist()}")
    print(f"  X axis: {x_axis.tolist()}")
    print(f"  Y axis: {y_axis.tolist()}")
    print(f"  Z axis: {z_axis.tolist()}")

    R = np.asarray(
        orientation["rotation_matrix"],
        dtype=np.float64,
    )

    print("  Rotation matrix:")
    for row in R:
        print(
            "    " + " ".join(
                f"{value: .6f}" for value in row
            )
        )


def process_image(
    image_path: Path,
    output_dir: Path,
    estimator: AlignmentEstimator,
) -> bool:
    depth_path = get_depth_path(image_path)

    if not depth_path.exists():
        print(f"[ERROR] Depth not found: {depth_path}")
        return False

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    depth = cv2.imread(
        str(depth_path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        print(f"[ERROR] Cannot load RGB: {image_path}")
        return False

    if depth is None:
        print(f"[ERROR] Cannot load depth: {depth_path}")
        return False

    if depth.ndim != 2:
        print(
            f"[ERROR] Depth must be single-channel: "
            f"{depth_path}"
        )
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
        )
    except Exception as error:
        print(
            f"[ERROR] Inference failed for "
            f"{image_path.name}: {error}"
        )
        return False

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    debug_path = output_dir / image_path.name
    json_path = output_dir / f"{image_path.stem}.json"

    mask_dir = (
        output_dir
        / "masks"
        / image_path.stem
    )

    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(debug_path),
        result["debug_image"],
    ):
        print(
            f"[ERROR] Failed to save debug image: "
            f"{debug_path}"
        )
        return False

    input_pose = result.get("input_pose")
    sam_mask = result.get("mask")

    sam_mask_path = mask_dir / "sam_mask.png"

    if sam_mask is not None:
        if not save_mask(
            sam_mask_path,
            sam_mask,
        ):
            print(
                f"[WARNING] Failed to save "
                f"SAM2 mask: {sam_mask_path}"
            )

    if input_pose is not None:
        plane = input_pose.get("plane")

        if plane is not None:
            obb_mask = plane.get("obb_mask")
            plane_mask = plane.get("plane_mask")

            if obb_mask is not None:
                obb_mask_path = (
                    mask_dir / "obb_mask.png"
                )

                if not save_mask(
                    obb_mask_path,
                    obb_mask,
                ):
                    print(
                        f"[WARNING] Failed to save "
                        f"OBB mask: {obb_mask_path}"
                    )

            if plane_mask is not None:
                plane_mask_path = (
                    mask_dir / "plane_mask.png"
                )

                if not save_mask(
                    plane_mask_path,
                    plane_mask,
                ):
                    print(
                        f"[WARNING] Failed to save "
                        f"plane mask: {plane_mask_path}"
                    )

    json_result = make_json_safe(result)

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            json_result,
            f,
            indent=4,
            ensure_ascii=False,
        )

    mask_area = (
        int(np.count_nonzero(sam_mask))
        if sam_mask is not None
        else 0
    )

    score = result.get("sam_score")

    print()
    print("=" * 60)
    print(f"[INPUT]  RGB  : {image_path}")
    print(f"[INPUT]  Depth: {depth_path}")
    print(f"[OUTPUT] Image: {debug_path}")
    print(f"[OUTPUT] JSON : {json_path}")
    print(f"[OUTPUT] Masks: {mask_dir}")
    print("=" * 60)

    print(f"[RESULT] Present: {result['present']}")

    print(
        f"         BBox={result['bbox']} | "
        f"SAM={score:.3f}"
        if score is not None
        else (
            f"         BBox={result['bbox']} | "
            f"SAM=N/A"
        )
    )

    print(f"         SAM mask area={mask_area}")

    print("-" * 60)

    print_pose(
        "GT",
        result.get("gt_pose"),
    )

    print("-" * 60)

    print_pose(
        "INPUT",
        input_pose,
    )

    print_geometry_debug(input_pose)

    print("-" * 60)

    delta = result.get("delta")

    if delta is None:
        print("DELTA: None")
    else:
        delta_location = delta.get("location")
        delta_rotation = delta.get("rotation")
        delta_euler = delta.get("rotation_euler_deg")
        delta_angle = delta.get("rotation_angle_deg")

        if delta_location is not None:
            xyz = np.asarray(
                delta_location,
                dtype=np.float64,
            ).reshape(-1)

            if xyz.size == 3:
                print("DELTA XYZ [m]:")
                print(f"  x: {xyz[0]:.6f}")
                print(f"  y: {xyz[1]:.6f}")
                print(f"  z: {xyz[2]:.6f}")

        if delta_rotation is not None:
            print_quaternion(
                "DELTA Quaternion",
                delta_rotation,
            )

        if delta_angle is not None:
            print("DELTA Rotation Angle:")
            print(
                f"  angle: "
                f"{float(delta_angle):.6f} deg"
            )

        if delta_euler is not None:
            euler = np.asarray(
                delta_euler,
                dtype=np.float64,
            ).reshape(-1)

            if euler.size == 3:
                print("DELTA RPY [deg]:")
                print(f"  roll : {euler[0]:.6f}")
                print(f"  pitch: {euler[1]:.6f}")
                print(f"  yaw  : {euler[2]:.6f}")

    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="SAM2 6DoF Alignment inference"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing *_color.jpg and *_depth.png",
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output directory",
    )

    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not DEFAULT_GT_JSON.is_file():
        raise FileNotFoundError(
            f"GT JSON not found: {DEFAULT_GT_JSON}"
        )

    images = find_images(input_path)

    if not images:
        raise RuntimeError(
            f"No *_color.jpg images found in: {input_path}"
        )

    print("=" * 60)
    print("6DoF ALIGNMENT INFERENCE")
    print("=" * 60)
    print(f"[INPUT]       {input_path}")
    print(f"[OUTPUT]      {output_dir}")
    print(f"[GT]          {DEFAULT_GT_JSON}")
    print(f"[INFO]        Found {len(images)} RGB image(s)")
    print(f"[INFO]        fx = {K[0, 0]:.6f}")
    print(f"[INFO]        fy = {K[1, 1]:.6f}")
    print(f"[INFO]        cx = {K[0, 2]:.6f}")
    print(f"[INFO]        cy = {K[1, 2]:.6f}")
    print("=" * 60)

    print("[INFO] Initializing AlignmentEstimator...")

    estimator = AlignmentEstimator(
        gt_json=DEFAULT_GT_JSON,
    )

    print("[INFO] AlignmentEstimator ready.")
    print("-" * 60)

    success = 0

    for image_path in images:
        if process_image(
            image_path,
            output_dir,
            estimator,
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