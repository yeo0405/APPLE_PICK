#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
GT_DIR = SCRIPT_DIR / "GT"
GT_MASK_DIR = GT_DIR / "gt_masks"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from geometry import calculate_pose

K = np.array([
    [1070.7897900136718, 0.0, 962.3093872070312],
    [0.0, 1070.5269785615235, 512.0372314453125],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def find_gt_image():
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        path = GT_DIR / f"GT{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"GT image not found in {GT_DIR}")


def load_json():
    path = GT_DIR / "gt.json"

    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")

    with open(path, "r", encoding="utf-8") as f:
        return path, json.load(f)


def load_mask(data):
    objects = data.get("objects", [])

    if not objects:
        raise RuntimeError("gt.json contains no objects")

    mask_file = objects[0]["mask"]["file"]
    mask_path = GT_DIR / mask_file

    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if mask is None:
        raise RuntimeError(
            f"Failed to load mask: {mask_path}"
        )

    return mask, mask_path


def load_depth():
    path = GT_DIR / "depth.png"

    depth = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if depth is None:
        raise RuntimeError(
            f"Failed to load depth: {path}"
        )

    if depth.ndim != 2:
        raise RuntimeError(
            f"Depth must be single-channel, got {depth.shape}"
        )

    return depth, path


def save_mask(path: Path, mask):
    if mask is None:
        return

    mask = (
        (np.asarray(mask) > 0)
        .astype(np.uint8)
        * 255
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(path),
        mask,
    ):
        raise RuntimeError(
            f"Failed to save mask: {path}"
        )


def convert_value(value, exclude=None):
    exclude = exclude or set()

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(
        value,
        (np.float16, np.float32, np.float64),
    ):
        return float(value)

    if isinstance(
        value,
        (np.int8, np.int16, np.int32, np.int64),
    ):
        return int(value)

    if isinstance(value, dict):
        return {
            key: convert_value(val, exclude)
            for key, val in value.items()
            if key not in exclude
        }

    if isinstance(value, (list, tuple)):
        return [
            convert_value(v, exclude)
            for v in value
        ]

    return value


def main():
    json_path, data = load_json()
    image_path = find_gt_image()
    mask, mask_path = load_mask(data)
    depth, depth_path = load_depth()

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Failed to load image: {image_path}"
        )

    image_shape = image.shape[:2]

    if mask.shape != image_shape:
        raise RuntimeError(
            f"Mask/image resolution mismatch: "
            f"{mask.shape} vs {image_shape}"
        )

    if depth.shape != image_shape:
        raise RuntimeError(
            f"Depth/image resolution mismatch: "
            f"{depth.shape} vs {image_shape}"
        )

    print("=" * 60)
    print("GENERATE / UPDATE GT POSE")
    print("=" * 60)
    print(f"Image      : {image_path}")
    print(f"Depth      : {depth_path}")
    print(f"Mask       : {mask_path}")
    print("=" * 60)

    result = calculate_pose(
        mask,
        depth,
        K,
    )

    if result is None:
        raise RuntimeError(
            "geometry.calculate_pose() failed"
        )

    location = result.get("location")
    rotation = result.get("rotation")

    if location is None:
        raise RuntimeError(
            "geometry.calculate_pose() returned no location"
        )

    if rotation is None:
        raise RuntimeError(
            "geometry.calculate_pose() returned no rotation. "
            "Check depth, mask and plane fitting."
        )

    location = np.asarray(
        location,
        dtype=np.float64,
    ).reshape(-1)

    rotation = np.asarray(
        rotation,
        dtype=np.float64,
    ).reshape(-1)

    if location.size != 3:
        raise RuntimeError(
            f"Invalid location: {location}"
        )

    if rotation.size != 4:
        raise RuntimeError(
            f"Invalid quaternion: {rotation}"
        )

    objects = data.get("objects")

    if not objects:
        raise RuntimeError(
            "gt.json contains no objects"
        )

    obj = objects[0]

    obj["xyz"] = [
        float(v)
        for v in location
    ]

    obj["quaternion"] = [
        float(v)
        for v in rotation
    ]

    obj["pose"] = {
        "xyz_unit": "meter",
        "quaternion_order": "xyzw",
    }

    if result.get("obb") is not None:
        obj["obb"] = convert_value(
            result["obb"]
        )

    plane = result.get("plane")

    if plane is not None:
        save_mask(
            GT_MASK_DIR / "obb_mask.png",
            plane.get("obb_mask"),
        )

        save_mask(
            GT_MASK_DIR / "plane_mask.png",
            plane.get("plane_mask"),
        )

        obj["plane"] = convert_value(
            plane,
            exclude={
                "points",
                "obb_mask",
                "plane_mask",
            },
        )

    if result.get("orientation_debug") is not None:
        obj["orientation_debug"] = convert_value(
            result["orientation_debug"]
        )

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 60)
    print("GT POSE UPDATED")
    print("=" * 60)
    print(f"JSON       : {json_path}")
    print(f"XYZ [m]    : {obj['xyz']}")
    print(f"Quaternion : {obj['quaternion']}")

    if "obb" in obj:
        print(f"OBB        : {obj['obb']}")

    if plane is not None:
        print(
            f"OBB mask   : "
            f"{GT_MASK_DIR / 'obb_mask.png'}"
        )
        print(
            f"Plane mask : "
            f"{GT_MASK_DIR / 'plane_mask.png'}"
        )
        print(
            f"OBB pixels : "
            f"{int(np.count_nonzero(plane.get('obb_mask')))}"
        )
        print(
            f"Plane pixels: "
            f"{int(np.count_nonzero(plane.get('plane_mask')))}"
        )
        print(
            f"Plane RMSE : "
            f"{plane.get('rmse')}"
        )
        print(
            f"Plane pts  : "
            f"{plane.get('point_count')}"
        )
        print(
            f"Normal     : "
            f"{plane.get('normal')}"
        )
        print(
            f"H inset    : "
            f"{plane.get('horizontal_inset')}"
        )
        print(
            f"V inset    : "
            f"{plane.get('vertical_inset')}"
        )

    if "orientation_debug" in obj:
        debug = obj["orientation_debug"]

        print(
            f"Image axis : "
            f"{debug.get('image_axis')}"
        )
        print(
            f"X axis     : "
            f"{debug.get('x_axis')}"
        )
        print(
            f"Y axis     : "
            f"{debug.get('y_axis')}"
        )
        print(
            f"Z axis     : "
            f"{debug.get('z_axis')}"
        )

    print("=" * 60)


if __name__ == "__main__":
    main()