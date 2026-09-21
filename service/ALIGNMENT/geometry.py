#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


DEFAULT_DEPTH_SCALE = 0.001
DEFAULT_CENTER_RADIUS = 3
DEFAULT_PLANE_HORIZONTAL_INSET = 0.2
DEFAULT_PLANE_VERTICAL_INSET = 0.3
DEFAULT_PLANE_DEPTH_MAD_SCALE = 4.0
DEFAULT_PLANE_MIN_DEPTH_RATIO = 0.01
DEFAULT_PLANE_MIN_DEPTH_METERS = 0.005
DEFAULT_MIN_PLANE_POINTS = 30


def fit_obb(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    if mask is None:
        return None

    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)

    if len(xs) < 3:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    rect = cv2.minAreaRect(points)
    (cx, cy), (width, height), angle = rect

    if width < height:
        width, height = height, width
        angle += 90.0

    while angle >= 90.0:
        angle -= 180.0

    while angle < -90.0:
        angle += 180.0

    return {
        "center": (float(cx), float(cy)),
        "width": float(width),
        "height": float(height),
        "angle_deg": float(angle),
    }


def pixel_to_3d(
    u: float,
    v: float,
    depth: np.ndarray,
    K: np.ndarray,
) -> Optional[np.ndarray]:
    h, w = depth.shape[:2]
    cx = int(round(u))
    cy = int(round(v))

    if cx < 0 or cx >= w or cy < 0 or cy >= h:
        return None

    radius = DEFAULT_CENTER_RADIUS
    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)

    values = depth[y1:y2, x1:x2].astype(np.float64)
    values = values[values > 0]

    if values.size == 0:
        return None

    z = float(np.median(values)) * DEFAULT_DEPTH_SCALE

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    px = float(K[0, 2])
    py = float(K[1, 2])

    if fx <= 0 or fy <= 0:
        return None

    X = (float(u) - px) * z / fx
    Y = (float(v) - py) * z / fy

    return np.array([X, Y, z], dtype=np.float64)


def get_inset_mask(
    mask: np.ndarray,
    obb: Dict[str, Any],
) -> np.ndarray:
    horizontal_inset = float(np.clip(
        DEFAULT_PLANE_HORIZONTAL_INSET,
        0.0,
        0.49,
    ))
    vertical_inset = float(np.clip(
        DEFAULT_PLANE_VERTICAL_INSET,
        0.0,
        0.49,
    ))

    center = np.asarray(obb["center"], dtype=np.float32)
    width = float(obb["width"])
    height = float(obb["height"])
    angle = float(obb["angle_deg"])

    width *= 1.0 - 2.0 * horizontal_inset
    height *= 1.0 - 2.0 * vertical_inset

    if width <= 1.0 or height <= 1.0:
        return np.zeros_like(mask, dtype=bool)

    box = cv2.boxPoints((
        tuple(center),
        (width, height),
        angle,
    )).astype(np.int32)

    inset_obb_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(inset_obb_mask, box, 1)

    return (mask > 0) & (inset_obb_mask > 0)


def fit_plane(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    obb: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if mask.shape != depth.shape:
        raise ValueError(
            f"Mask/depth resolution mismatch: {mask.shape} vs {depth.shape}"
        )

    K = np.asarray(K, dtype=np.float64)

    if K.shape != (3, 3):
        raise ValueError("Camera intrinsics K must be 3x3")

    obb_mask = get_inset_mask(mask, obb)
    ys, xs = np.where(obb_mask)

    if xs.size == 0:
        return None

    depth_values = depth[ys, xs].astype(np.float64)
    valid = depth_values > 0

    if np.count_nonzero(valid) < DEFAULT_MIN_PLANE_POINTS:
        return None

    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    depth_values = depth_values[valid]

    depth_median = float(np.median(depth_values))
    deviations = np.abs(depth_values - depth_median)
    depth_mad = float(np.median(deviations))

    depth_threshold = max(
        DEFAULT_PLANE_DEPTH_MAD_SCALE * depth_mad,
        depth_median * DEFAULT_PLANE_MIN_DEPTH_RATIO,
        DEFAULT_PLANE_MIN_DEPTH_METERS / DEFAULT_DEPTH_SCALE,
    )

    depth_valid = (
        np.abs(depth_values - depth_median)
        <= depth_threshold
    )

    xs = xs[depth_valid]
    ys = ys[depth_valid]
    depth_values = depth_values[depth_valid]

    if xs.size < DEFAULT_MIN_PLANE_POINTS:
        return None

    plane_mask = np.zeros_like(obb_mask, dtype=bool)
    plane_mask[
        ys.astype(np.int32),
        xs.astype(np.int32),
    ] = True

    valid_ratio = float(
        xs.size / max(np.count_nonzero(valid), 1)
    )

    depth_values *= DEFAULT_DEPTH_SCALE

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    if fx <= 0 or fy <= 0:
        return None

    X = (xs - cx) * depth_values / fx
    Y = (ys - cy) * depth_values / fy
    Z = depth_values

    points = np.column_stack((X, Y, Z))
    center = np.mean(points, axis=0)
    centered = points - center

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = vh[-1]
    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-8:
        return None

    normal /= normal_norm

    if normal[2] < 0:
        normal = -normal

    distances = np.abs(centered @ normal)
    rmse = float(np.sqrt(np.mean(distances ** 2)))

    return {
        "center": center,
        "normal": normal,
        "points": points,
        "point_count": int(points.shape[0]),
        "rmse": rmse,
        "horizontal_inset": DEFAULT_PLANE_HORIZONTAL_INSET,
        "vertical_inset": DEFAULT_PLANE_VERTICAL_INSET,
        "min_points": DEFAULT_MIN_PLANE_POINTS,
        "depth_median": depth_median * DEFAULT_DEPTH_SCALE,
        "depth_mad": depth_mad * DEFAULT_DEPTH_SCALE,
        "depth_threshold": depth_threshold * DEFAULT_DEPTH_SCALE,
        "valid_ratio": valid_ratio,
        "obb_mask": obb_mask,
        "plane_mask": plane_mask,
    }


def fit_orientation(
    plane: Dict[str, Any],
    obb: Dict[str, Any],
    depth: np.ndarray,
    K: np.ndarray,
) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
    points = np.asarray(
        plane.get("points"),
        dtype=np.float64,
    )

    if points.ndim != 2 or points.shape[0] < 3:
        return None

    normal = np.asarray(
        plane["normal"],
        dtype=np.float64,
    )

    normal_norm = np.linalg.norm(normal)

    if normal_norm < 1e-8:
        return None

    normal /= normal_norm

    obb_center = np.asarray(
        obb["center"],
        dtype=np.float64,
    )

    width = float(obb["width"])
    height = float(obb["height"])
    angle = float(obb["angle_deg"])

    if width <= 1.0 or height <= 1.0:
        return None

    rect = (
        tuple(obb_center),
        (width, height),
        angle,
    )

    box = cv2.boxPoints(rect).astype(np.float64)

    edge01 = box[1] - box[0]
    edge12 = box[2] - box[1]

    len01 = np.linalg.norm(edge01)
    len12 = np.linalg.norm(edge12)

    if len01 < 1e-8 or len12 < 1e-8:
        return None

    if len01 >= len12:
        image_axis = edge01 / len01
    else:
        image_axis = edge12 / len12

    image_axis_3d = np.array([
        image_axis[0],
        image_axis[1],
        0.0,
    ], dtype=np.float64)

    image_axis_3d -= (
        np.dot(image_axis_3d, normal) * normal
    )

    axis_norm = np.linalg.norm(image_axis_3d)

    if axis_norm < 1e-8:
        return None

    x_axis = image_axis_3d / axis_norm

    projected = (
        points
        - np.outer(points @ normal, normal)
    )

    centered = (
        projected
        - np.mean(projected, axis=0)
    )

    covariance = centered.T @ centered

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    principal_axis = eigenvectors[
        :,
        np.argmax(eigenvalues),
    ]

    principal_axis -= (
        np.dot(principal_axis, normal) * normal
    )

    principal_norm = np.linalg.norm(principal_axis)

    if principal_norm < 1e-8:
        return None

    principal_axis /= principal_norm

    if np.dot(principal_axis, x_axis) < 0:
        principal_axis = -principal_axis

    x_axis = principal_axis

    y_axis = np.cross(normal, x_axis)
    y_norm = np.linalg.norm(y_axis)

    if y_norm < 1e-8:
        return None

    y_axis /= y_norm

    x_axis = np.cross(y_axis, normal)
    x_norm = np.linalg.norm(x_axis)

    if x_norm < 1e-8:
        return None

    x_axis /= x_norm

    R = np.column_stack((
        x_axis,
        y_axis,
        normal,
    ))

    if not np.isfinite(R).all():
        return None

    if not np.allclose(
        R.T @ R,
        np.eye(3),
        atol=1e-3,
    ):
        return None

    if np.linalg.det(R) < 0:
        return None

    p1_2d = (
        obb_center
        - image_axis * width * 0.35
    )

    p2_2d = (
        obb_center
        + image_axis * width * 0.35
    )

    p1 = pixel_to_3d(
        p1_2d[0],
        p1_2d[1],
        depth,
        K,
    )

    p2 = pixel_to_3d(
        p2_2d[0],
        p2_2d[1],
        depth,
        K,
    )

    orientation_debug = {
        "image_axis": image_axis.copy(),
        "p1_2d": p1_2d.copy(),
        "p2_2d": p2_2d.copy(),
        "p1": p1.copy() if p1 is not None else None,
        "p2": p2.copy() if p2 is not None else None,
        "x_axis": x_axis.copy(),
        "y_axis": y_axis.copy(),
        "z_axis": normal.copy(),
        "rotation_matrix": R.copy(),
        "center_radius": DEFAULT_CENTER_RADIUS,
        "plane_point_count": int(points.shape[0]),
    }

    return (
        rotation_matrix_to_quaternion(R),
        orientation_debug,
    )


def rotation_matrix_to_quaternion(
    R: np.ndarray,
) -> Optional[np.ndarray]:
    R = np.asarray(R, dtype=np.float64)

    if R.shape != (3, 3):
        raise ValueError("Rotation matrix must be 3x3")

    trace = np.trace(R)

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(
            1.0
            + R[0, 0]
            - R[1, 1]
            - R[2, 2]
        ) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(
            1.0
            + R[1, 1]
            - R[0, 0]
            - R[2, 2]
        ) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(
            1.0
            + R[2, 2]
            - R[0, 0]
            - R[1, 1]
        ) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array(
        [qx, qy, qz, qw],
        dtype=np.float64,
    )

    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return None

    q /= norm

    if q[3] < 0:
        q = -q

    return q


def calculate_pose(
    mask: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
) -> Optional[Dict[str, Any]]:
    obb = fit_obb(mask)

    if obb is None:
        return None

    location = pixel_to_3d(
        obb["center"][0],
        obb["center"][1],
        depth,
        K,
    )

    if location is None:
        return None

    plane = fit_plane(
        mask,
        depth,
        K,
        obb,
    )

    if plane is None:
        return {
            "location": location.tolist(),
            "rotation": None,
            "obb": obb,
            "plane": None,
            "orientation_debug": None,
        }

    orientation_result = fit_orientation(
        plane,
        obb,
        depth,
        K,
    )

    if orientation_result is None:
        rotation = None
        orientation_debug = None
    else:
        rotation, orientation_debug = orientation_result

    return {
        "location": location.tolist(),
        "rotation": (
            rotation.tolist()
            if rotation is not None
            else None
        ),
        "obb": obb,
        "plane": plane,
        "orientation_debug": orientation_debug,
    }
