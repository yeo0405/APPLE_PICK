#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

MIN_RAY_PLANE_COS = 0.05
AXIS_SAMPLE_RATIO = 0.35

DEFAULT_CENTER_RADIUS = 3
DEFAULT_PLANE_HORIZONTAL_INSET = 0.2
DEFAULT_PLANE_VERTICAL_INSET = 0.3
DEFAULT_MIN_PLANE_POINTS = 30


def fit_obb(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    if mask is None:
        return None

    binary = (mask > 0).astype(np.uint8)
    ys, xs = np.where(binary > 0)

    if len(xs) < 3:
        return None

    points = np.column_stack((xs, ys)).astype(np.float32)
    (cx, cy), (width, height), angle = cv2.minAreaRect(points)

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


def backproject_to_plane(
    us: np.ndarray,
    vs: np.ndarray,
    K: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
    min_cos: float = MIN_RAY_PLANE_COS,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    像素射线与平面求交。
    平面: n · X = n · p；射线: X = t * (x, y, 1)。
    返回 (Nx3 相机系坐标, N 个有效标记)。
    """
    us = np.atleast_1d(np.asarray(us, dtype=np.float64))
    vs = np.atleast_1d(np.asarray(vs, dtype=np.float64))

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    dirs = np.column_stack(((us - cx) / fx, (vs - cy) / fy, np.ones_like(us)))
    dirs_unit = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

    denom = dirs @ plane_normal
    valid = np.abs(dirs_unit @ plane_normal) > min_cos

    t = np.zeros_like(denom)
    t[valid] = float(plane_normal @ plane_point) / denom[valid]
    valid &= t > 0

    return dirs * t[:, None], valid


def rotation_matrix_to_quaternion(R: np.ndarray) -> Optional[np.ndarray]:
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
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return None

    q /= norm
    if q[3] < 0:
        q = -q
    return q


def _build_plane_info(
    mask: np.ndarray,
    K: np.ndarray,
    plane_point: np.ndarray,
    normal: np.ndarray,
) -> Dict[str, Any]:
    """构造与原 fit_plane 字段一致的 plane 字典（深度由目标 pose 给定）。"""
    plane_mask = mask > 0
    ys, xs = np.where(plane_mask)

    pts, ok = backproject_to_plane(
        xs.astype(np.float64),
        ys.astype(np.float64),
        K,
        plane_point,
        normal,
    )
    pts = pts[ok]

    if pts.shape[0] > 0:
        distances = np.abs((pts - plane_point) @ normal)
        rmse = float(np.sqrt(np.mean(distances**2)))  # 理论上约等于 0
    else:
        rmse = 0.0

    plane_depth = float(plane_point[2])
    mask_count = max(int(np.count_nonzero(plane_mask)), 1)

    return {
        "center": plane_point.copy(),
        "normal": normal.copy(),
        "points": pts,
        "point_count": int(pts.shape[0]),
        "rmse": rmse,
        "horizontal_inset": DEFAULT_PLANE_HORIZONTAL_INSET,
        "vertical_inset": DEFAULT_PLANE_VERTICAL_INSET,
        "min_points": DEFAULT_MIN_PLANE_POINTS,
        "depth_median": plane_depth,
        "depth_mad": 0.0,
        "depth_threshold": 0.0,
        "valid_ratio": float(pts.shape[0]) / mask_count,
        "obb_mask": plane_mask.copy(),
        "plane_mask": plane_mask,
        "source": "target_pose",
    }


def calculate_pose(
    mask: np.ndarray,
    K: np.ndarray,
    plane_point: np.ndarray,
    plane_normal: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """
    不使用深度图。平面由目标 pose 给定 (过 plane_point, 法向量 plane_normal)，
    mask 的像素沿相机射线投影到该平面上，得到：
      - location: OBB 中心射线 ∩ 平面
      - rotation: OBB 长轴方向在平面内的 3D 方向 + 平面法向量
    """
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        return None

    plane_point = np.asarray(plane_point, dtype=np.float64).reshape(3)
    normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-8:
        return None
    normal = normal / n_norm
    if normal[2] < 0:  # 法向量朝 +Z，与原逻辑一致
        normal = -normal

    obb = fit_obb(mask)
    if obb is None or obb["width"] <= 1.0 or obb["height"] <= 1.0:
        return None

    # ---- 位置：OBB 中心射线 ∩ 平面 ----
    cu, cv_ = obb["center"]
    pts, ok = backproject_to_plane([cu], [cv_], K, plane_point, normal)
    if not ok[0]:
        return None
    location = pts[0]

    plane = _build_plane_info(mask, K, plane_point, normal)

    # ---- 朝向：OBB 长轴两端点射线 ∩ 平面 ----
    angle = np.deg2rad(obb["angle_deg"])
    image_axis = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    center_2d = np.array([cu, cv_], dtype=np.float64)

    p1_2d = center_2d - image_axis * obb["width"] * AXIS_SAMPLE_RATIO
    p2_2d = center_2d + image_axis * obb["width"] * AXIS_SAMPLE_RATIO

    axis_pts, ok = backproject_to_plane(
        [p1_2d[0], p2_2d[0]],
        [p1_2d[1], p2_2d[1]],
        K,
        plane_point,
        normal,
    )
    if not ok.all():
        return {
            "location": location.tolist(),
            "rotation": None,
            "obb": obb,
            "plane": plane,
            "orientation_debug": None,
        }

    p1, p2 = axis_pts[0], axis_pts[1]

    x_axis = p2 - p1
    x_axis -= np.dot(x_axis, normal) * normal
    x_norm = np.linalg.norm(x_axis)
    if x_norm < 1e-8:
        return {
            "location": location.tolist(),
            "rotation": None,
            "obb": obb,
            "plane": plane,
            "orientation_debug": None,
        }
    x_axis /= x_norm

    y_axis = np.cross(normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    x_axis = np.cross(y_axis, normal)
    x_axis /= np.linalg.norm(x_axis)

    # 与原逻辑一致：翻转 X 和 Z（等价于绕 Y 转 180°，det(R) 仍为 +1）
    x_axis = -x_axis
    z_axis = -normal

    R = np.column_stack((x_axis, y_axis, z_axis))

    if (
        not np.isfinite(R).all()
        or not np.allclose(R.T @ R, np.eye(3), atol=1e-3)
        or np.linalg.det(R) < 0
    ):
        return {
            "location": location.tolist(),
            "rotation": None,
            "obb": obb,
            "plane": plane,
            "orientation_debug": None,
        }

    rotation = rotation_matrix_to_quaternion(R)

    orientation_debug = {
        "image_axis": image_axis.copy(),
        "p1_2d": p1_2d.copy(),
        "p2_2d": p2_2d.copy(),
        "p1": p1.copy(),
        "p2": p2.copy(),
        "x_axis": x_axis.copy(),
        "y_axis": y_axis.copy(),
        "z_axis": z_axis.copy(),
        "rotation_matrix": R.copy(),
        "center_radius": DEFAULT_CENTER_RADIUS,
        "plane_point_count": int(plane["point_count"]),
    }

    return {
        "location": location.tolist(),
        "rotation": (rotation.tolist() if rotation is not None else None),
        "obb": obb,
        "plane": plane,
        "orientation_debug": orientation_debug,
    }
