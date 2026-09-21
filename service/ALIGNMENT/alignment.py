#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from sam2.sam2_image_predictor import SAM2ImagePredictor

from .geometry import calculate_pose


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"

SAM2_CONFIG = MODEL_DIR / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = MODEL_DIR / "sam2.1_hiera_small.pt"

MIN_MASK_AREA = 100


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Invalid zero quaternion")

    q /= norm

    if q[3] < 0:
        q = -q

    return q


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    return np.array(
        [-q[0], -q[1], -q[2], q[3]],
        dtype=np.float64,
    )


def quaternion_multiply(
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    x, y, z, w = q

    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    return np.degrees(
        np.array([roll, pitch, yaw])
    )


class AlignmentEstimator:
    def __init__(
        self,
        gt_json: Path,
        device: Optional[str] = None,
    ):
        self.gt_json = Path(gt_json)
        self.gt_data = self._load_json(self.gt_json)

        self.gt_image_path = self._resolve_gt_image()
        self.gt_mask = self._load_gt_mask()
        self.gt_depth = self._load_gt_depth()

        self.gt_image = cv2.imread(
            str(self.gt_image_path),
            cv2.IMREAD_COLOR,
        )

        if self.gt_image is None:
            raise RuntimeError(
                f"Failed to load GT image: {self.gt_image_path}"
            )

        self.gt_pose = self._load_gt_pose()
        self.gt_obb = self._load_gt_obb()

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.predictor = self._load_sam2()

        self.gt_height, self.gt_width = self.gt_image.shape[:2]

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_gt_image(self) -> Path:
        for ext in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        ):
            path = self.gt_json.parent / f"GT{ext}"

            if path.is_file():
                return path

        raise FileNotFoundError(
            f"GT image not found in {self.gt_json.parent}"
        )

    def _load_gt_mask(self) -> np.ndarray:
        objects = self.gt_data.get("objects", [])

        if not objects:
            raise RuntimeError(
                "gt.json contains no objects"
            )

        mask_file = objects[0]["mask"]["file"]
        mask_path = self.gt_json.parent / mask_file

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Failed to load GT mask: {mask_path}"
            )

        return mask

    def _load_gt_depth(self) -> np.ndarray:
        path = self.gt_json.parent / "depth.png"

        depth = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            raise RuntimeError(
                f"Failed to load GT depth: {path}"
            )

        if depth.ndim != 2:
            raise RuntimeError(
                f"GT depth must be single-channel: {depth.shape}"
            )

        return depth

    def _load_gt_pose(self) -> Dict[str, Any]:
        obj = self.gt_data["objects"][0]

        if "xyz" not in obj:
            raise RuntimeError(
                "GT object contains no xyz"
            )

        if "quaternion" not in obj:
            raise RuntimeError(
                "GT object contains no quaternion"
            )

        return {
            "location": np.asarray(
                obj["xyz"],
                dtype=np.float64,
            ),
            "rotation": normalize_quaternion(
                np.asarray(
                    obj["quaternion"],
                    dtype=np.float64,
                )
            ),
        }

    def _load_gt_obb(self) -> Optional[Dict[str, Any]]:
        obj = self.gt_data["objects"][0]

        if "obb" not in obj:
            return None

        return obj["obb"]

    def _load_sam2(self):
        if not SAM2_CONFIG.is_file():
            raise FileNotFoundError(
                f"SAM2 config not found: {SAM2_CONFIG}"
            )

        if not SAM2_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}"
            )

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(
            version_base=None,
            config_dir=str(SAM2_CONFIG.parent),
        ):
            cfg = compose(
                config_name=SAM2_CONFIG.stem,
            )

        model = instantiate(cfg.model)

        checkpoint = torch.load(
            SAM2_CHECKPOINT,
            map_location="cpu",
        )

        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        missing, unexpected = model.load_state_dict(
            checkpoint,
            strict=False,
        )

        if missing:
            print(
                f"SAM2 missing keys: {len(missing)}"
            )

        if unexpected:
            print(
                f"SAM2 unexpected keys: {len(unexpected)}"
            )

        model.to(self.device)
        model.eval()

        return SAM2ImagePredictor(model)

    @staticmethod
    def _fit_obb(
        mask: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        if mask is None:
            return None

        binary = (mask > 0).astype(np.uint8)
        ys, xs = np.where(binary > 0)

        if len(xs) < 3:
            return None

        points = np.column_stack(
            (xs, ys)
        ).astype(np.float32)

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
            "center": (
                float(cx),
                float(cy),
            ),
            "width": float(width),
            "height": float(height),
            "angle_deg": float(angle),
        }

    def _get_gt_bbox(self) -> np.ndarray:
        obb = self.gt_obb

        if obb is not None:
            cx, cy = obb["center"]
            width = float(obb["width"])
            height = float(obb["height"])
            angle = float(obb["angle_deg"])

            box = cv2.boxPoints(
                (
                    (float(cx), float(cy)),
                    (width, height),
                    angle,
                )
            )

            return box.astype(np.float32)

        ys, xs = np.where(self.gt_mask > 0)

        if len(xs) == 0:
            raise RuntimeError(
                "GT mask contains no pixels"
            )

        return np.array([
            [xs.min(), ys.min()],
            [xs.max(), ys.min()],
            [xs.max(), ys.max()],
            [xs.min(), ys.max()],
        ], dtype=np.float32)

    def _predict_mask(
        self,
        image: np.ndarray,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        self.predictor.set_image(rgb)

        bbox = self._get_gt_bbox()

        box = np.array([
            bbox[:, 0].min(),
            bbox[:, 1].min(),
            bbox[:, 0].max(),
            bbox[:, 1].max(),
        ], dtype=np.float32)

        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=True,
        )

        if masks is None or len(masks) == 0:
            raise RuntimeError(
                "SAM2 returned no masks"
            )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        ).reshape(-1)

        candidates = []

        for i, candidate in enumerate(masks):
            area = int(
                np.count_nonzero(candidate)
            )

            if area >= MIN_MASK_AREA:
                candidates.append(
                    (
                        float(scores[i]),
                        area,
                        candidate,
                    )
                )

        if not candidates:
            raise RuntimeError(
                "SAM2 returned no valid mask"
            )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        score, _, mask = candidates[0]

        return (
            (mask > 0).astype(np.uint8) * 255,
            score,
            box,
        )

    @staticmethod
    def _calculate_pose(
        mask: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        K = np.asarray(
            K,
            dtype=np.float64,
        )

        if K.shape != (3, 3):
            raise ValueError(
                f"K must have shape (3, 3), got {K.shape}"
            )

        pose = calculate_pose(
            mask,
            depth,
            K,
        )

        if pose is None:
            return None

        plane = pose.get("plane")

        if plane is not None:
            pose["debug_masks"] = {
                "obb_mask": plane.get("obb_mask"),
                "plane_mask": plane.get("plane_mask"),
            }

        pose["geometry_debug"] = pose.get(
            "orientation_debug"
        )

        return pose

    def _calculate_delta(
        self,
        input_pose: Dict[str, Any],
    ) -> Dict[str, Any]:
        gt_location = np.asarray(
            self.gt_pose["location"],
            dtype=np.float64,
        )

        input_location = np.asarray(
            input_pose["location"],
            dtype=np.float64,
        )

        gt_rotation = normalize_quaternion(
            self.gt_pose["rotation"]
        )

        input_rotation = normalize_quaternion(
            input_pose["rotation"]
        )

        delta_location = (
            gt_location - input_location
        )

        delta_rotation = quaternion_multiply(
            gt_rotation,
            quaternion_inverse(input_rotation),
        )

        delta_rotation = normalize_quaternion(
            delta_rotation
        )

        delta_euler = quaternion_to_euler(
            delta_rotation
        )

        delta_angle = 2.0 * np.arccos(
            np.clip(
                abs(delta_rotation[3]),
                -1.0,
                1.0,
            )
        )

        return {
            "location": delta_location,
            "rotation": delta_rotation,
            "rotation_euler_deg": delta_euler,
            "rotation_angle_deg": float(
                np.degrees(delta_angle)
            ),
        }

    @staticmethod
    def _draw_text(
        image: np.ndarray,
        result: Dict[str, Any],
    ) -> None:
        input_pose = result.get("input_pose")
        delta = result.get("delta")

        lines = [
            f"",
        ]

        if input_pose is not None:
            location = np.asarray(
                input_pose["location"]
            ).reshape(3)

            rotation = np.asarray(
                input_pose["rotation"]
            ).reshape(4)

            lines.extend([
                "Input XYZ: "
                f"{location[0]:.4f}, "
                f"{location[1]:.4f}, "
                f"{location[2]:.4f}",
                "Input Q: "
                f"{rotation[0]:.4f}, "
                f"{rotation[1]:.4f}, "
                f"{rotation[2]:.4f}, "
                f"{rotation[3]:.4f}",
            ])

            plane = input_pose.get("plane")

            # if plane is not None:
            #     normal = np.asarray(
            #         plane["normal"]
            #     ).reshape(3)

            #     lines.extend([
            #         "Plane N: "
            #         f"{normal[0]:.3f}, "
            #         f"{normal[1]:.3f}, "
            #         f"{normal[2]:.3f}",
            #         f"Plane pts: "
            #         f"{plane.get('point_count', 0)}",
            #         f"Plane RMSE: "
            #         f"{plane.get('rmse', 0.0):.5f}",
            #         "Inset H/V: "
            #         f"{plane.get('horizontal_inset', 0.0):.2f}/"
            #         f"{plane.get('vertical_inset', 0.0):.2f}",
            #     ])

            geometry_debug = input_pose.get(
                "orientation_debug"
            )

            # if geometry_debug is not None:
            #     x_axis = np.asarray(
            #         geometry_debug["x_axis"]
            #     ).reshape(3)

            #     y_axis = np.asarray(
            #         geometry_debug["y_axis"]
            #     ).reshape(3)

            #     z_axis = np.asarray(
            #         geometry_debug["z_axis"]
            #     ).reshape(3)

            #     lines.extend([
            #         "X axis: "
            #         f"{x_axis[0]:.3f}, "
            #         f"{x_axis[1]:.3f}, "
            #         f"{x_axis[2]:.3f}",
            #         "Y axis: "
            #         f"{y_axis[0]:.3f}, "
            #         f"{y_axis[1]:.3f}, "
            #         f"{y_axis[2]:.3f}",
            #         "Z axis: "
            #         f"{z_axis[0]:.3f}, "
            #         f"{z_axis[1]:.3f}, "
            #         f"{z_axis[2]:.3f}",
            #     ])

        if delta is not None:
            location = np.asarray(
                delta["location"]
            ).reshape(3)

            rotation = np.asarray(
                delta["rotation"]
            ).reshape(4)

            euler = np.asarray(
                delta["rotation_euler_deg"]
            ).reshape(3)

            lines.extend([
                "Delta XYZ: "
                f"{location[0]:.4f}, "
                f"{location[1]:.4f}, "
                f"{location[2]:.4f}",
                "Delta Q: "
                f"{rotation[0]:.4f}, "
                f"{rotation[1]:.4f}, "
                f"{rotation[2]:.4f}, "
                f"{rotation[3]:.4f}",
                # "Delta Angle: "
                # f"{delta['rotation_angle_deg']:.2f} deg",
                "Delta RPY: "
                f"{euler[0]:.2f}, "
                f"{euler[1]:.2f}, "
                f"{euler[2]:.2f}",
            ])

        x = 20
        y = 30

        for line in lines:
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 28

    def _draw_debug(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        input_obb: Optional[Dict[str, Any]],
        result: Dict[str, Any],
    ) -> np.ndarray:
        debug = image.copy()
        overlay = debug.copy()

        overlay[mask > 0] = (
            0.5 * overlay[mask > 0]
            + 0.5 * np.array(
                [0, 255, 0],
                dtype=np.float64,
            )
        ).astype(np.uint8)

        debug = cv2.addWeighted(
            debug,
            0.7,
            overlay,
            0.3,
            0.0,
        )

        if self.gt_obb is not None:
            cx, cy = self.gt_obb["center"]
            width = float(self.gt_obb["width"])
            height = float(self.gt_obb["height"])
            angle = float(self.gt_obb["angle_deg"])

            box = cv2.boxPoints(
                (
                    (float(cx), float(cy)),
                    (width, height),
                    angle,
                )
            ).astype(np.int32)

            cv2.polylines(
                debug,
                [box],
                True,
                (255, 0, 0),
                2,
            )

        if input_obb is not None:
            cx, cy = input_obb["center"]
            width = float(input_obb["width"])
            height = float(input_obb["height"])
            angle = float(input_obb["angle_deg"])

            box = cv2.boxPoints(
                (
                    (float(cx), float(cy)),
                    (width, height),
                    angle,
                )
            ).astype(np.int32)

            cv2.polylines(
                debug,
                [box],
                True,
                (0, 0, 255),
                2,
            )

        self._draw_text(
            debug,
            result,
        )

        return debug

    def predict(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> Dict[str, Any]:
        if image is None:
            raise ValueError(
                "Input image is None"
            )

        if depth is None:
            raise ValueError(
                "Input depth is None"
            )

        if K is None:
            raise ValueError(
                "Camera intrinsic matrix K is None"
            )

        K = np.asarray(
            K,
            dtype=np.float64,
        )

        if K.shape != (3, 3):
            raise ValueError(
                f"K must have shape (3, 3), got {K.shape}"
            )

        if image.shape[:2] != depth.shape[:2]:
            raise ValueError(
                "Input RGB/depth resolution mismatch: "
                f"{image.shape[:2]} vs {depth.shape[:2]}"
            )

        mask, score, bbox = self._predict_mask(
            image
        )

        input_obb = self._fit_obb(mask)

        input_pose = self._calculate_pose(
            mask,
            depth,
            K,
        )

        delta = None

        if (
            input_pose is not None
            and input_pose.get("rotation") is not None
        ):
            delta = self._calculate_delta(
                input_pose
            )

        result = {
            "present": input_obb is not None,
            "mask": mask,
            "bbox": bbox,
            "sam_score": score,
            "gt_pose": self.gt_pose,
            "gt_obb": self.gt_obb,
            "input_obb": input_obb,
            "input_pose": input_pose,
            "delta": delta,
        }

        result["debug_image"] = self._draw_debug(
            image,
            mask,
            input_obb,
            result,
        )

        return result

