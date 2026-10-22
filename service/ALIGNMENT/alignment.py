#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .geometry import calculate_pose

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"
DINOV3_ROOT = PROJECT_ROOT / "service" / "dinov3"
GT_PATH = PROJECT_ROOT / "service" / "ALIGNMENT" / "GT" / "gt.json"

SAM2_CONFIG = MODEL_DIR / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = MODEL_DIR / "sam2.1_hiera_small.pt"
DINO_WEIGHTS = MODEL_DIR / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

IMAGE_SIZE = 224
DINO_THRESHOLD = 0.75
MIN_MASK_AREA = 100
MIN_ROI_SIZE = 4


RESOLVE_TARGET_SYMMETRY = True
TARGET_FLIPS = (
    ("none", np.diag([1.0, 1.0, 1.0])),
    ("flip_x", np.diag([1.0, -1.0, -1.0])),
    ("flip_y", np.diag([-1.0, 1.0, -1.0])),
    ("flip_z", np.diag([-1.0, -1.0, 1.0])),
)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Invalid zero quaternion")
    q = q / norm
    if q[3] < 0:
        q = -q
    return q


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float64)


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = normalize_quaternion(q1)
    x2, y2, z2, w2 = normalize_quaternion(q2)
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quaternion_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quaternion(q)
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.copysign(np.pi / 2.0, sinp) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))

    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                (R[2, 1] - R[1, 2]) / s,
                (R[0, 2] - R[2, 0]) / s,
                (R[1, 0] - R[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        q = np.array(
            [
                0.25 * s,
                (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s,
                (R[2, 1] - R[1, 2]) / s,
            ],
            dtype=np.float64,
        )
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        q = np.array(
            [
                (R[0, 1] + R[1, 0]) / s,
                0.25 * s,
                (R[1, 2] + R[2, 1]) / s,
                (R[0, 2] - R[2, 0]) / s,
            ],
            dtype=np.float64,
        )
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        q = np.array(
            [
                (R[0, 2] + R[2, 0]) / s,
                (R[1, 2] + R[2, 1]) / s,
                0.25 * s,
                (R[1, 0] - R[0, 1]) / s,
            ],
            dtype=np.float64,
        )

    return normalize_quaternion(q)


def rotation_angle_between(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    cos = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


class AlignmentDINO:
    def __init__(
        self,
        gt_image: np.ndarray,
        gt_mask: np.ndarray,
        device: str,
    ):
        self.device = device
        self.dino = self._load_dino()
        self.gt_feature = self._extract_feature(gt_image, gt_mask)

    def _load_dino(self):
        if not DINO_WEIGHTS.is_file():
            raise FileNotFoundError(f"DINOv3 checkpoint not found: {DINO_WEIGHTS}")

        if not DINOV3_ROOT.is_dir():
            raise FileNotFoundError(f"DINOv3 source directory not found: {DINOV3_ROOT}")

        model = torch.hub.load(
            str(DINOV3_ROOT),
            "dinov3_vits16",
            source="local",
            weights=str(DINO_WEIGHTS),
        )
        model = model.to(self.device).eval()
        print(f"[DINOv3] Device: {self.device}")
        print(f"[DINOv3] Weights: {DINO_WEIGHTS}")
        return model

    @staticmethod
    def _crop_mask(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Image must have shape HxWx3")

        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError("Mask must be single-channel")

        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        binary = mask > 0
        ys, xs = np.where(binary)
        if len(xs) == 0:
            raise ValueError("Mask is empty")

        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        crop = image[y1:y2, x1:x2].copy()
        crop_mask = binary[y1:y2, x1:x2]
        crop[~crop_mask] = 0
        return crop

    def _extract_feature(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> torch.Tensor:
        crop = self._crop_mask(image, mask)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = cv2.resize(
            crop,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        tensor = (
            torch.from_numpy(crop)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():
            features = self.dino.forward_features(tensor)["x_norm_patchtokens"][0]
            feature = F.normalize(features.mean(dim=0), dim=0)

        return feature

    def similarity(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        feature = self._extract_feature(image, mask)
        return float(torch.dot(feature, self.gt_feature).clamp(-1.0, 1.0).item())


class AlignmentEstimator:
    def __init__(
        self,
        height: float,
        width: float,
        device: Optional[str] = None,
        roi_scale: float = 1.0,
    ):
        self.gt_json = GT_PATH.resolve()
        self.gt_data = self._load_json(self.gt_json)
        self.gt_image_path = self._resolve_gt_image()
        self.gt_mask = self._load_gt_mask()
        self.gt_depth = self._load_gt_depth()
        self.gt_image = cv2.imread(
            str(self.gt_image_path),
            cv2.IMREAD_COLOR,
        )

        if self.gt_image is None:
            raise RuntimeError(f"Failed to load GT image: {self.gt_image_path}")

        if self.gt_mask.shape != self.gt_image.shape[:2]:
            raise RuntimeError(
                f"GT image/mask resolution mismatch: "
                f"{self.gt_image.shape[:2]} vs {self.gt_mask.shape}"
            )

        self.height = float(height)
        self.width = float(width)
        self.roi_scale = float(roi_scale)

        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive values in meters")
        if self.roi_scale <= 0:
            raise ValueError("roi_scale must be positive")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but CUDA is unavailable: {self.device}"
            )

        self.predictor = self._load_sam2()
        self.dino = AlignmentDINO(
            self.gt_image,
            self.gt_mask,
            self.device,
        )
        self.gt_height, self.gt_width = self.gt_image.shape[:2]

        # 仅用于 debug 图显示：最近一次 delta 选用的目标翻转
        self._last_target_flip = "none"

        print(f"[Alignment] GT JSON: {self.gt_json}")
        print(f"[Alignment] GT image: {self.gt_image_path}")
        print(f"[Alignment] Device: {self.device}")

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Alignment GT JSON not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_gt_image(self) -> Path:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            path = self.gt_json.parent / f"GT{ext}"
            if path.is_file():
                return path

        raise FileNotFoundError(f"GT image not found in {self.gt_json.parent}")

    def _load_gt_mask(self) -> np.ndarray:
        objects = self.gt_data.get("objects", [])
        if not objects:
            raise RuntimeError("gt.json contains no objects")

        mask_info = objects[0].get("mask", {})
        mask_file = mask_info.get("file")
        if not mask_file:
            raise RuntimeError("GT object has no mask.file")

        mask_path = self.gt_json.parent / mask_file
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to load GT mask: {mask_path}")

        return mask

    def _load_gt_depth(self) -> np.ndarray:
        path = self.gt_json.parent / "depth.png"
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise RuntimeError(f"Failed to load GT depth: {path}")
        if depth.ndim != 2:
            raise RuntimeError(f"GT depth must be single-channel: {depth.shape}")

        return depth

    def _load_sam2(self) -> SAM2ImagePredictor:
        if not SAM2_CONFIG.is_file():
            raise FileNotFoundError(f"SAM2 config not found: {SAM2_CONFIG}")
        if not SAM2_CHECKPOINT.is_file():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(
            version_base=None,
            config_dir=str(MODEL_DIR),
        ):
            cfg = compose(
                config_name=SAM2_CONFIG.stem,
                overrides=[
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                    "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                ],
            )
            OmegaConf.resolve(cfg)
            model = instantiate(cfg.model, _recursive_=True)

        _load_checkpoint(model, str(SAM2_CHECKPOINT))
        model = model.to(self.device).eval()

        print(f"[SAM2] Device: {self.device}")
        print(f"[SAM2] Config: {SAM2_CONFIG}")
        print(f"[SAM2] Checkpoint: {SAM2_CHECKPOINT}")

        return SAM2ImagePredictor(model)

    def _roi_from_target(
        self,
        image_shape: Tuple[int, ...],
        K: np.ndarray,
        location: np.ndarray,
        rotation: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image_shape[:2]
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        location = np.asarray(location, dtype=np.float64).reshape(3)
        rotation = normalize_quaternion(rotation)
        R = quaternion_to_matrix(rotation)

        half_w = self.width * self.roi_scale / 2.0
        half_h = self.height * self.roi_scale / 2.0

        local_corners = np.array(
            [
                [-half_w, -half_h, 0.0],
                [half_w, -half_h, 0.0],
                [half_w, half_h, 0.0],
                [-half_w, half_h, 0.0],
            ],
            dtype=np.float64,
        )

        camera_points = (R @ local_corners.T).T + location
        if np.any(camera_points[:, 2] <= 1e-6):
            raise ValueError("Target ROI crosses or lies behind the camera plane")

        projected = (K @ camera_points.T).T
        polygon = projected[:, :2] / projected[:, 2:3]

        if (
            (np.any(polygon[:, 0] < 0) and np.all(polygon[:, 0] < 0))
            or (np.any(polygon[:, 0] >= w) and np.all(polygon[:, 0] >= w))
            or (np.any(polygon[:, 1] < 0) and np.all(polygon[:, 1] < 0))
            or (np.any(polygon[:, 1] >= h) and np.all(polygon[:, 1] >= h))
        ):
            raise ValueError("Projected target ROI is outside the image")

        polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        x1 = int(np.floor(polygon[:, 0].min()))
        y1 = int(np.floor(polygon[:, 1].min()))
        x2 = int(np.ceil(polygon[:, 0].max()))
        y2 = int(np.ceil(polygon[:, 1].max()))

        if x2 - x1 < MIN_ROI_SIZE or y2 - y1 < MIN_ROI_SIZE:
            raise ValueError(f"Projected ROI is too small: {(x1, y1, x2, y2)}")

        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        return polygon.astype(np.int32), box

    def _predict_mask(
        self,
        image: np.ndarray,
        box: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], float, float]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        with torch.inference_mode():
            if self.device.startswith("cuda"):
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                ):
                    masks, scores, _ = self.predictor.predict(
                        box=box,
                        multimask_output=True,
                        return_logits=False,
                    )
            else:
                masks, scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=True,
                    return_logits=False,
                )

        if masks is None or len(masks) == 0:
            return None, 0.0, 0.0

        scores = np.asarray(scores, dtype=np.float64).reshape(-1)
        candidates = []

        for i, candidate in enumerate(masks):
            mask = (candidate > 0).astype(np.uint8) * 255
            area = int(np.count_nonzero(mask))
            if area < MIN_MASK_AREA:
                continue

            try:
                similarity = self.dino.similarity(image, mask)
            except ValueError:
                continue

            candidates.append(
                {
                    "mask": mask,
                    "sam_score": float(scores[i]),
                    "dino_similarity": similarity,
                    "area": area,
                }
            )

        if not candidates:
            return None, 0.0, 0.0

        accepted = [
            item for item in candidates if item["dino_similarity"] >= DINO_THRESHOLD
        ]

        if accepted:
            best = max(
                accepted,
                key=lambda item: (
                    item["sam_score"],
                    item["dino_similarity"],
                ),
            )
            return (
                best["mask"],
                best["sam_score"],
                best["dino_similarity"],
            )

        best = max(
            candidates,
            key=lambda item: (
                item["dino_similarity"],
                item["sam_score"],
            ),
        )
        return None, best["sam_score"], best["dino_similarity"]

    def _calculate_input_pose(
        self,
        mask: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        pose = calculate_pose(mask, depth, K)
        if pose is None:
            return None

        plane = pose.get("plane")
        if plane is not None:
            pose["debug_masks"] = {
                "obb_mask": plane.get("obb_mask"),
                "plane_mask": plane.get("plane_mask"),
            }

        pose["geometry_debug"] = pose.get("orientation_debug")
        return pose

    def _resolve_target_rotation(
        self,
        R_target: np.ndarray,
        R_input: np.ndarray,
    ) -> Tuple[str, np.ndarray]:
        """Pick, among the target's 180-degree-symmetric equivalents, the
        rotation closest to the detected one.

        Returns (flip_name, R_target_resolved).
        """
        if not RESOLVE_TARGET_SYMMETRY:
            return "none", R_target

        best_name = "none"
        best_R = R_target
        best_angle = rotation_angle_between(R_target, R_input)

        for name, S in TARGET_FLIPS[1:]:
            candidate = R_target @ S
            angle = rotation_angle_between(candidate, R_input)
            if angle < best_angle - 1e-9:
                best_name, best_R, best_angle = name, candidate, angle

        return best_name, best_R

    def _calculate_delta(
        self,
        target_location: np.ndarray,
        target_rotation: np.ndarray,
        input_pose: Dict[str, Any],
    ) -> Dict[str, Any]:
        """T_delta = T_target @ inv(T_input), in the camera frame.

        The target rotation is first resolved against the 180-degree
        symmetries of the target rectangle (see RESOLVE_TARGET_SYMMETRY),
        so a target whose axes use a different convention than
        calculate_pose does not produce a spurious ~180 degree delta.
        """
        target_location = np.asarray(
            target_location,
            dtype=np.float64,
        ).reshape(3)
        target_rotation = normalize_quaternion(target_rotation)
        input_location = np.asarray(
            input_pose["location"],
            dtype=np.float64,
        ).reshape(3)
        input_rotation = normalize_quaternion(input_pose["rotation"])

        R_target_raw = quaternion_to_matrix(target_rotation)
        R_input = quaternion_to_matrix(input_rotation)

        flip_name, R_target = self._resolve_target_rotation(R_target_raw, R_input)
        self._last_target_flip = flip_name

        T_target = np.eye(4, dtype=np.float64)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = target_location

        T_input = np.eye(4, dtype=np.float64)
        T_input[:3, :3] = R_input
        T_input[:3, 3] = input_location

        T_delta = np.matmul(T_target, np.linalg.inv(T_input))
        delta_location = T_delta[:3, 3]
        delta_rotation = rotation_matrix_to_quaternion(T_delta[:3, :3])

        angle = 2.0 * np.arccos(np.clip(abs(delta_rotation[3]), 0.0, 1.0))

        return {
            "location": delta_location,
            "rotation": delta_rotation,
            "rotation_euler_deg": quaternion_to_euler(delta_rotation),
            "rotation_angle_deg": float(np.degrees(angle)),
            "transform": T_delta,
        }

    @staticmethod
    def _fit_obb(mask: np.ndarray) -> Optional[Dict[str, Any]]:
        if mask is None:
            return None

        ys, xs = np.where(mask > 0)
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

    @staticmethod
    def _draw_axes(
        image: np.ndarray,
        K: np.ndarray,
        location: np.ndarray,
        rotation: np.ndarray,
        length: float = 0.04,
    ) -> None:
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        location = np.asarray(location, dtype=np.float64).reshape(3)
        R = quaternion_to_matrix(rotation)

        points = np.array(
            [
                location,
                location + R[:, 0] * length,
                location + R[:, 1] * length,
                location + R[:, 2] * length,
            ],
            dtype=np.float64,
        )

        if np.any(points[:, 2] <= 1e-6):
            return

        projected = (K @ points.T).T
        projected = projected[:, :2] / projected[:, 2:3]
        p0, px, py, pz = np.round(projected).astype(int)

        cv2.line(
            image,
            tuple(p0),
            tuple(px),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            tuple(p0),
            tuple(py),
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.line(
            image,
            tuple(p0),
            tuple(pz),
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_debug(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
        box: np.ndarray,
        mask: Optional[np.ndarray],
        result: Dict[str, Any],
        K: np.ndarray,
    ) -> np.ndarray:
        debug = image.copy()

        cv2.polylines(
            debug,
            [polygon.astype(np.int32)],
            True,
            (255, 0, 0),
            2,
        )

        x1, y1, x2, y2 = np.round(box).astype(int)
        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

        if mask is not None:
            overlay = debug.copy()
            overlay[mask > 0] = (0, 255, 0)
            debug = cv2.addWeighted(debug, 0.7, overlay, 0.3, 0.0)

            obb = result.get("input_obb")
            if obb is not None:
                rect = (
                    obb["center"],
                    (obb["width"], obb["height"]),
                    obb["angle_deg"],
                )
                corners = cv2.boxPoints(rect).astype(np.int32)
                cv2.polylines(
                    debug,
                    [corners],
                    True,
                    (0, 0, 255),
                    2,
                )

        input_pose = result.get("input_pose")
        if input_pose is not None:
            self._draw_axes(
                debug,
                K,
                np.asarray(input_pose["location"]),
                np.asarray(input_pose["rotation"]),
            )

        lines = [
            f"SAM2: {result.get('sam_score', 0.0):.3f}",
            f"DINO similarity: {result.get('dino_similarity', 0.0):.3f}",
            f"Present: {result.get('present', False)}",
        ]

        delta = result.get("delta")
        if delta is not None:
            t = np.asarray(delta["location"]).reshape(3)
            q = np.asarray(delta["rotation"]).reshape(4)
            e = np.asarray(delta["rotation_euler_deg"]).reshape(3)
            lines.extend(
                [
                    f"Delta XYZ: {t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f} m",
                    f"Delta Q: {q[0]:.4f}, {q[1]:.4f}, {q[2]:.4f}, {q[3]:.4f}",
                    f"Delta angle: {delta['rotation_angle_deg']:.2f} deg",
                    f"Delta RPY: {e[0]:.2f}, {e[1]:.2f}, {e[2]:.2f}",
                ]
            )
            if self._last_target_flip != "none":
                lines.append(f"Target axes flipped: {self._last_target_flip}")

        y = 28
        for line in lines:
            cv2.putText(
                debug,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

        return debug

    def predict(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        location: np.ndarray,
        rotation: np.ndarray,
    ) -> Dict[str, Any]:
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape HxWx3")
        if depth is None or depth.ndim != 2:
            raise ValueError("depth must be a single-channel image")
        if image.shape[:2] != depth.shape[:2]:
            raise ValueError(
                f"Input RGB/depth resolution mismatch: "
                f"{image.shape[:2]} vs {depth.shape[:2]}"
            )

        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        location = np.asarray(location, dtype=np.float64).reshape(3)
        rotation = normalize_quaternion(rotation)

        if not np.all(np.isfinite(K)):
            raise ValueError("K contains non-finite values")
        if not np.all(np.isfinite(location)):
            raise ValueError("location contains non-finite values")

        polygon, box = self._roi_from_target(
            image.shape,
            K,
            location,
            rotation,
        )

        mask, sam_score, dino_similarity = self._predict_mask(image, box)
        input_obb = self._fit_obb(mask) if mask is not None else None
        input_pose = None
        delta = None
        self._last_target_flip = "none"

        if mask is not None:
            input_pose = self._calculate_input_pose(mask, depth, K)
            if (
                input_pose is not None
                and input_pose.get("rotation") is not None
                and input_pose.get("location") is not None
            ):
                delta = self._calculate_delta(
                    location,
                    rotation,
                    input_pose,
                )

        result = {
            "present": mask is not None and input_obb is not None,
            "mask": mask,
            "bbox": box,
            "roi_polygon": polygon,
            "sam_score": float(sam_score),
            "dino_similarity": float(dino_similarity),
            "target_pose": {
                "location": location.copy(),
                "rotation": rotation.copy(),
            },
            "input_obb": input_obb,
            "input_pose": input_pose,
            "delta": delta,
        }

        result["debug_image"] = self._draw_debug(
            image,
            polygon,
            box,
            mask,
            result,
            K,
        )

        return result
