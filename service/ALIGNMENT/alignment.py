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

# ---- DINO patch-level filtering ----
PATCH_SIZE = 16
ROI_LONG_SIDE = 448
GT_CROP_MARGIN = 0.1
GT_PATCH_COVERAGE = 0.9
PATCH_SIM_THRESHOLD = 0.75
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MIN_MASK_AREA = 100
MIN_ROI_SIZE = 4


DEBUG_MASK_MODE = "dino"  # "none" | "sam_raw" | "dino" | "plane"

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
    """Patch-level DINOv3 similarity against a GT object prototype."""

    def __init__(
        self,
        gt_image: np.ndarray,
        gt_mask: np.ndarray,
        device: str,
    ):
        self.device = device
        self.dino = self._load_dino()
        self.prototype = self._build_prototype(gt_image, gt_mask)

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
    def _resized_shape(h: int, w: int) -> Tuple[int, int]:
        scale = ROI_LONG_SIDE / float(max(h, w))
        new_h = max(PATCH_SIZE, int(round(h * scale / PATCH_SIZE)) * PATCH_SIZE)
        new_w = max(PATCH_SIZE, int(round(w * scale / PATCH_SIZE)) * PATCH_SIZE)
        return new_h, new_w

    def _patch_tokens(self, bgr: np.ndarray) -> torch.Tensor:
        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError("Image must have shape HxWx3")

        h, w = bgr.shape[:2]
        new_h, new_w = self._resized_shape(h, w)

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        tensor = ((tensor - mean) / std).unsqueeze(0).to(self.device)

        gh, gw = new_h // PATCH_SIZE, new_w // PATCH_SIZE

        with torch.inference_mode():
            tokens = self.dino.forward_features(tensor)["x_norm_patchtokens"][0]

        tokens = tokens.reshape(gh, gw, -1)
        return F.normalize(tokens, dim=-1)

    def _build_prototype(
        self, gt_image: np.ndarray, gt_mask: np.ndarray
    ) -> torch.Tensor:
        if gt_mask.shape != gt_image.shape[:2]:
            gt_mask = cv2.resize(
                gt_mask.astype(np.uint8),
                (gt_image.shape[1], gt_image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        binary = gt_mask > 0
        ys, xs = np.where(binary)
        if len(xs) == 0:
            raise ValueError("GT mask is empty")

        H, W = gt_image.shape[:2]
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        mx = int(round((x2 - x1) * GT_CROP_MARGIN))
        my = int(round((y2 - y1) * GT_CROP_MARGIN))
        x1, x2 = max(0, x1 - mx), min(W, x2 + mx)
        y1, y2 = max(0, y1 - my), min(H, y2 + my)

        crop = gt_image[y1:y2, x1:x2]
        crop_mask = binary[y1:y2, x1:x2].astype(np.float32)

        tokens = self._patch_tokens(crop)
        gh, gw = tokens.shape[:2]

        coverage = cv2.resize(crop_mask, (gw, gh), interpolation=cv2.INTER_AREA)
        selected = coverage >= GT_PATCH_COVERAGE
        if not selected.any():
            selected = coverage >= 0.5
        if not selected.any():
            raise ValueError("No GT patch lies inside the GT mask")

        selected_t = torch.from_numpy(selected).to(tokens.device)
        prototype = F.normalize(tokens[selected_t].mean(dim=0), dim=0)

        print(f"[DINOv3] GT prototype from {int(selected.sum())} patch(es)")
        return prototype

    def similarity_map(self, crop_bgr: np.ndarray) -> np.ndarray:
        tokens = self._patch_tokens(crop_bgr)
        sim = (tokens @ self.prototype).clamp(-1.0, 1.0)
        sim = sim.float().cpu().numpy()

        h, w = crop_bgr.shape[:2]
        return cv2.resize(sim, (w, h), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )


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

        self._last_target_flip = "none"
        self._last_debug: Dict[str, Any] = {}

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

    @staticmethod
    def _clean_object_mask(mask: np.ndarray) -> Optional[np.ndarray]:
        if mask is None:
            return None

        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError("Mask must be single-channel")

        binary = (mask > 0).astype(np.uint8)

        if np.count_nonzero(binary) == 0:
            return None

        kernel = np.ones((3, 3), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        if np.count_nonzero(binary) == 0:
            return None

        return (binary > 0).astype(np.uint8) * 255

    @staticmethod
    def _largest_component(binary: np.ndarray) -> Optional[np.ndarray]:
        binary = (np.asarray(binary) > 0).astype(np.uint8)
        if np.count_nonzero(binary) == 0:
            return None

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if num_labels <= 1:
            return None

        areas = stats[1:, cv2.CC_STAT_AREA]
        best = int(np.argmax(areas)) + 1
        return (labels == best).astype(np.uint8) * 255

    def _roi_similarity(self, image: np.ndarray, box: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in np.round(box)]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        sim_full = np.full((h, w), -1.0, dtype=np.float32)
        if x2 - x1 < MIN_ROI_SIZE or y2 - y1 < MIN_ROI_SIZE:
            return sim_full

        crop = image[y1:y2, x1:x2]
        sim_full[y1:y2, x1:x2] = self.dino.similarity_map(crop)
        return sim_full

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

        sim_map = self._roi_similarity(image, box)
        dino_region = sim_map >= PATCH_SIM_THRESHOLD

        self._last_debug = {
            "sim_map": sim_map,
            "sam_raw": None,
        }

        candidates = []

        for i, candidate in enumerate(masks):
            raw_mask = (candidate > 0).astype(np.uint8) * 255

            try:
                raw_mask = self._clean_object_mask(raw_mask)
            except ValueError:
                continue

            if raw_mask is None:
                continue

            overlap = ((raw_mask > 0) & dino_region).astype(np.uint8) * 255
            overlap = self._clean_object_mask(overlap)
            if overlap is None:
                continue

            final = self._largest_component(overlap)
            if final is None:
                continue

            area = int(np.count_nonzero(final))
            if area < MIN_MASK_AREA:
                continue

            mean_sim = float(sim_map[final > 0].mean())

            candidates.append(
                {
                    "raw": raw_mask,
                    "mask": final,
                    "sam_score": float(scores[i]),
                    "dino_similarity": mean_sim,
                    "area": area,
                }
            )

        if not candidates:
            return None, 0.0, 0.0

        best = max(
            candidates,
            key=lambda item: (item["area"], item["sam_score"]),
        )

        self._last_debug["sam_raw"] = best["raw"]

        return (
            best["mask"],
            best["sam_score"],
            best["dino_similarity"],
        )

    def _calculate_input_pose(
        self,
        mask: np.ndarray,
        K: np.ndarray,
        target_location: np.ndarray,
        target_rotation: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        """
        不使用深度图。目标 pose 定义物件所在平面：
        过 target_location，法向量为目标旋转矩阵的第三列 (局部 Z 轴)。
        mask 像素沿相机射线投影到该平面上，得到输入 pose。
        """
        R_target = quaternion_to_matrix(target_rotation)
        plane_normal = R_target[:, 2]

        pose = calculate_pose(mask, K, target_location, plane_normal)
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
        target_location = np.asarray(target_location, dtype=np.float64).reshape(3)
        target_rotation = normalize_quaternion(target_rotation)
        input_location = np.asarray(input_pose["location"], dtype=np.float64).reshape(3)
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

    def _get_display_mask(
        self, result: Dict[str, Any]
    ) -> Tuple[Optional[np.ndarray], str]:
        """
        Decide which mask to show in debug image.
        If selected mode is invalid or mask missing, return (None, "")
        """
        mode = str(DEBUG_MASK_MODE).lower()

        if mode == "none":
            return None, ""

        if mode == "sam_raw":
            mask = self._last_debug.get("sam_raw")
            return (
                mask.astype(np.uint8) * 255 if mask is not None else None
            ), "SAM Raw"

        if mode == "dino":
            mask = result.get("dino_mask")
            return (mask.astype(np.uint8) if mask is not None else None), "DINO"

        if mode == "plane":
            input_pose = result.get("input_pose")
            plane = input_pose.get("plane") if input_pose is not None else None
            if plane is not None:
                mask = plane.get("plane_mask")
                if mask is not None:
                    return (np.asarray(mask, dtype=np.uint8)), "Plane"
            # fallback to dino when plane mask doesn't exist
            mask = result.get("dino_mask")
            return (mask.astype(np.uint8) if mask is not None else None), "DINO"

        return None, ""

    def _draw_debug(
        self,
        image: np.ndarray,
        polygon: np.ndarray,
        box: np.ndarray,
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

        # Raw SAM2 mask contour (magenta) is always drawn if available
        sam_raw = self._last_debug.get("sam_raw")
        if sam_raw is not None and sam_raw.shape[:2] == debug.shape[:2]:
            contours, _ = cv2.findContours(
                (sam_raw > 0).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(debug, contours, -1, (255, 0, 255), 1)

        # Decide display mask according to DEBUG_MASK_MODE
        display_mask, display_name = self._get_display_mask(result)

        if display_mask is not None:
            display_mask = np.asarray(display_mask, dtype=bool)
            if display_mask.shape[:2] == debug.shape[:2]:
                overlay = debug.copy()
                overlay[display_mask] = (0, 255, 0)
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
            location = input_pose.get("location")
            rotation = input_pose.get("rotation")

            if location is not None and rotation is not None:
                self._draw_axes(
                    debug,
                    K,
                    np.asarray(location),
                    np.asarray(rotation),
                )

        lines = [
            f"SAM2: {result.get('sam_score', 0.0):.3f}",
            f"DINO patch sim (mean): {result.get('dino_similarity', 0.0):.3f} "
            f"(thr {PATCH_SIM_THRESHOLD:.2f})",
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
                (0, 0, 0),
                5,
                cv2.LINE_AA,
            )
            cv2.putText(
                debug,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 24

        return debug

    def predict(
        self,
        image: np.ndarray,
        depth: Optional[np.ndarray],
        K: np.ndarray,
        location: np.ndarray,
        rotation: np.ndarray,
    ) -> Dict[str, Any]:
        # depth 仅为保持接口不变而保留，不再参与计算 (可传 None)。
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape HxWx3")

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

        self._last_debug = {}
        mask, sam_score, dino_similarity = self._predict_mask(image, box)

        dino_mask = mask.copy() if mask is not None else None
        input_obb = self._fit_obb(mask) if mask is not None else None
        input_pose = None
        delta = None
        self._last_target_flip = "none"

        if mask is not None:
            input_pose = self._calculate_input_pose(
                mask,
                K,
                location,
                rotation,
            )
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
            "dino_mask": dino_mask,
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
            result,
            K,
        )

        return result
