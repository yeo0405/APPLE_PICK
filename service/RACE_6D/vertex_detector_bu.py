"""RGB keypoint refinement for RACE-6D pose refinement.

Pipeline:
  1. Project the RACE-6D pose onto the RGB image.
  2. Build the pose-guided ROI for KeypointRefinementNet.
  3. Predict four refined keypoints using KeypointRefinementNet.
  4. Search independently within a local circular neighborhood around
     each ML keypoint using Harris response and RGB gradients.
  5. Validate the four RGB-refined corners as a quadrilateral.
  6. Use RGB corners only when all four points pass validation;
     otherwise fall back to the ML predictions.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from models.keypoint_refinement import KeypointRefinementNet

logger = logging.getLogger(__name__)


# ============================================================
# Global parameters
# ============================================================

# Model
DEFAULT_DEVICE = "cuda"
CHECKPOINT_DEFAULT = "runs/keypoint_refinement/best.pth"

# ML refinement
MAX_DELTA = 0.1
IMAGE_SIZE = 224
ROI_PADDING = 0.25
MIN_ROI_SIZE = 64
MIN_CORRECTION_PX = 0.0

# Geometry validation
MAX_CORNER_SHIFT_PX = 40.0
MIN_FACE_COS = 0.2
MIN_EDGE_LEN_PX = 5.0
MIN_ROI_PX = 30.0

CORNER_TOL_PX = 40.0
AREA_RATIO_MIN = 0.5
AREA_RATIO_MAX = 2.0

MAX_REPROJ_PX = 5.0
MAX_TRANS_MM = 50.0
MAX_ROT_DEG = 15.0

SIGMA_ROT_DEG = 5.0
SIGMA_TRANS_MM = 10.0

# RGB local corner refinement
RGB_REFINE_ENABLED = True
RGB_SEARCH_RADIUS = 6.0

RGB_HARRIS_BLOCK_SIZE = 2
RGB_HARRIS_KSIZE = 3
RGB_HARRIS_K = 0.04

RGB_MIN_HARRIS_RESPONSE = 1e-7

RGB_HARRIS_WEIGHT = 0.75
RGB_GRADIENT_WEIGHT = 0.15
RGB_DISTANCE_WEIGHT = 0.10

RGB_AREA_RATIO_MIN = 0.75
RGB_AREA_RATIO_MAX = 1.25


# ============================================================
# Utility
# ============================================================


def build_condition(
    size_mm,
    scale_mm: float,
) -> np.ndarray:
    sx, sy, sz = [max(float(v), 1e-6) for v in size_mm]

    return np.array(
        [
            sx / scale_mm,
            sy / scale_mm,
            sz / scale_mm,
            math.log(sx / sy),
            math.log(sz / sx),
            math.log(sz / sy),
        ],
        dtype=np.float32,
    )


# ============================================================
# Parameters
# ============================================================


@dataclass
class RefineParams:
    checkpoint: str = CHECKPOINT_DEFAULT
    device: str = DEFAULT_DEVICE

    max_delta: float = MAX_DELTA

    image_size: int = IMAGE_SIZE
    roi_padding: float = ROI_PADDING
    min_roi_size: int = MIN_ROI_SIZE

    min_correction_px: float = MIN_CORRECTION_PX

    max_corner_shift_px: float = MAX_CORNER_SHIFT_PX
    min_face_cos: float = MIN_FACE_COS
    min_edge_len_px: float = MIN_EDGE_LEN_PX
    min_roi_px: float = MIN_ROI_PX

    corner_tol_px: float = CORNER_TOL_PX
    area_ratio_min: float = AREA_RATIO_MIN
    area_ratio_max: float = AREA_RATIO_MAX

    max_reproj_px: float = MAX_REPROJ_PX
    max_trans_mm: float = MAX_TRANS_MM
    max_rot_deg: float = MAX_ROT_DEG

    sigma_rot_deg: float = SIGMA_ROT_DEG
    sigma_trans_mm: float = SIGMA_TRANS_MM

    rgb_refine_enabled: bool = RGB_REFINE_ENABLED
    rgb_search_radius: float = RGB_SEARCH_RADIUS

    rgb_harris_block_size: int = RGB_HARRIS_BLOCK_SIZE
    rgb_harris_ksize: int = RGB_HARRIS_KSIZE
    rgb_harris_k: float = RGB_HARRIS_K

    rgb_min_harris_response: float = RGB_MIN_HARRIS_RESPONSE

    rgb_harris_weight: float = RGB_HARRIS_WEIGHT
    rgb_gradient_weight: float = RGB_GRADIENT_WEIGHT
    rgb_distance_weight: float = RGB_DISTANCE_WEIGHT

    rgb_area_ratio_min: float = RGB_AREA_RATIO_MIN
    rgb_area_ratio_max: float = RGB_AREA_RATIO_MAX


# ============================================================
# Result
# ============================================================


@dataclass
class VertexDetectionResult:
    ok: bool
    info: Dict[str, Any]

    corners_pred: Optional[np.ndarray] = None
    corners_bottom: Optional[np.ndarray] = None
    corners_obs: Optional[np.ndarray] = None

    lines: Optional[List[Optional[Tuple[np.ndarray, np.ndarray, int]]]] = None
    valid_idx: Optional[List[int]] = None

    edge_pts: Optional[List[np.ndarray]] = None
    edge_segs: Optional[List[np.ndarray]] = None

    c3d: Optional[np.ndarray] = None
    K: Optional[np.ndarray] = None
    R0: Optional[np.ndarray] = None
    t0: Optional[np.ndarray] = None

    corners_ml: Optional[np.ndarray] = None
    corners_rgb: Optional[np.ndarray] = None


# ============================================================
# Vertex Detector
# ============================================================


class VertexDetector:

    IMAGENET_MEAN = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    IMAGENET_STD = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    ).reshape(1, 1, 3)

    def __init__(
        self,
        boxes: Dict[int, Dict[str, np.ndarray]],
        params: RefineParams,
    ):
        self.boxes = boxes
        self.p = params

        self.cond_dim: int = 0
        self.size_scale_mm: float = 300.0

        self.device = torch.device(
            params.device
            if params.device == "cpu" or torch.cuda.is_available()
            else "cpu"
        )

        self.model = self._load_model()

    # ========================================================
    # Configuration
    # ========================================================

    @classmethod
    def from_config(
        cls,
        cfg: Optional[dict],
    ) -> Optional["VertexDetector"]:

        if not cfg or not cfg.get("ENABLED", False):
            return None

        boxes: Dict[int, Dict[str, np.ndarray]] = {}

        for label, item in (cfg.get("LABELS") or {}).items():

            size = item["SIZE_M"]

            size_mm = (
                np.array(
                    [
                        size["X"],
                        size["Y"],
                        size["Z"],
                    ],
                    dtype=np.float64,
                )
                * 1000.0
            )

            offset_mm = (
                np.asarray(
                    item.get("ORIGIN_OFFSET_M", [0, 0, 0]),
                    dtype=np.float64,
                )
                * 1000.0
            )

            boxes[int(label)] = {
                "size_mm": size_mm,
                "offset_mm": offset_mm,
            }

        params_cfg = cfg.get("PARAMS") or {}

        checkpoint = params_cfg.get(
            "CHECKPOINT",
            CHECKPOINT_DEFAULT,
        )

        params = RefineParams(
            checkpoint=str(checkpoint),
        )

        logger.info("VertexDetector parameters are configured in vertex_detector.py")

        return cls(
            boxes=boxes,
            params=params,
        )

    # ========================================================
    # Model
    # ========================================================

    def _load_model(self) -> torch.nn.Module:

        checkpoint_path = Path(self.p.checkpoint).expanduser()

        if not checkpoint_path.is_absolute():

            module_dir = Path(__file__).resolve().parent

            candidates = [
                Path.cwd() / checkpoint_path,
                module_dir / checkpoint_path,
                module_dir.parent / checkpoint_path,
            ]

            checkpoint_path = next(
                (path.resolve() for path in candidates if path.is_file()),
                candidates[0].resolve(),
            )

        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Keypoint refinement checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            str(checkpoint_path),
            map_location="cpu",
            weights_only=False,
        )

        state_dict = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )

        ckpt_cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}

        ckpt_cfg = ckpt_cfg or {}

        ckpt_model = ckpt_cfg.get("model", {}) or {}

        if "cond_dim" in ckpt_model:
            self.cond_dim = int(ckpt_model["cond_dim"])

        elif "cond_encoder.0.weight" in state_dict:
            self.cond_dim = int(state_dict["cond_encoder.0.weight"].shape[1])

        else:
            self.cond_dim = 0

        self.size_scale_mm = float(ckpt_model.get("size_scale_mm", 300.0))

        model = KeypointRefinementNet(
            pretrained=False,
            max_delta=float(self.p.max_delta),
            cond_dim=self.cond_dim,
        )

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.to(self.device)
        model.eval()

        logger.info(
            "Loaded KeypointRefinementNet: %s",
            checkpoint_path,
        )

        logger.info(
            "Device: %s",
            self.device,
        )

        logger.info(
            "Model settings: cond_dim=%d size_scale_mm=%.1f "
            "max_delta=%.3f image_size=%d roi_padding=%.2f "
            "min_roi_size=%d",
            self.cond_dim,
            self.size_scale_mm,
            self.p.max_delta,
            self.p.image_size,
            self.p.roi_padding,
            self.p.min_roi_size,
        )

        logger.info(
            "RGB refinement: enabled=%s radius=%.1f " "harris_threshold=%.2e",
            self.p.rgb_refine_enabled,
            self.p.rgb_search_radius,
            self.p.rgb_min_harris_response,
        )

        return model

    # ========================================================
    # Projection
    # ========================================================

    @staticmethod
    def _project(
        pts_cam: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:

        z = np.maximum(
            pts_cam[:, 2],
            1e-3,
        )

        u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]

        return np.stack(
            [u, v],
            axis=1,
        )

    # ========================================================
    # Object geometry
    # ========================================================

    def _face_corners_obj(
        self,
        box: Dict[str, np.ndarray],
        sign: int,
    ) -> np.ndarray:

        hx, hy, hz = box["size_mm"] / 2.0

        corners = np.array(
            [
                [-hx, -hy, sign * hz],
                [+hx, -hy, sign * hz],
                [+hx, +hy, sign * hz],
                [-hx, +hy, sign * hz],
            ],
            dtype=np.float64,
        )

        return corners + box["offset_mm"]

    def _pick_visible_face(
        self,
        box: Dict[str, np.ndarray],
        R: np.ndarray,
        t: np.ndarray,
    ) -> Tuple[int, np.ndarray, float]:

        best = None

        for sign in (+1, -1):

            normal = R @ np.array(
                [0.0, 0.0, float(sign)],
                dtype=np.float64,
            )

            center_obj = box["offset_mm"] + np.array(
                [
                    0.0,
                    0.0,
                    sign * box["size_mm"][2] / 2.0,
                ],
                dtype=np.float64,
            )

            center = R @ center_obj + t

            cos = float(-normal @ center / (np.linalg.norm(center) + 1e-9))

            if best is None or cos > best[2]:
                best = (
                    sign,
                    self._face_corners_obj(box, sign),
                    cos,
                )

        return best

    # ========================================================
    # ROI
    # ========================================================

    def _get_roi(
        self,
        keypoints: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> Tuple[float, float, float]:

        points = keypoints[:, :2].copy()

        points[:, 0] *= image_width
        points[:, 1] *= image_height

        valid = keypoints[:, 2] > 0

        if valid.sum() >= 2:
            points = points[valid]

        x_min = float(points[:, 0].min())
        x_max = float(points[:, 0].max())

        y_min = float(points[:, 1].min())
        y_max = float(points[:, 1].max())

        center_x = (x_min + x_max) * 0.5
        center_y = (y_min + y_max) * 0.5

        span_x = x_max - x_min
        span_y = y_max - y_min

        roi_side = max(
            span_x,
            span_y,
            float(self.p.min_roi_size),
        ) * (1.0 + 2.0 * float(self.p.roi_padding))

        return center_x, center_y, roi_side

    @staticmethod
    def _crop_square(
        image: np.ndarray,
        center_x: float,
        center_y: float,
        side: float,
        output_size: int,
    ) -> Tuple[np.ndarray, float, float, float]:

        side = max(
            1,
            int(round(side)),
        )

        x0 = int(math.floor(center_x - side * 0.5))

        y0 = int(math.floor(center_y - side * 0.5))

        pad = side + 2

        padded = cv2.copyMakeBorder(
            image,
            pad,
            pad,
            pad,
            pad,
            cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )

        crop_x0 = x0 + pad
        crop_y0 = y0 + pad

        crop = padded[
            crop_y0 : crop_y0 + side,
            crop_x0 : crop_x0 + side,
        ]

        if crop.shape[:2] != (side, side):
            raise RuntimeError("ROI crop size mismatch.")

        crop = cv2.resize(
            crop,
            (output_size, output_size),
            interpolation=cv2.INTER_LINEAR,
        )

        return (
            crop,
            float(x0),
            float(y0),
            float(side),
        )

    # ========================================================
    # Heatmaps
    # ========================================================

    @staticmethod
    def _make_gaussian_heatmaps(
        keypoints: np.ndarray,
        resolution: int,
        sigma: float = 4.0,
    ) -> np.ndarray:

        heatmaps = np.zeros(
            (4, resolution, resolution),
            dtype=np.float32,
        )

        for i, (x, y, visibility) in enumerate(keypoints):

            if visibility <= 0:
                continue

            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue

            px = x * (resolution - 1)
            py = y * (resolution - 1)

            radius = max(
                1,
                int(math.ceil(3.0 * sigma)),
            )

            x0 = max(
                0,
                int(math.floor(px)) - radius,
            )

            x1 = min(
                resolution,
                int(math.floor(px)) + radius + 1,
            )

            y0 = max(
                0,
                int(math.floor(py)) - radius,
            )

            y1 = min(
                resolution,
                int(math.floor(py)) + radius + 1,
            )

            yy, xx = np.mgrid[
                y0:y1,
                x0:x1,
            ]

            heatmaps[i, y0:y1, x0:x1] = np.exp(
                -((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma * sigma)
            )

        return heatmaps

    # ========================================================
    # Model input
    # ========================================================

    def _prepare_input(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
    ) -> Tuple[
        torch.Tensor,
        np.ndarray,
        float,
        float,
        float,
    ]:

        height, width = image.shape[:2]

        offset = keypoints.copy()

        center_x, center_y, roi_side = self._get_roi(
            offset,
            width,
            height,
        )

        crop, x0, y0, roi_side = self._crop_square(
            image,
            center_x,
            center_y,
            roi_side,
            int(self.p.image_size),
        )

        crop = cv2.cvtColor(
            crop,
            cv2.COLOR_BGR2RGB,
        )

        scale = float(self.p.image_size) / roi_side

        offset[:, 0] = (offset[:, 0] * width - x0) * scale / self.p.image_size

        offset[:, 1] = (offset[:, 1] * height - y0) * scale / self.p.image_size

        crop = crop.astype(np.float32) / 255.0

        crop = (crop - self.IMAGENET_MEAN) / self.IMAGENET_STD

        input_heatmaps = self._make_gaussian_heatmaps(
            offset,
            int(self.p.image_size),
            sigma=4.0,
        )

        model_input = np.concatenate(
            [
                crop.transpose(2, 0, 1),
                input_heatmaps,
            ],
            axis=0,
        )

        initial_coords = offset[:, :2].astype(np.float32)

        return (
            torch.from_numpy(model_input.astype(np.float32)),
            initial_coords,
            x0,
            y0,
            roi_side,
        )

    # ========================================================
    # RGB feature maps
    # ========================================================

    def _build_rgb_feature_maps(
        self,
        bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        gray = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2GRAY,
        )

        gray = gray.astype(np.float32) / 255.0

        gray = cv2.GaussianBlur(
            gray,
            (3, 3),
            0.8,
        )

        harris = cv2.cornerHarris(
            gray,
            blockSize=max(
                2,
                int(self.p.rgb_harris_block_size),
            ),
            ksize=max(
                3,
                int(self.p.rgb_harris_ksize),
            ),
            k=float(self.p.rgb_harris_k),
        )

        lab = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)

        gradients = []

        for channel in (
            gray,
            lab[:, :, 0] / 255.0,
            lab[:, :, 1] / 255.0,
            lab[:, :, 2] / 255.0,
        ):

            gx = cv2.Sobel(
                channel,
                cv2.CV_32F,
                1,
                0,
                ksize=3,
            )

            gy = cv2.Sobel(
                channel,
                cv2.CV_32F,
                0,
                1,
                ksize=3,
            )

            gradients.append(cv2.magnitude(gx, gy))

        gradient = np.maximum.reduce(gradients)

        return harris, gradient

    # ========================================================
    # RGB local corner search
    # ========================================================

    def _find_rgb_corner(
        self,
        point: np.ndarray,
        harris: np.ndarray,
        gradient: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:

        height, width = harris.shape

        radius = float(self.p.rgb_search_radius)

        cx = float(point[0])
        cy = float(point[1])

        x0 = max(
            0,
            int(math.floor(cx - radius)),
        )

        x1 = min(
            width - 1,
            int(math.ceil(cx + radius)),
        )

        y0 = max(
            0,
            int(math.floor(cy - radius)),
        )

        y1 = min(
            height - 1,
            int(math.ceil(cy + radius)),
        )

        if x1 < x0 or y1 < y0:
            return None, {
                "reason": "search_region_outside_image",
                "candidate_count": 0,
            }

        yy, xx = np.mgrid[
            y0 : y1 + 1,
            x0 : x1 + 1,
        ]

        distance = np.sqrt(
            (xx.astype(np.float32) - cx) ** 2 + (yy.astype(np.float32) - cy) ** 2
        )

        circle_mask = distance <= radius

        local_harris = harris[
            y0 : y1 + 1,
            x0 : x1 + 1,
        ]

        local_gradient = gradient[
            y0 : y1 + 1,
            x0 : x1 + 1,
        ]

        valid = circle_mask & np.isfinite(local_harris) & np.isfinite(local_gradient)

        if not valid.any():
            return None, {
                "reason": "no_valid_candidates",
                "candidate_count": 0,
            }

        positive_harris = np.maximum(
            local_harris,
            0.0,
        )

        max_harris = float(positive_harris[valid].max())

        if not np.isfinite(max_harris) or max_harris < float(
            self.p.rgb_min_harris_response
        ):
            return None, {
                "reason": "harris_response_too_weak",
                "candidate_count": int(valid.sum()),
                "max_harris_response": max_harris,
            }

        harris_norm = np.clip(
            positive_harris
            / max(
                max_harris,
                1e-12,
            ),
            0.0,
            1.0,
        )

        valid_gradients = local_gradient[valid]

        grad_scale = float(
            np.percentile(
                valid_gradients,
                95,
            )
        )

        gradient_norm = np.clip(
            local_gradient
            / max(
                grad_scale,
                1e-6,
            ),
            0.0,
            1.0,
        )

        distance_norm = np.clip(
            distance
            / max(
                radius,
                1e-6,
            ),
            0.0,
            1.0,
        )

        score = (
            float(self.p.rgb_harris_weight) * harris_norm
            + float(self.p.rgb_gradient_weight) * gradient_norm
            + float(self.p.rgb_distance_weight) * (1.0 - distance_norm)
        )

        score[~valid] = -np.inf

        best_y, best_x = np.unravel_index(
            int(np.argmax(score)),
            score.shape,
        )

        candidate = np.array(
            [
                float(x0 + best_x),
                float(y0 + best_y),
            ],
            dtype=np.float64,
        )

        shift = float(np.linalg.norm(candidate - point))

        details = {
            "reason": "candidate_found",
            "candidate_count": int(valid.sum()),
            "max_harris_response": max_harris,
            "selected_harris_response": float(local_harris[best_y, best_x]),
            "selected_gradient": float(local_gradient[best_y, best_x]),
            "selected_score": float(score[best_y, best_x]),
            "shift_px": shift,
        }

        return candidate, details

    # ========================================================
    # Quadrilateral validation
    # ========================================================

    @staticmethod
    def _quadrilateral_valid(
        corners: np.ndarray,
        min_edge_len_px: float,
    ) -> Tuple[bool, Dict[str, Any]]:

        corners = np.asarray(
            corners,
            dtype=np.float32,
        ).reshape(4, 2)

        if not np.isfinite(corners).all():
            return False, {
                "reason": "non_finite_corners",
            }

        edges = (
            np.roll(
                corners,
                -1,
                axis=0,
            )
            - corners
        )

        edge_lengths = np.linalg.norm(
            edges,
            axis=1,
        )

        if (edge_lengths < float(min_edge_len_px)).any():
            return False, {
                "reason": "edge_too_short",
                "edge_lengths_px": edge_lengths.tolist(),
            }

        contour = corners.reshape(
            -1,
            1,
            2,
        )

        if not cv2.isContourConvex(np.round(contour).astype(np.int32)):
            return False, {
                "reason": "non_convex_corners",
                "edge_lengths_px": edge_lengths.tolist(),
            }

        area = abs(float(cv2.contourArea(contour)))

        if area < 1.0:
            return False, {
                "reason": "quadrilateral_area_too_small",
                "area": area,
            }

        return True, {
            "reason": "valid",
            "edge_lengths_px": edge_lengths.tolist(),
            "area": area,
        }

    # ========================================================
    # RGB refinement
    # ========================================================

    def _refine_rgb_corners(
        self,
        bgr: np.ndarray,
        corners_ml: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:

        info: Dict[str, Any] = {
            "enabled": bool(self.p.rgb_refine_enabled),
            "search_radius_px": float(self.p.rgb_search_radius),
            "accepted": False,
            "reason": "not_attempted",
        }

        if not self.p.rgb_refine_enabled:
            info["reason"] = "disabled"
            return None, info

        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            info["reason"] = "invalid_rgb_image"
            return None, info

        corners_ml = np.asarray(
            corners_ml,
            dtype=np.float64,
        ).reshape(4, 2)

        if not np.isfinite(corners_ml).all():
            info["reason"] = "invalid_ml_corners"
            return None, info

        height, width = bgr.shape[:2]

        if (
            (corners_ml[:, 0] < 0).any()
            or (corners_ml[:, 0] >= width).any()
            or (corners_ml[:, 1] < 0).any()
            or (corners_ml[:, 1] >= height).any()
        ):
            info["reason"] = "ml_corner_outside_image"
            return None, info

        harris, gradient = self._build_rgb_feature_maps(bgr)

        candidates = np.zeros(
            (4, 2),
            dtype=np.float64,
        )

        point_details = []
        all_valid = True

        for i in range(4):

            candidate, details = self._find_rgb_corner(
                corners_ml[i],
                harris,
                gradient,
            )

            details["index"] = i

            if candidate is None:
                all_valid = False
                point_details.append(details)
                continue

            candidates[i] = candidate
            point_details.append(details)

        info["points"] = point_details

        if not all_valid:
            info["reason"] = "one_or_more_corners_not_found"
            return candidates, info

        shifts = np.linalg.norm(
            candidates - corners_ml,
            axis=1,
        )

        info["corner_shift_px"] = shifts.tolist()
        info["corner_shift_mean_px"] = float(shifts.mean())
        info["corner_shift_max_px"] = float(shifts.max())

        if (shifts > float(self.p.rgb_search_radius) + 1e-3).any():
            info["reason"] = "rgb_corner_shift_exceeds_radius"
            return candidates, info

        valid, geom = self._quadrilateral_valid(
            candidates,
            self.p.min_edge_len_px,
        )

        info["geometry"] = geom

        if not valid:
            info["reason"] = "rgb_" + str(geom["reason"])
            return candidates, info

        area_ml = abs(
            float(cv2.contourArea(corners_ml.astype(np.float32).reshape(-1, 1, 2)))
        )

        area_rgb = float(geom["area"])

        area_ratio = area_rgb / max(
            area_ml,
            1e-6,
        )

        info["area_ratio"] = float(area_ratio)

        if not (
            float(self.p.rgb_area_ratio_min)
            <= area_ratio
            <= float(self.p.rgb_area_ratio_max)
        ):
            info["reason"] = "rgb_area_ratio_invalid"
            return candidates, info

        info["accepted"] = True
        info["reason"] = "rgb_refinement_accepted"

        return candidates, info

    # ========================================================
    # Lines and segments
    # ========================================================

    @staticmethod
    def _line_from_points(
        p0: np.ndarray,
        p1: np.ndarray,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:

        direction = p1 - p0

        length = float(np.linalg.norm(direction))

        if length < 1e-6:
            return None

        direction = direction / length
        point = 0.5 * (p0 + p1)

        return (
            point.astype(np.float64),
            direction.astype(np.float64),
            2,
        )

    @staticmethod
    def _build_lines(
        corners: np.ndarray,
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray, int]]]:

        lines = []

        for i in range(4):

            j = (i + 1) % 4

            lines.append(
                VertexDetector._line_from_points(
                    corners[i],
                    corners[j],
                )
            )

        return lines

    @staticmethod
    def _build_segments(
        corners: np.ndarray,
    ) -> List[np.ndarray]:

        segments = []

        for i in range(4):

            j = (i + 1) % 4

            segments.append(
                np.stack(
                    [
                        corners[i],
                        corners[j],
                    ]
                ).astype(np.float64)
            )

        return segments

    # ========================================================
    # Detection
    # ========================================================

    def detect(
        self,
        bgr: np.ndarray,
        camera: np.ndarray,
        label: int,
        R: np.ndarray,
        t_mm: np.ndarray,
    ) -> VertexDetectionResult:

        p = self.p

        R0 = np.asarray(
            R,
            dtype=np.float64,
        )

        t0 = np.asarray(
            t_mm,
            dtype=np.float64,
        ).reshape(3)

        K = np.asarray(
            camera,
            dtype=np.float64,
        )

        info: Dict[str, Any] = {
            "ok": False,
            "reason": None,
            "method": "keypoint_refinement_net+rgb_local_corner",
            "model_checkpoint": str(p.checkpoint),
            "device": str(self.device),
            "cond_dim": int(self.cond_dim),
            "rgb_refine_enabled": bool(p.rgb_refine_enabled),
        }

        pred = None
        pred_b = None
        c3d = None

        corners_ml = None
        corners_rgb = None

        def fail(
            reason: str,
            **kwargs,
        ) -> VertexDetectionResult:

            info.update(kwargs)
            info["ok"] = False
            info["reason"] = reason

            return VertexDetectionResult(
                ok=False,
                info=info,
                corners_pred=pred,
                corners_bottom=pred_b,
                corners_obs=None,
                lines=None,
                valid_idx=[],
                edge_pts=[],
                edge_segs=[],
                c3d=c3d,
                K=K,
                R0=R0,
                t0=t0,
                corners_ml=corners_ml,
                corners_rgb=corners_rgb,
            )

        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            return fail("invalid_rgb_image")

        box = self.boxes.get(int(label))

        if box is None:
            return fail("no_size_for_label")

        sign, c3d, face_cos = self._pick_visible_face(
            box,
            R0,
            t0,
        )

        info["face_cos"] = float(face_cos)

        if face_cos < p.min_face_cos:
            return fail("face_grazing")

        c3d_b = self._face_corners_obj(
            box,
            -sign,
        )

        cam_pts = c3d @ R0.T + t0
        cam_b = c3d_b @ R0.T + t0

        if (cam_pts[:, 2] <= 0).any() or (cam_b[:, 2] <= 0).any():
            return fail("behind_camera")

        pred = self._project(
            cam_pts,
            K,
        )

        pred_b = self._project(
            cam_b,
            K,
        )

        info["corners_pred"] = pred.tolist()

        h, w = bgr.shape[:2]

        if not np.isfinite(pred).all() or not np.isfinite(pred_b).all():
            return fail("invalid_projected_corners")

        allp = np.vstack(
            [
                pred,
                pred_b,
            ]
        )

        pad = 6

        x0 = max(
            0,
            int(np.floor(allp[:, 0].min())) - pad,
        )

        y0 = max(
            0,
            int(np.floor(allp[:, 1].min())) - pad,
        )

        x1 = min(
            w,
            int(np.ceil(allp[:, 0].max())) + pad,
        )

        y1 = min(
            h,
            int(np.ceil(allp[:, 1].max())) + pad,
        )

        if x1 <= x0 or y1 <= y0 or (x1 - x0) < p.min_roi_px or (y1 - y0) < p.min_roi_px:
            return fail(
                "roi_too_small",
                roi=[x0, y0, x1, y1],
            )

        offset_keypoints = np.concatenate(
            [
                pred.astype(np.float32),
                np.ones(
                    (4, 1),
                    dtype=np.float32,
                ),
            ],
            axis=1,
        )

        offset_keypoints[:, 0] /= float(w)
        offset_keypoints[:, 1] /= float(h)

        (
            model_input,
            initial_coords,
            roi_x0,
            roi_y0,
            roi_side,
        ) = self._prepare_input(
            bgr,
            offset_keypoints,
        )

        model_input = model_input.unsqueeze(0).to(self.device)

        initial_keypoints = (
            torch.from_numpy(initial_coords).float().unsqueeze(0).to(self.device)
        )

        cond = None

        if self.cond_dim > 0:

            cond = (
                torch.from_numpy(
                    build_condition(
                        box["size_mm"],
                        self.size_scale_mm,
                    )
                )
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

        with torch.inference_mode():

            output = self.model(
                model_input,
                initial_keypoints,
                cond,
            )

        if (
            not isinstance(output, dict)
            or "keypoints" not in output
            or "delta" not in output
        ):
            return fail("invalid_model_output")

        pred_roi = output["keypoints"][0].float().detach().cpu().numpy()

        predicted_delta_roi = output["delta"][0].float().detach().cpu().numpy()

        if pred_roi.shape != (4, 2) or not np.isfinite(pred_roi).all():
            return fail("invalid_predicted_keypoints")

        if (
            predicted_delta_roi.shape != (4, 2)
            or not np.isfinite(predicted_delta_roi).all()
        ):
            return fail("invalid_predicted_delta")

        pred_pixels = np.zeros(
            (4, 2),
            dtype=np.float32,
        )

        pred_pixels[:, 0] = roi_x0 + pred_roi[:, 0] * roi_side

        pred_pixels[:, 1] = roi_y0 + pred_roi[:, 1] * roi_side

        corners_ml = pred_pixels.astype(np.float64)

        raw_shifts = np.linalg.norm(
            corners_ml - pred,
            axis=1,
        )

        info["raw_corner_shift_mean_px"] = float(raw_shifts.mean())

        info["correction_skipped"] = False

        if float(p.min_correction_px) > 0.0 and float(raw_shifts.mean()) < float(
            p.min_correction_px
        ):
            corners_ml = pred.copy()

            info["correction_skipped"] = True

        shifts = np.linalg.norm(
            corners_ml - pred,
            axis=1,
        )

        info["roi"] = {
            "x0": float(roi_x0),
            "y0": float(roi_y0),
            "side": float(roi_side),
            "image_size": int(p.image_size),
        }

        info["initial_keypoints_roi"] = initial_coords.tolist()

        info["predicted_keypoints_roi"] = pred_roi.tolist()

        info["predicted_delta_roi"] = predicted_delta_roi.tolist()

        info["corner_shift_px"] = shifts.tolist()

        info["corner_shift_mean_px"] = float(shifts.mean())

        info["corner_shift_max_px"] = float(shifts.max())

        info["corners_ml"] = corners_ml.tolist()

        if not np.isfinite(corners_ml).all():
            return fail("invalid_corners_ml")

        if float(shifts.max()) > float(p.max_corner_shift_px):
            return fail(
                "corner_shift_too_large",
                max_corner_shift_px=float(p.max_corner_shift_px),
            )

        # Validate ML corners before RGB refinement.
        ml_valid, ml_geom = self._quadrilateral_valid(
            corners_ml,
            p.min_edge_len_px,
        )

        info["ml_geometry"] = ml_geom

        if not ml_valid:
            return fail("ml_" + str(ml_geom["reason"]))

        # RGB local refinement.
        corners_obs = corners_ml.copy()

        rgb_info = {
            "enabled": bool(p.rgb_refine_enabled),
            "accepted": False,
            "reason": "not_attempted",
        }

        rgb_candidates = None

        if p.rgb_refine_enabled:

            rgb_candidates, rgb_info = self._refine_rgb_corners(
                bgr,
                corners_ml,
            )

            if rgb_info.get("accepted", False):

                corners_rgb = rgb_candidates.copy()
                corners_obs = corners_rgb.copy()

            else:

                corners_rgb = None
                corners_obs = corners_ml.copy()

        info["rgb_refine"] = rgb_info

        info["rgb_refine_ok"] = bool(rgb_info.get("accepted", False))

        info["rgb_refine_reason"] = str(rgb_info.get("reason", "unknown"))

        info["rgb_refine_accepted"] = bool(rgb_info.get("accepted", False))

        info["rgb_corner_shift_px"] = rgb_info.get("corner_shift_px")

        info["rgb_corner_shift_mean_px"] = rgb_info.get("corner_shift_mean_px")

        info["corners_rgb"] = corners_rgb.tolist() if corners_rgb is not None else None

        info["corners_obs"] = corners_obs.tolist()

        info["final_method"] = (
            "rgb_local_corner"
            if info["rgb_refine_accepted"]
            else "keypoint_refinement_net"
        )

        # Validate final corners.
        valid, geom = self._quadrilateral_valid(
            corners_obs,
            p.min_edge_len_px,
        )

        info["final_geometry"] = geom

        if not valid:
            return fail("final_" + str(geom["reason"]))

        edge_lengths = np.linalg.norm(
            np.roll(
                corners_obs,
                -1,
                axis=0,
            )
            - corners_obs,
            axis=1,
        )

        info["edge_lengths_px"] = edge_lengths.tolist()

        area_pred = abs(cv2.contourArea(pred.astype(np.float32).reshape(-1, 1, 2)))

        area_obs = abs(
            cv2.contourArea(corners_obs.astype(np.float32).reshape(-1, 1, 2))
        )

        area_ratio = float(area_obs / max(area_pred, 1e-6))

        info["area_ratio"] = area_ratio

        if not (float(p.area_ratio_min) <= area_ratio <= float(p.area_ratio_max)):
            return fail("area_ratio_invalid")

        lines = self._build_lines(corners_obs)

        segments = self._build_segments(corners_obs)

        valid_idx = [i for i, line in enumerate(lines) if line is not None]

        edge_pts = [segments[i].copy() for i in valid_idx]

        edge_segs = [segments[i].copy() for i in valid_idx]

        info["valid_edges"] = valid_idx
        info["n_edges"] = len(valid_idx)
        info["reason"] = "vertices_detected"
        info["ok"] = True

        return VertexDetectionResult(
            ok=True,
            info=info,
            corners_pred=pred,
            corners_bottom=pred_b,
            corners_obs=corners_obs,
            lines=lines,
            valid_idx=valid_idx,
            edge_pts=edge_pts,
            edge_segs=edge_segs,
            c3d=c3d,
            K=K,
            R0=R0,
            t0=t0,
            corners_ml=corners_ml,
            corners_rgb=corners_rgb,
        )

    # ========================================================
    # Debug drawing
    # ========================================================

    @staticmethod
    def draw(
        debug: np.ndarray,
        res: VertexDetectionResult,
    ) -> None:
        """Draw projected, ML and accepted RGB-refined keypoints."""

        def ip(q):
            return (
                int(round(float(q[0]))),
                int(round(float(q[1]))),
            )

        # RACE-6D projected corners: orange.
        if res.corners_pred is not None:

            for i, q in enumerate(res.corners_pred):

                if not np.isfinite(q).all():
                    continue

                cv2.circle(
                    debug,
                    ip(q),
                    5,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug,
                    f"I{i}",
                    (
                        int(q[0]) + 6,
                        int(q[1]) - 6,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )

        # ML corners: yellow.
        if res.corners_ml is not None:

            for i, q in enumerate(res.corners_ml):

                if not np.isfinite(q).all():
                    continue

                cv2.circle(
                    debug,
                    ip(q),
                    5,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug,
                    f"M{i}",
                    (
                        int(q[0]) + 6,
                        int(q[1]) + 12,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        # Accepted RGB corners: green.
        if res.corners_rgb is not None:

            for i, q in enumerate(res.corners_rgb):

                if not np.isfinite(q).all():
                    continue

                cv2.circle(
                    debug,
                    ip(q),
                    6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug,
                    f"R{i}",
                    (
                        int(q[0]) + 6,
                        int(q[1]) - 6,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

        # Final quadrilateral.
        if res.corners_obs is not None:

            corners = res.corners_obs

            rgb_accepted = res.corners_rgb is not None

            line_color = (0, 255, 0) if rgb_accepted else (255, 0, 255)

            point_color = (0, 255, 0) if rgb_accepted else (0, 255, 255)

            for i in range(4):

                j = (i + 1) % 4

                if (
                    not np.isfinite(corners[i]).all()
                    or not np.isfinite(corners[j]).all()
                ):
                    continue

                cv2.line(
                    debug,
                    ip(corners[i]),
                    ip(corners[j]),
                    line_color,
                    2,
                    cv2.LINE_AA,
                )

            for i, q in enumerate(corners):

                if not np.isfinite(q).all():
                    continue

                cv2.circle(
                    debug,
                    ip(q),
                    4,
                    point_color,
                    -1,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug,
                    f"P{i}",
                    (
                        int(q[0]) + 6,
                        int(q[1]) - 6,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    point_color,
                    1,
                    cv2.LINE_AA,
                )
