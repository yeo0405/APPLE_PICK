"""RGB keypoint refinement for RACE-6D pose refinement.

The trained KeypointRefinementNet replaces traditional image-processing
vertex detection.

Pipeline:
  1. Project the RACE-6D pose onto the RGB image.
  2. Build a pose-guided ROI from the four predicted face corners.
  3. Prepare RGB + initial-keypoint heatmaps.
  4. Predict refined keypoints using KeypointRefinementNet.
  5. Return VertexDetectionResult for pose_refine.py.

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
import yaml

from models.keypoint_refinement import KeypointRefinementNet

logger = logging.getLogger(__name__)


@dataclass
class RefineParams:
    # Model
    checkpoint: str = "runs/keypoint_refinement/best.pth"
    device: str = "cuda"
    max_delta: float = 0.1

    # ROI
    image_size: int = 224
    roi_padding: float = 0.25
    min_roi_size: int = 64

    # Sanity
    max_corner_shift_px: float = 40.0
    min_face_cos: float = 0.2
    min_edge_len_px: float = 5.0
    min_roi_px: float = 30.0

    # Pose refinement validation
    corner_tol_px: float = 40.0
    area_ratio_min: float = 0.5
    area_ratio_max: float = 2.0
    max_reproj_px: float = 5.0
    max_trans_mm: float = 50.0
    max_rot_deg: float = 15.0

    # Pose prior
    sigma_rot_deg: float = 5.0
    sigma_trans_mm: float = 10.0


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

        self.device = torch.device(
            params.device
            if params.device == "cpu" or torch.cuda.is_available()
            else "cpu"
        )

        self.model = self._load_model()

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

        valid = {
            "checkpoint",
            "device",
            "max_delta",
            "image_size",
            "roi_padding",
            "min_roi_size",
            "max_corner_shift_px",
            "min_face_cos",
            "min_edge_len_px",
            "min_roi_px",
            "corner_tol_px",
            "area_ratio_min",
            "area_ratio_max",
            "max_reproj_px",
            "max_trans_mm",
            "max_rot_deg",
            "sigma_rot_deg",
            "sigma_trans_mm",
        }

        kwargs = {}

        for key, value in params_cfg.items():
            key = key.lower()

            if key in valid:
                kwargs[key] = value
            else:
                logger.warning(
                    "Unknown RGB_REFINE param ignored: %s",
                    key,
                )

        return cls(
            boxes=boxes,
            params=RefineParams(**kwargs),
        )

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
                f"Keypoint refinement checkpoint not found: " f"{checkpoint_path}"
            )

        model = KeypointRefinementNet(
            pretrained=False,
            max_delta=float(self.p.max_delta),
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

        return model

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

        return crop, float(x0), float(y0), float(side)

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

            yy, xx = np.mgrid[y0:y1, x0:x1]

            heatmaps[i, y0:y1, x0:x1] = np.exp(
                -((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma * sigma)
            )

        return heatmaps

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
            "method": "keypoint_refinement_net",
            "model_checkpoint": str(p.checkpoint),
            "device": str(self.device),
        }

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
            )

        pred = None
        pred_b = None
        c3d = None

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
                roi=[
                    x0,
                    y0,
                    x1,
                    y1,
                ],
            )

        # All four projected corners are used as visible initial points.
        # Visibility is kept at 1 to match the offline inference pipeline.
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

        with torch.inference_mode():
            output = self.model(
                model_input,
                initial_keypoints,
            )

        if (
            not isinstance(output, dict)
            or "keypoints" not in output
            or "delta" not in output
        ):
            return fail("invalid_model_output")

        pred_roi = output["keypoints"][0].detach().cpu().numpy()

        predicted_delta_roi = output["delta"][0].detach().cpu().numpy()

        if pred_roi.shape != (4, 2) or not np.isfinite(pred_roi).all():
            return fail("invalid_predicted_keypoints")

        # Convert normalized ROI coordinates back to image coordinates.
        pred_pixels = np.zeros(
            (4, 2),
            dtype=np.float32,
        )

        pred_pixels[:, 0] = roi_x0 + pred_roi[:, 0] * roi_side

        pred_pixels[:, 1] = roi_y0 + pred_roi[:, 1] * roi_side

        pred_pixels = pred_pixels.astype(np.float64)

        shifts = np.linalg.norm(
            pred_pixels - pred,
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

        info["corners_obs"] = pred_pixels.tolist()

        if not np.isfinite(pred_pixels).all():
            return fail("invalid_corners_obs")

        if float(shifts.max()) > float(p.max_corner_shift_px):
            return fail(
                "corner_shift_too_large",
                max_corner_shift_px=float(p.max_corner_shift_px),
            )

        # Check the refined quadrilateral.
        edge_lengths = np.linalg.norm(
            np.roll(
                pred_pixels,
                -1,
                axis=0,
            )
            - pred_pixels,
            axis=1,
        )

        info["edge_lengths_px"] = edge_lengths.tolist()

        if (edge_lengths < float(p.min_edge_len_px)).any():
            return fail("edge_too_short")

        contour = pred_pixels.astype(np.float32).reshape(
            -1,
            1,
            2,
        )

        if not cv2.isContourConvex(contour.astype(np.int32)):
            return fail("non_convex_corners")

        area_pred = abs(
            cv2.contourArea(
                pred.astype(np.float32).reshape(
                    -1,
                    1,
                    2,
                )
            )
        )

        area_obs = abs(cv2.contourArea(contour))

        area_ratio = float(area_obs / max(area_pred, 1e-6))

        info["area_ratio"] = area_ratio

        if not (0.5 <= area_ratio <= 2.0):
            return fail("area_ratio_invalid")

        # Build compatibility lines from the four predicted corners.
        lines = self._build_lines(pred_pixels)

        segments = self._build_segments(pred_pixels)

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
            corners_obs=pred_pixels,
            lines=lines,
            valid_idx=valid_idx,
            edge_pts=edge_pts,
            edge_segs=edge_segs,
            c3d=c3d,
            K=K,
            R0=R0,
            t0=t0,
        )

    @staticmethod
    def draw(
        debug: np.ndarray,
        res: VertexDetectionResult,
    ) -> None:
        """Draw projected and refined keypoints."""

        def ip(q):
            return (
                int(round(float(q[0]))),
                int(round(float(q[1]))),
            )

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

        if res.corners_obs is not None:
            corners = res.corners_obs

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
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            for i, q in enumerate(corners):
                if not np.isfinite(q).all():
                    continue

                cv2.circle(
                    debug,
                    ip(q),
                    5,
                    (0, 255, 255),
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
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
