"""RGB keypoint refinement for RACE-6D pose refinement.

Pipeline:
  1. Project the RACE-6D pose onto the RGB image (candidate "race").
  2. Build a pose-guided ROI from the four projected face corners.
  3. Predict refined corners with KeypointRefinementNet (candidate "ml"),
     conditioned on the box size if the checkpoint was trained with it.
  4. Classical RGB snapping: for every candidate quadrilateral, fit the real
     image edge along each side and move each corner to the intersection of
     its two adjacent edge lines (falls back to the candidate corner when
     unreliable).
  5. Score each snapped quadrilateral by its edge evidence in the image and
     keep the best one (ML is preferred unless RACE is clearly better).
  6. Optionally project the result onto a rigid pose (IPPE).
  7. Return VertexDetectionResult for pose_refine.py.

Model / preprocessing settings (cond_dim, size_scale_mm, max_delta,
image_size, roi_padding, min_roi_size) are read from the checkpoint's saved
config so inference always matches training.
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


def build_condition(size_mm, scale_mm: float) -> np.ndarray:
    """Object-size condition vector. Must be identical to train.py."""
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


@dataclass
class RefineParams:
    # Model (overridden by the checkpoint's saved config when available)
    checkpoint: str = "runs/keypoint_refinement/best.pth"
    device: str = "cuda"
    max_delta: float = 0.3

    # ROI (overridden by the checkpoint's saved config when available)
    image_size: int = 256
    roi_padding: float = 1.0
    min_roi_size: int = 32

    # Skip the network correction when it is smaller than this (px).
    # 0 disables the guard. Small offsets (< ~10px) tend to be made worse.
    min_correction_px: float = 0.0

    # Classical RGB corner snapping
    snap_enabled: bool = True
    snap_search_px: float = 14.0  # search range across each edge
    snap_samples: int = 24  # sample points per edge
    snap_edge_margin: float = 0.12  # skip the ends of each edge
    snap_min_grad: float = 4.0  # min gradient (gray levels / px)
    snap_inlier_tol_px: float = 1.5
    snap_min_inlier: float = 0.5  # min inlier ratio of the line fit
    snap_min_angle_deg: float = 25.0  # min angle between the two lines
    snap_max_shift_px: float = 20.0  # max move from the candidate corner
    snap_allow_partial: bool = True  # one valid line only: project onto it

    # Multi-candidate selection
    snap_multi: bool = True  # also snap from the RACE-6D projected corners
    snap_select_margin: float = 0.05  # race must beat ml by this ratio to be chosen
    snap_rigid: bool = False  # project the result onto a rigid pose (IPPE)
    snap_rigid_max_px: float = 4.0  # skip rigid projection if residual is larger

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
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(
        self,
        boxes: Dict[int, Dict[str, np.ndarray]],
        params: RefineParams,
    ):
        self.boxes = boxes
        self.p = params

        # Filled in by _load_model from the checkpoint
        self.cond_dim: int = 0
        self.size_scale_mm: float = 300.0

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
                np.array([size["X"], size["Y"], size["Z"]], dtype=np.float64) * 1000.0
            )

            offset_mm = (
                np.asarray(item.get("ORIGIN_OFFSET_M", [0, 0, 0]), dtype=np.float64)
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
            "min_correction_px",
            "snap_enabled",
            "snap_search_px",
            "snap_samples",
            "snap_edge_margin",
            "snap_min_grad",
            "snap_inlier_tol_px",
            "snap_min_inlier",
            "snap_min_angle_deg",
            "snap_max_shift_px",
            "snap_allow_partial",
            "snap_multi",
            "snap_select_margin",
            "snap_rigid",
            "snap_rigid_max_px",
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
                logger.warning("Unknown RGB_REFINE param ignored: %s", key)

        return cls(boxes=boxes, params=RefineParams(**kwargs))

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
        ckpt_dataset = ckpt_cfg.get("dataset", {}) or {}

        if "cond_dim" in ckpt_model:
            self.cond_dim = int(ckpt_model["cond_dim"])
        elif "cond_encoder.0.weight" in state_dict:
            self.cond_dim = int(state_dict["cond_encoder.0.weight"].shape[1])
        else:
            self.cond_dim = 0

        self.size_scale_mm = float(ckpt_model.get("size_scale_mm", 300.0))

        if "max_delta" in ckpt_model:
            self.p.max_delta = float(ckpt_model["max_delta"])

        if "image_size" in ckpt_dataset:
            self.p.image_size = int(ckpt_dataset["image_size"])

        if "roi_padding" in ckpt_dataset:
            self.p.roi_padding = float(ckpt_dataset["roi_padding"])

        if "min_roi_size" in ckpt_dataset:
            self.p.min_roi_size = int(ckpt_dataset["min_roi_size"])

        model = KeypointRefinementNet(
            pretrained=False,
            max_delta=float(self.p.max_delta),
            cond_dim=self.cond_dim,
        )

        model.load_state_dict(state_dict, strict=True)

        model.to(self.device)
        model.eval()

        logger.info("Loaded KeypointRefinementNet: %s", checkpoint_path)
        logger.info("Device: %s", self.device)
        logger.info(
            "Model settings: cond_dim=%d size_scale_mm=%.1f max_delta=%.3f "
            "image_size=%d roi_padding=%.2f min_roi_size=%d",
            self.cond_dim,
            self.size_scale_mm,
            self.p.max_delta,
            self.p.image_size,
            self.p.roi_padding,
            self.p.min_roi_size,
        )

        return model

    @staticmethod
    def _project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
        z = np.maximum(pts_cam[:, 2], 1e-3)

        u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]

        return np.stack([u, v], axis=1)

    def _face_corners_obj(self, box: Dict[str, np.ndarray], sign: int) -> np.ndarray:
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
            normal = R @ np.array([0.0, 0.0, float(sign)], dtype=np.float64)

            center_obj = box["offset_mm"] + np.array(
                [0.0, 0.0, sign * box["size_mm"][2] / 2.0], dtype=np.float64
            )

            center = R @ center_obj + t

            cos = float(-normal @ center / (np.linalg.norm(center) + 1e-9))

            if best is None or cos > best[2]:
                best = (sign, self._face_corners_obj(box, sign), cos)

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

        roi_side = max(
            x_max - x_min,
            y_max - y_min,
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
        side = max(1, int(round(side)))

        x0 = int(math.floor(center_x - side * 0.5))
        y0 = int(math.floor(center_y - side * 0.5))

        pad = side + 2

        padded = cv2.copyMakeBorder(
            image, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        crop = padded[y0 + pad : y0 + pad + side, x0 + pad : x0 + pad + side]

        if crop.shape[:2] != (side, side):
            raise RuntimeError("ROI crop size mismatch.")

        crop = cv2.resize(
            crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR
        )

        return crop, float(x0), float(y0), float(side)

    @staticmethod
    def _make_gaussian_heatmaps(
        keypoints: np.ndarray,
        resolution: int,
        sigma: float = 4.0,
    ) -> np.ndarray:
        heatmaps = np.zeros((4, resolution, resolution), dtype=np.float32)

        for i, (x, y, visibility) in enumerate(keypoints):
            if visibility <= 0:
                continue

            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                continue

            px = x * (resolution - 1)
            py = y * (resolution - 1)

            radius = max(1, int(math.ceil(3.0 * sigma)))

            x0 = max(0, int(math.floor(px)) - radius)
            x1 = min(resolution, int(math.floor(px)) + radius + 1)
            y0 = max(0, int(math.floor(py)) - radius)
            y1 = min(resolution, int(math.floor(py)) + radius + 1)

            yy, xx = np.mgrid[y0:y1, x0:x1]

            heatmaps[i, y0:y1, x0:x1] = np.exp(
                -((xx - px) ** 2 + (yy - py) ** 2) / (2.0 * sigma * sigma)
            )

        return heatmaps

    def _prepare_input(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
    ) -> Tuple[torch.Tensor, np.ndarray, float, float, float]:
        height, width = image.shape[:2]

        offset = keypoints.copy()

        center_x, center_y, roi_side = self._get_roi(offset, width, height)

        crop, x0, y0, roi_side = self._crop_square(
            image, center_x, center_y, roi_side, int(self.p.image_size)
        )

        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        scale = float(self.p.image_size) / roi_side

        offset[:, 0] = (offset[:, 0] * width - x0) * scale / self.p.image_size
        offset[:, 1] = (offset[:, 1] * height - y0) * scale / self.p.image_size

        crop = crop.astype(np.float32) / 255.0
        crop = (crop - self.IMAGENET_MEAN) / self.IMAGENET_STD

        input_heatmaps = self._make_gaussian_heatmaps(
            offset, int(self.p.image_size), sigma=4.0
        )

        model_input = np.concatenate([crop.transpose(2, 0, 1), input_heatmaps], axis=0)

        initial_coords = offset[:, :2].astype(np.float32)

        return (
            torch.from_numpy(model_input.astype(np.float32)),
            initial_coords,
            x0,
            y0,
            roi_side,
        )

    # ------------------------------------------------------------------
    # Classical RGB corner snapping
    # ------------------------------------------------------------------

    def _edge_line_fit(self, gx, gy, a, b):
        """Fit a line to the real image edge near the segment a->b.

        Returns (point, direction, inlier_ratio, mean_strength) or None.
        """
        p = self.p

        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        length = float(np.linalg.norm(b - a))

        if length < 20.0:
            return None

        d = (b - a) / length
        n = np.array([-d[1], d[0]], dtype=np.float64)

        N = int(p.snap_samples)
        S = float(p.snap_search_px)
        m = float(p.snap_edge_margin)

        ts = np.linspace(m, 1.0 - m, N)
        base = a[None, :] + ts[:, None] * (b - a)[None, :]  # (N, 2)

        offs = np.arange(-S, S + 1e-6, 0.5, dtype=np.float64)  # (M,)
        M = len(offs)

        pos = base[:, None, :] + offs[None, :, None] * n[None, None, :]  # (N, M, 2)

        map_x = pos[..., 0].astype(np.float32)
        map_y = pos[..., 1].astype(np.float32)

        sx = cv2.remap(
            gx, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        sy = cv2.remap(
            gy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        # Gradient component across the edge (signed)
        proj = sx * n[0] + sy * n[1]

        # Prefer edges close to the predicted line
        prior = np.exp(-0.5 * (offs / (0.6 * S)) ** 2)[None, :]

        score_pos = np.clip(proj, 0, None) * prior
        score_neg = np.clip(-proj, 0, None) * prior

        # One polarity per edge (object vs background); robust to inner texture
        score = (
            score_pos
            if score_pos.max(axis=1).sum() >= score_neg.max(axis=1).sum()
            else score_neg
        )

        rows = np.arange(N)
        idx = score.argmax(axis=1)
        peak = score[rows, idx]

        keep = (peak >= float(p.snap_min_grad)) & (idx > 0) & (idx < M - 1)

        if keep.sum() < max(6, int(0.4 * N)):
            return None

        # Sub-pixel refinement (parabola through the peak)
        y0 = score[rows, np.clip(idx - 1, 0, M - 1)]
        y1 = score[rows, idx]
        y2 = score[rows, np.clip(idx + 1, 0, M - 1)]

        den = y0 - 2.0 * y1 + y2
        delta = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
        delta = np.clip(delta, -1.0, 1.0)

        off = offs[idx] + delta * 0.5

        pts = (base + off[:, None] * n[None, :])[keep].astype(np.float32)

        # Robust fit followed by inlier refit
        vx, vy, fx, fy = cv2.fitLine(
            pts.reshape(-1, 1, 2), cv2.DIST_HUBER, 0, 0.01, 0.01
        ).reshape(-1)

        pt = np.array([fx, fy], dtype=np.float64)
        v = np.array([vx, vy], dtype=np.float64)

        tol = float(p.snap_inlier_tol_px)

        for _ in range(2):
            res = np.abs((pts - pt) @ np.array([-v[1], v[0]]))
            inl = res < tol

            if inl.sum() < 5:
                return None

            vx, vy, fx, fy = cv2.fitLine(
                pts[inl].reshape(-1, 1, 2), cv2.DIST_L2, 0, 0.01, 0.01
            ).reshape(-1)

            pt = np.array([fx, fy], dtype=np.float64)
            v = np.array([vx, vy], dtype=np.float64)

        res = np.abs((pts - pt) @ np.array([-v[1], v[0]]))
        inlier_ratio = float((res < tol).sum()) / float(N)

        return pt, v, inlier_ratio, float(peak[keep].mean())

    @staticmethod
    def _intersect_lines(l1, l2, min_angle_deg):
        p1, d1 = l1[0], l1[1]
        p2, d2 = l2[0], l2[1]

        cross = d1[0] * d2[1] - d1[1] * d2[0]

        if abs(cross) < math.sin(math.radians(min_angle_deg)):
            return None

        w = p2 - p1
        t = (w[0] * d2[1] - w[1] * d2[0]) / cross

        return p1 + t * d1

    def _gradient_field(self, bgr: np.ndarray, corners: np.ndarray):
        """Colour gradient (per pixel, the channel with the strongest gradient)
        over a ROI covering all candidate corners.

        Returns (gx, gy, origin) or None.
        """
        p = self.p
        h, w = bgr.shape[:2]

        pad = int(math.ceil(float(p.snap_search_px))) + 8

        x0 = max(0, int(np.floor(corners[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(corners[:, 1].min())) - pad)
        x1 = min(w, int(np.ceil(corners[:, 0].max())) + pad)
        y1 = min(h, int(np.ceil(corners[:, 1].max())) + pad)

        if x1 - x0 < 16 or y1 - y0 < 16:
            return None

        roi = cv2.GaussianBlur(bgr[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.0)

        gxs = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3, scale=1.0 / 8.0)
        gys = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3, scale=1.0 / 8.0)

        best = (gxs**2 + gys**2).argmax(axis=2)[..., None]

        gx = np.take_along_axis(gxs, best, axis=2)[..., 0]
        gy = np.take_along_axis(gys, best, axis=2)[..., 0]

        return (
            np.ascontiguousarray(gx, dtype=np.float32),
            np.ascontiguousarray(gy, dtype=np.float32),
            np.array([x0, y0], dtype=np.float64),
        )

    def _snap_one(self, gx, gy, origin, corners):
        """Snap one candidate quadrilateral. Returns (corners, info)."""
        p = self.p

        corners = np.asarray(corners, dtype=np.float64)
        local = corners - origin

        snap_info: Dict[str, Any] = {
            "snapped": [False] * 4,
            "reason": [""] * 4,
            "shift_px": [0.0] * 4,
            "inlier_ratio": [None] * 4,
        }

        lines = []

        for i in range(4):
            fit = self._edge_line_fit(gx, gy, local[i], local[(i + 1) % 4])

            if fit is not None and fit[2] < float(p.snap_min_inlier):
                fit = None

            lines.append(fit)

        out = corners.copy()

        for i in range(4):
            l_prev = lines[(i - 1) % 4]
            l_next = lines[i]

            vals = [lines[k][2] for k in ((i - 1) % 4, i) if lines[k] is not None]
            snap_info["inlier_ratio"][i] = float(min(vals)) if vals else None

            if l_prev is not None and l_next is not None:
                q = self._intersect_lines(l_prev, l_next, p.snap_min_angle_deg)

                if q is None:
                    snap_info["reason"][i] = "lines_nearly_parallel"
                    continue

                kind = "intersection"

            elif p.snap_allow_partial and (l_prev is not None or l_next is not None):
                ln = l_prev if l_prev is not None else l_next
                normal = np.array([-ln[1][1], ln[1][0]])
                q = local[i] - float((local[i] - ln[0]) @ normal) * normal
                kind = "partial"

            else:
                snap_info["reason"][i] = "no_valid_edge"
                continue

            shift = float(np.linalg.norm(q - local[i]))

            if not np.isfinite(shift) or shift > float(p.snap_max_shift_px):
                snap_info["reason"][i] = "shift_too_large"
                continue

            out[i] = q + origin
            snap_info["snapped"][i] = True
            snap_info["reason"][i] = kind
            snap_info["shift_px"][i] = shift

        return out, snap_info

    def _quad_energy(self, gx, gy, quad_local, n_samples: int = 40):
        """Edge evidence of a quadrilateral: per edge |mean signed gradient|
        times the fraction of samples with the same polarity. Real boundaries
        keep a consistent sign; inner texture edges flip and cancel out."""
        scores = []

        for i in range(4):
            a = quad_local[i]
            b = quad_local[(i + 1) % 4]

            length = float(np.linalg.norm(b - a))

            if length < 20.0:
                scores.append(0.0)
                continue

            d = (b - a) / length
            n = np.array([-d[1], d[0]])

            ts = np.linspace(0.12, 0.88, n_samples)
            pts = a[None, :] + ts[:, None] * (b - a)[None, :]

            mx = pts[:, 0].astype(np.float32).reshape(1, -1)
            my = pts[:, 1].astype(np.float32).reshape(1, -1)

            sx = cv2.remap(
                gx, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )[0]
            sy = cv2.remap(
                gy, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )[0]

            proj = sx * n[0] + sy * n[1]
            mean = float(proj.mean())

            same = (np.sign(proj) == np.sign(mean)) & (
                np.abs(proj) > 0.5 * float(self.p.snap_min_grad)
            )

            scores.append(abs(mean) * float(same.mean()))

        return float(np.mean(scores)), scores

    @staticmethod
    def _rigid_project(quad, K, c3d):
        """Fit a pose to the 4 coplanar corners (IPPE) and reproject them."""
        try:
            obj = np.ascontiguousarray(c3d, dtype=np.float64)
            img = np.ascontiguousarray(quad, dtype=np.float64).reshape(-1, 1, 2)
            cam = np.ascontiguousarray(K, dtype=np.float64)

            ok, rvec, tvec = cv2.solvePnP(obj, img, cam, None, flags=cv2.SOLVEPNP_IPPE)

            if not ok:
                return None

            proj, _ = cv2.projectPoints(obj, rvec, tvec, cam, None)

            return proj.reshape(-1, 2)
        except cv2.error:
            return None

    def _snap_select(
        self,
        bgr: np.ndarray,
        candidates: Dict[str, np.ndarray],
        K: np.ndarray,
        c3d: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Snap every candidate quad, score it by image evidence, keep the best.

        The first candidate ("ml") is the default; "race" is chosen only if its
        score beats ml by snap_select_margin.
        """
        p = self.p

        names = list(candidates.keys())
        stacked = np.vstack(
            [np.asarray(candidates[n], dtype=np.float64) for n in names]
        )

        info: Dict[str, Any] = {"chosen": names[0], "candidates": {}}

        field = self._gradient_field(bgr, stacked)

        if field is None:
            info["error"] = "roi_too_small"
            return np.asarray(candidates[names[0]], dtype=np.float64).copy(), info

        gx, gy, origin = field

        results: Dict[str, Dict[str, Any]] = {}

        for name in names:
            out, one = self._snap_one(gx, gy, origin, candidates[name])

            n_ok = int(sum(one["snapped"]))
            energy, edge_scores = self._quad_energy(gx, gy, out - origin)
            score = energy * math.sqrt(n_ok / 4.0)

            results[name] = {"corners": out, "score": score}

            info["candidates"][name] = {
                "score": float(score),
                "energy": float(energy),
                "edge_scores": [float(s) for s in edge_scores],
                "n_snapped": n_ok,
                "corners": out.tolist(),
                "snap": one,
            }

        chosen = names[0]

        if "race" in results and "ml" in results:
            if results["race"]["score"] > results["ml"]["score"] * (
                1.0 + float(p.snap_select_margin)
            ):
                chosen = "race"

            info["agreement_px"] = float(
                np.linalg.norm(
                    results["race"]["corners"] - results["ml"]["corners"], axis=1
                ).mean()
            )

        info["chosen"] = chosen

        corners = results[chosen]["corners"]

        if p.snap_rigid:
            q = self._rigid_project(corners, K, c3d)

            if q is not None:
                residual = float(np.linalg.norm(q - corners, axis=1).mean())
                info["rigid_residual_px"] = residual
                info["rigid_applied"] = False

                if residual <= float(p.snap_rigid_max_px):
                    corners = q
                    info["rigid_applied"] = True

        return corners, info

    # ------------------------------------------------------------------

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

        return point.astype(np.float64), direction.astype(np.float64), 2

    @staticmethod
    def _build_lines(
        corners: np.ndarray,
    ) -> List[Optional[Tuple[np.ndarray, np.ndarray, int]]]:
        lines = []

        for i in range(4):
            j = (i + 1) % 4
            lines.append(VertexDetector._line_from_points(corners[i], corners[j]))

        return lines

    @staticmethod
    def _build_segments(corners: np.ndarray) -> List[np.ndarray]:
        segments = []

        for i in range(4):
            j = (i + 1) % 4
            segments.append(np.stack([corners[i], corners[j]]).astype(np.float64))

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

        R0 = np.asarray(R, dtype=np.float64)
        t0 = np.asarray(t_mm, dtype=np.float64).reshape(3)
        K = np.asarray(camera, dtype=np.float64)

        info: Dict[str, Any] = {
            "ok": False,
            "reason": None,
            "method": "keypoint_refinement_net",
            "model_checkpoint": str(p.checkpoint),
            "device": str(self.device),
            "cond_dim": int(self.cond_dim),
        }

        pred = None
        pred_b = None
        c3d = None

        def fail(reason: str, **kwargs) -> VertexDetectionResult:
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

        box = self.boxes.get(int(label))

        if box is None:
            return fail("no_size_for_label")

        sign, c3d, face_cos = self._pick_visible_face(box, R0, t0)

        info["face_cos"] = float(face_cos)

        if face_cos < p.min_face_cos:
            return fail("face_grazing")

        c3d_b = self._face_corners_obj(box, -sign)

        cam_pts = c3d @ R0.T + t0
        cam_b = c3d_b @ R0.T + t0

        if (cam_pts[:, 2] <= 0).any() or (cam_b[:, 2] <= 0).any():
            return fail("behind_camera")

        pred = self._project(cam_pts, K)
        pred_b = self._project(cam_b, K)

        info["corners_pred"] = pred.tolist()

        h, w = bgr.shape[:2]

        if not np.isfinite(pred).all() or not np.isfinite(pred_b).all():
            return fail("invalid_projected_corners")

        allp = np.vstack([pred, pred_b])

        pad = 6

        x0 = max(0, int(np.floor(allp[:, 0].min())) - pad)
        y0 = max(0, int(np.floor(allp[:, 1].min())) - pad)
        x1 = min(w, int(np.ceil(allp[:, 0].max())) + pad)
        y1 = min(h, int(np.ceil(allp[:, 1].max())) + pad)

        if x1 <= x0 or y1 <= y0 or (x1 - x0) < p.min_roi_px or (y1 - y0) < p.min_roi_px:
            return fail("roi_too_small", roi=[x0, y0, x1, y1])

        # All four projected corners are used as visible initial points.
        offset_keypoints = np.concatenate(
            [pred.astype(np.float32), np.ones((4, 1), dtype=np.float32)],
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
        ) = self._prepare_input(bgr, offset_keypoints)

        model_input = model_input.unsqueeze(0).to(self.device)

        initial_keypoints = (
            torch.from_numpy(initial_coords).float().unsqueeze(0).to(self.device)
        )

        cond = None

        if self.cond_dim > 0:
            cond = (
                torch.from_numpy(build_condition(box["size_mm"], self.size_scale_mm))
                .float()
                .unsqueeze(0)
                .to(self.device)
            )

        with torch.inference_mode():
            output = self.model(model_input, initial_keypoints, cond)

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

        # Normalized ROI coordinates -> image pixels.
        pred_pixels = np.zeros((4, 2), dtype=np.float64)
        pred_pixels[:, 0] = roi_x0 + pred_roi[:, 0] * roi_side
        pred_pixels[:, 1] = roi_y0 + pred_roi[:, 1] * roi_side

        raw_shifts = np.linalg.norm(pred_pixels - pred, axis=1)

        info["raw_corner_shift_mean_px"] = float(raw_shifts.mean())

        # Optional guard: tiny corrections are within the network's noise
        # floor, so keep the original projected corners instead.
        info["correction_skipped"] = False

        if float(p.min_correction_px) > 0.0 and float(raw_shifts.mean()) < float(
            p.min_correction_px
        ):
            pred_pixels = pred.copy()
            info["correction_skipped"] = True

        # ---- classical RGB snapping + candidate selection ----
        info["corners_ml"] = pred_pixels.tolist()
        info["snap"] = None

        if p.snap_enabled:
            candidates: Dict[str, np.ndarray] = {"ml": pred_pixels}

            if p.snap_multi:
                candidates["race"] = pred.copy()

            pred_pixels, snap_info = self._snap_select(bgr, candidates, K, c3d)
            info["snap"] = snap_info

        shifts = np.linalg.norm(pred_pixels - pred, axis=1)

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

        edge_lengths = np.linalg.norm(
            np.roll(pred_pixels, -1, axis=0) - pred_pixels, axis=1
        )

        info["edge_lengths_px"] = edge_lengths.tolist()

        if (edge_lengths < float(p.min_edge_len_px)).any():
            return fail("edge_too_short")

        contour = pred_pixels.astype(np.float32).reshape(-1, 1, 2)

        if not cv2.isContourConvex(contour.astype(np.int32)):
            return fail("non_convex_corners")

        area_pred = abs(cv2.contourArea(pred.astype(np.float32).reshape(-1, 1, 2)))
        area_obs = abs(cv2.contourArea(contour))

        area_ratio = float(area_obs / max(area_pred, 1e-6))

        info["area_ratio"] = area_ratio

        if not (p.area_ratio_min <= area_ratio <= p.area_ratio_max):
            return fail("area_ratio_invalid")

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
    def draw(debug: np.ndarray, res: VertexDetectionResult) -> None:
        """Draw RACE projection (orange I), ML output (blue M) and the final
        selected corners (yellow P, magenta outline)."""

        def ip(q):
            return int(round(float(q[0]))), int(round(float(q[1])))

        if res.corners_pred is not None:
            for i, q in enumerate(res.corners_pred):
                if not np.isfinite(q).all():
                    continue

                cv2.circle(debug, ip(q), 5, (0, 165, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    debug,
                    f"I{i}",
                    (int(q[0]) + 6, int(q[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 165, 255),
                    1,
                    cv2.LINE_AA,
                )

        ml = res.info.get("corners_ml") if res.info else None

        if ml is not None:
            for i, q in enumerate(np.asarray(ml, dtype=np.float64)):
                if not np.isfinite(q).all():
                    continue

                cv2.circle(debug, ip(q), 4, (255, 128, 0), 1, cv2.LINE_AA)
                cv2.putText(
                    debug,
                    f"M{i}",
                    (int(q[0]) + 6, int(q[1]) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 128, 0),
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
                    debug, ip(corners[i]), ip(corners[j]), (255, 0, 255), 2, cv2.LINE_AA
                )

            for i, q in enumerate(corners):
                if not np.isfinite(q).all():
                    continue

                cv2.circle(debug, ip(q), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    debug,
                    f"P{i}",
                    (int(q[0]) + 6, int(q[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            snap = res.info.get("snap") if res.info else None

            if snap and "chosen" in snap:
                cv2.putText(
                    debug,
                    f"chosen: {snap['chosen']}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
