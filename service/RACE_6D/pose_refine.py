"""RGB-only post-processing for RACE-6D cuboid poses.

Pipeline:
  1. Select the camera-facing Z face and project its four corners.
  2. Collect gradient peaks along each predicted edge.
  3. Generate line hypotheses using vectorised RANSAC-style search.
  4. Evaluate each line using gradient, offset prior and boundary evidence.
  5. Jointly select edge candidates using corner and opposite-edge geometry.
  6. Build four observed corners from line intersections and optional endpoints.
  7. Refine pose using detected line constraints and a RACE pose prior.
  8. Keep the original RACE pose if validation fails.

Translation unit: mm.
Box size and origin offset in config: meters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

logger = logging.getLogger(__name__)

_MIN_PAIR_SEP = 0.15


@dataclass
class RefineParams:
    # Edge search
    band_px: float = 30.0
    weight_ratio: float = 3.0
    samples_per_edge: int = 40
    sample_spacing_px: float = 4.0
    min_samples_per_edge: int = 16
    max_samples_per_edge: int = 100
    edge_margin: float = 0.0

    # Gradient
    min_grad: float = 3.0
    grad_cap: float = 25.0
    peak_min_distance_px: float = 1.0
    max_peaks_per_sample: int = 3
    polarity_penalty: float = 0.15

    # Search direction
    search_angle_deg: float = 8.0
    angle_step_deg: float = 0.5
    support_px: float = 2.5
    min_support_frac: float = 0.35

    # Side wall
    wall_outer_ratio: float = 0.75
    min_out_px: float = 10.0

    # Line fitting
    min_valid_edges: int = 3
    inlier_px: float = 2.0
    min_points: int = 8
    max_edge_angle_deg: float = 10.0
    min_line_coverage: float = 0.35
    max_line_fit_error_px: float = 2.5
    ransac_max_hypotheses: int = 1500

    # Corner construction
    edge_endpoint_as_corner: bool = False

    # Sanity
    max_reproj_px: float = 2.0
    max_trans_mm: float = 30.0
    max_rot_deg: float = 16.0
    area_ratio_min: float = 0.7
    area_ratio_max: float = 1.4
    min_face_cos: float = 0.2
    min_edge_len_px: float = 20.0
    min_roi_px: float = 30.0
    blur_ksize: int = 3

    # Pose prior
    sigma_rot_deg: float = 5.0
    sigma_trans_mm: float = 10.0

    # Offset prior
    offset_prior_center_px: float = 0.0
    offset_prior_sigma_in_px: float = 10.0
    offset_prior_sigma_out_px: float = 16.0

    # Boundary model
    boundary_enabled: bool = True
    boundary_strip_near_px: float = 2.0
    boundary_strip_far_px: float = 8.0
    boundary_interior_shrink: float = 0.6
    boundary_similar_sigma: float = 3.0
    boundary_min_sigma: float = 6.0

    # Boundary score weights
    boundary_weight: float = 0.85
    boundary_inside_weight: float = 0.45
    boundary_contrast_weight: float = 0.25
    boundary_continuity_weight: float = 0.30

    # Lab colour contrast
    boundary_contrast_min: float = 5.0
    boundary_contrast_full: float = 22.0
    boundary_contrast_consistency: float = 8.0
    boundary_min_inside_similarity: float = 0.35
    boundary_max_outside_similarity: float = 0.75

    min_boundary_score: float = -1.0

    # Candidates and joint selection
    candidates_per_edge: int = 8
    joint_candidates: int = 4
    nms_offset_px: float = 3.0
    nms_angle_deg: float = 2.0
    min_candidate_unary: float = 0.01
    corner_tol_px: float = 30.0
    corner_penalty: float = 0.5
    opposite_penalty: float = 0.5
    opposite_ratio_tol: float = 0.3
    opposite_angle_tol_deg: float = 6.0

    # Corner gradient evidence
    corner_evidence_enabled: bool = True
    corner_evidence_radius_px: int = 4
    corner_evidence_weight: float = 0.12
    corner_gradient_min: float = 3.0


@dataclass
class RefineResult:
    ok: bool
    R: np.ndarray
    t_mm: np.ndarray
    info: Dict[str, Any]
    corners_pred: Optional[np.ndarray] = None
    corners_obs: Optional[np.ndarray] = None
    corners_ref: Optional[np.ndarray] = None
    corners_bottom: Optional[np.ndarray] = None
    edge_pts: Optional[List[np.ndarray]] = None
    edge_segs: Optional[List[np.ndarray]] = None


class PoseRefiner:
    def __init__(
        self,
        boxes: Dict[int, Dict[str, np.ndarray]],
        params: RefineParams,
    ):
        self.boxes = boxes
        self.p = params

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["PoseRefiner"]:
        if not cfg or not cfg.get("ENABLED", False):
            return None

        boxes: Dict[int, Dict[str, np.ndarray]] = {}

        for label, item in (cfg.get("LABELS") or {}).items():
            s = item["SIZE_M"]
            size_mm = (
                np.array(
                    [s["X"], s["Y"], s["Z"]],
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

        valid = {f.name for f in fields(RefineParams)}
        kwargs = {}

        for k, v in (cfg.get("PARAMS") or {}).items():
            key = k.lower()
            if key in valid:
                kwargs[key] = v
            else:
                logger.warning("Unknown RGB_REFINE param ignored: %s", k)

        return cls(boxes, RefineParams(**kwargs))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    @staticmethod
    def _project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
        z = np.maximum(pts_cam[:, 2], 1e-3)
        u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
        return np.stack([u, v], axis=1)

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
                [0.0, 0.0, sign * box["size_mm"][2] / 2.0],
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

    @staticmethod
    def _point_line_distance(points, p0, direction) -> np.ndarray:
        return np.abs(
            (points[:, 0] - p0[0]) * direction[1]
            - (points[:, 1] - p0[1]) * direction[0]
        )

    @staticmethod
    def _line_angle_error_deg(direction, expected) -> float:
        cross = abs(direction[0] * expected[1] - direction[1] * expected[0])
        dot = abs(float(direction @ expected))
        return float(np.degrees(np.arctan2(cross, dot + 1e-12)))

    @staticmethod
    def _normalize_direction(direction) -> np.ndarray:
        direction = np.asarray(direction, dtype=np.float64)
        return direction / (np.linalg.norm(direction) + 1e-12)

    @staticmethod
    def _orient(
        direction: np.ndarray,
        expected: np.ndarray,
    ) -> np.ndarray:
        return direction if float(direction @ expected) >= 0.0 else -direction

    @staticmethod
    def _wrap_half_pi(deg: float) -> float:
        return ((deg + 90.0) % 180.0) - 90.0

    @staticmethod
    def _dir_angle_deg(direction: np.ndarray) -> float:
        return float(np.degrees(np.arctan2(direction[1], direction[0])))

    @staticmethod
    def _line_offset(pt, d, mid, normal) -> float:
        cross_nd = normal[0] * d[1] - normal[1] * d[0]

        if abs(cross_nd) < 1e-6:
            cross_nd = 1e-6

        cross_p = (pt[0] - mid[0]) * d[1] - (pt[1] - mid[1]) * d[0]

        return float(cross_p / cross_nd)

    def _offset_prior(self, s):
        p = self.p
        s = np.asarray(s, dtype=np.float64)
        c = float(p.offset_prior_center_px)

        sigma = np.where(
            s > c,
            float(p.offset_prior_sigma_out_px),
            float(p.offset_prior_sigma_in_px),
        )

        sigma = np.maximum(sigma, 1e-3)

        return np.exp(-0.5 * ((s - c) / sigma) ** 2)

    def _sample_edge(self, length: float) -> np.ndarray:
        p = self.p

        n = int(np.ceil(length / max(p.sample_spacing_px, 1.0))) + 1

        n = max(
            n,
            p.min_samples_per_edge,
            p.samples_per_edge,
        )

        n = min(n, p.max_samples_per_edge)

        margin = float(np.clip(p.edge_margin, 0.0, 0.49))

        return np.linspace(
            margin,
            1.0 - margin,
            n,
        )

    # ------------------------------------------------------------------
    # Gradient candidates
    # ------------------------------------------------------------------
    def _collect_gradient_candidates(
        self,
        gray,
        p0,
        p1,
        normal,
        band_in,
        band_out,
        offset,
    ):
        p = self.p

        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        ts = self._sample_edge(length)

        S = len(ts)
        Pm = max(1, int(p.max_peaks_per_sample))

        arrs = {
            "pts": np.full(
                (S, Pm, 2),
                np.nan,
                dtype=np.float64,
            ),
            "valid": np.zeros((S, Pm), dtype=bool),
            "grad": np.zeros((S, Pm), dtype=np.float64),
            "pol": np.zeros((S, Pm), dtype=np.int8),
            "ts": ts,
        }

        info = {
            "gradient_candidate_count": 0,
            "sample_count": int(S),
        }

        base = p0[None, :] + ts[:, None] * direction[None, :]

        step = 0.5

        offsets = np.arange(
            -band_in,
            band_out + step * 0.5,
            step,
            dtype=np.float32,
        )

        if len(offsets) < 3:
            return arrs, info

        h, w = gray.shape

        mx = (base[:, None, 0] + offsets[None, :] * normal[0]).astype(np.float32)

        my = (base[:, None, 1] + offsets[None, :] * normal[1]).astype(np.float32)

        inside = (mx >= 1) & (mx < w - 2) & (my >= 1) & (my < h - 2)

        prof = cv2.remap(
            gray,
            mx,
            my,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        grad = np.gradient(prof, step, axis=1)
        magnitude = np.abs(grad)
        inner = magnitude[:, 1:-1]

        is_peak = (
            (inner >= magnitude[:, :-2])
            & (inner >= magnitude[:, 2:])
            & (inner >= p.min_grad)
            & inside[:, 1:-1]
            & inside[:, :-2]
            & inside[:, 2:]
        )

        min_gap = max(
            1,
            int(round(p.peak_min_distance_px / step)),
        )

        for s in range(S):
            js = np.nonzero(is_peak[s])[0] + 1

            if js.size == 0:
                continue

            order = js[np.argsort(-magnitude[s, js])]
            selected: List[int] = []

            for j in order:
                j = int(j)

                if any(abs(j - o) < min_gap for o in selected):
                    continue

                selected.append(j)

                if len(selected) >= Pm:
                    break

            for m, j in enumerate(selected):
                arrs["pts"][s, m] = base[s] + float(offsets[j]) * normal + offset

                arrs["valid"][s, m] = True
                arrs["grad"][s, m] = float(magnitude[s, j])
                arrs["pol"][s, m] = 1 if grad[s, j] >= 0 else -1

        info["gradient_candidate_count"] = int(arrs["valid"].sum())

        return arrs, info

    # ------------------------------------------------------------------
    # RANSAC-style line hypotheses
    # ------------------------------------------------------------------
    def _rank_hypotheses(
        self,
        arrs,
        expected,
        mid_g,
        normal,
        length,
    ):
        p = self.p

        pts_pad = arrs["pts"]
        valid_pad = arrs["valid"]

        S, Pm = valid_pad.shape

        flat_idx = np.nonzero(valid_pad.ravel())[0]

        if flat_idx.size < max(2, p.min_points):
            return []

        flat_pts = pts_pad.reshape(-1, 2)[flat_idx]
        flat_sample = (flat_idx // Pm).astype(np.int32)

        ii, jj = np.triu_indices(len(flat_pts), k=1)

        keep = flat_sample[ii] != flat_sample[jj]
        ii, jj = ii[keep], jj[keep]

        if ii.size == 0:
            return []

        delta = flat_pts[jj] - flat_pts[ii]
        norm = np.linalg.norm(delta, axis=1)

        keep = norm >= max(_MIN_PAIR_SEP * length, 1e-6)

        ii = ii[keep]
        delta = delta[keep]
        norm = norm[keep]

        if ii.size == 0:
            return []

        d = delta / norm[:, None]

        dot = d @ expected

        d = (
            d
            * np.where(
                dot < 0.0,
                -1.0,
                1.0,
            )[:, None]
        )

        cross = np.abs(d[:, 0] * expected[1] - d[:, 1] * expected[0])

        ang = np.degrees(np.arctan2(cross, np.abs(dot)))

        keep = ang <= p.search_angle_deg

        ii = ii[keep]
        d = d[keep]

        if ii.size == 0:
            return []

        if ii.size > p.ransac_max_hypotheses:
            sel = np.linspace(
                0,
                ii.size - 1,
                int(p.ransac_max_hypotheses),
            ).astype(np.int64)

            ii = ii[sel]
            d = d[sel]

        p0 = flat_pts[ii]

        rel_x = pts_pad[None, :, :, 0] - p0[:, None, None, 0]

        rel_y = pts_pad[None, :, :, 1] - p0[:, None, None, 1]

        dist = np.abs(rel_x * d[:, None, None, 1] - rel_y * d[:, None, None, 0])

        dist = np.where(
            valid_pad[None, :, :],
            dist,
            np.inf,
        )

        dmin = dist.min(axis=2)
        amin = dist.argmin(axis=2)

        inl = dmin <= p.inlier_px
        n_in = inl.sum(axis=1)

        keep = n_in >= p.min_points

        if not keep.any():
            return []

        p0 = p0[keep]
        d = d[keep]
        dmin = dmin[keep]
        amin = amin[keep]
        inl = inl[keep]
        n_in = n_in[keep]

        Hn = len(n_in)

        rows = np.arange(S)[None, :]
        sel_pts = pts_pad[rows, amin]

        t_all = (sel_pts[..., 0] - mid_g[0]) * expected[0] + (
            sel_pts[..., 1] - mid_g[1]
        ) * expected[1]

        tmax = np.where(
            inl,
            t_all,
            -np.inf,
        ).max(axis=1)

        tmin = np.where(
            inl,
            t_all,
            np.inf,
        ).min(axis=1)

        coverage = np.clip(
            (tmax - tmin) / max(length, 1e-6),
            0.0,
            1.0,
        )

        d_sorted = np.sort(
            np.where(inl, dmin, np.inf),
            axis=1,
        )

        resid = d_sorted[
            np.arange(Hn),
            (n_in - 1) // 2,
        ]

        g_sel = arrs["grad"][rows, amin]

        grad_score = (
            np.minimum(
                g_sel / max(p.grad_cap, 1e-6),
                1.0,
            )
            * inl
        ).sum(axis=1) / n_in

        pol_sel = arrs["pol"][rows, amin]

        pos = ((pol_sel > 0) & inl).sum(axis=1)
        pol_cons = np.maximum(pos, n_in - pos) / n_in
        support = n_in / float(S)

        base = (
            support
            * (0.5 + 0.5 * coverage)
            * (0.5 + 0.5 * grad_score)
            * (1.0 - p.polarity_penalty * (1.0 - pol_cons))
            / (1.0 + resid)
        )

        cross_nd = normal[0] * d[:, 1] - normal[1] * d[:, 0]

        cross_nd = np.where(
            np.abs(cross_nd) < 1e-6,
            1e-6,
            cross_nd,
        )

        cross_p = (p0[:, 0] - mid_g[0]) * d[:, 1] - (p0[:, 1] - mid_g[1]) * d[:, 0]

        s_mid = cross_p / cross_nd
        score = base * self._offset_prior(s_mid)

        ang_signed = np.degrees(
            np.arctan2(
                expected[0] * d[:, 1] - expected[1] * d[:, 0],
                d @ expected,
            )
        )

        order = np.argsort(-score)[:300]
        picked: List[int] = []

        for h in order:
            h = int(h)

            if any(
                abs(s_mid[h] - s_mid[q]) < p.nms_offset_px
                and abs(ang_signed[h] - ang_signed[q]) < p.nms_angle_deg
                for q in picked
            ):
                continue

            picked.append(h)

            if len(picked) >= max(
                1,
                int(p.candidates_per_edge),
            ):
                break

        return [
            {
                "pts": sel_pts[h][inl[h]],
                "score": float(score[h]),
                "s_mid": float(s_mid[h]),
                "angle": float(ang_signed[h]),
            }
            for h in picked
        ]

    # ------------------------------------------------------------------
    # Robust line fitting
    # ------------------------------------------------------------------
    def _fit_line(
        self,
        pts: np.ndarray,
        expected_direction: Optional[np.ndarray] = None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
        p = self.p

        if pts is None or len(pts) < p.min_points:
            return None

        pts = np.asarray(
            pts,
            dtype=np.float32,
        ).reshape(-1, 2)

        vx, vy, x0, y0 = cv2.fitLine(
            pts,
            cv2.DIST_HUBER,
            0,
            0.01,
            0.01,
        ).ravel()

        direction = self._normalize_direction(np.array([vx, vy], dtype=np.float64))

        point = np.array(
            [x0, y0],
            dtype=np.float64,
        )

        distances = self._point_line_distance(
            pts.astype(np.float64),
            point,
            direction,
        )

        keep = distances <= p.inlier_px

        if keep.sum() < p.min_points:
            return None

        vx, vy, x0, y0 = cv2.fitLine(
            pts[keep],
            cv2.DIST_L2,
            0,
            0.01,
            0.01,
        ).ravel()

        direction = self._normalize_direction(np.array([vx, vy], dtype=np.float64))

        point = np.array(
            [x0, y0],
            dtype=np.float64,
        )

        distances = self._point_line_distance(
            pts.astype(np.float64),
            point,
            direction,
        )

        keep = distances <= p.inlier_px

        if keep.sum() < p.min_points:
            return None

        if expected_direction is not None:
            angle = self._line_angle_error_deg(
                direction,
                expected_direction,
            )

            if angle > p.max_edge_angle_deg:
                return None

        return point, direction, int(keep.sum())

    # ------------------------------------------------------------------
    # Face interior Lab model
    # ------------------------------------------------------------------
    def _interior_model(
        self,
        lab: np.ndarray,
        pred_local: np.ndarray,
    ):
        p = self.p

        center = pred_local.mean(axis=0)

        poly = center + float(p.boundary_interior_shrink) * (pred_local - center)

        mask = np.zeros(
            lab.shape[:2],
            dtype=np.uint8,
        )

        cv2.fillConvexPoly(
            mask,
            np.rint(poly).astype(np.int32),
            255,
        )

        px = lab[mask > 0]

        if len(px) < 150:
            return None

        if len(px) > 6000:
            px = px[:: max(1, len(px) // 6000)]

        mu = np.median(px, axis=0)

        mad = np.median(np.abs(px - mu), axis=0) * 1.4826

        sig = np.maximum(
            mad,
            float(p.boundary_min_sigma),
        )

        return (
            mu.astype(np.float32),
            sig.astype(np.float32),
        )

    # ------------------------------------------------------------------
    # Boundary evidence
    # ------------------------------------------------------------------
    def _boundary_stats(
        self,
        lab,
        interior,
        pt,
        d,
        normal,
        t_lo,
        t_hi,
        offset,
    ) -> Optional[Dict[str, float]]:
        p = self.p
        mu, sig = interior

        span = max(t_hi - t_lo, 1.0)

        n = int(
            np.clip(
                np.ceil(span / 4.0) + 1,
                8,
                60,
            )
        )

        ts = np.linspace(t_lo, t_hi, n)

        base = pt[None, :] + ts[:, None] * d[None, :] - offset[None, :]

        near = max(
            float(p.boundary_strip_near_px),
            0.5,
        )

        far = max(
            float(p.boundary_strip_far_px),
            near + 1.0,
        )

        mags = np.arange(
            near,
            far + 1e-6,
            1.0,
        )

        if len(mags) < 2:
            return None

        k = len(mags)

        offs = np.concatenate([-mags, mags]).astype(np.float32)

        mx = (base[:, 0:1] + offs[None, :] * normal[0]).astype(np.float32)

        my = (base[:, 1:2] + offs[None, :] * normal[1]).astype(np.float32)

        h, w = lab.shape[:2]

        valid = (mx >= 1) & (mx < w - 1) & (my >= 1) & (my < h - 1)

        samp = cv2.remap(
            lab,
            mx,
            my,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        z = (samp - mu[None, None, :]) / sig[None, None, :]

        dist = np.sqrt(np.mean(z * z, axis=2))

        similar = dist < float(p.boundary_similar_sigma)

        valid_in = valid[:, :k]
        valid_out = valid[:, k:]

        n_in = valid_in.sum(axis=1)
        n_out = valid_out.sum(axis=1)

        need = max(
            2,
            int(np.ceil(k * 0.6)),
        )

        good = (n_in >= need) & (n_out >= need)

        if int(good.sum()) < 5:
            return None

        in_frac = (similar[:, :k] & valid_in).sum(axis=1) / np.maximum(n_in, 1)

        out_frac = (similar[:, k:] & valid_out).sum(axis=1) / np.maximum(n_out, 1)

        inside_lab = samp[:, :k, :]
        outside_lab = samp[:, k:, :]

        paired_delta = np.linalg.norm(
            inside_lab - outside_lab,
            axis=2,
        )

        paired_valid = valid_in & valid_out
        paired_count = paired_valid.sum(axis=1)

        contrast_per_sample = np.where(
            paired_valid,
            paired_delta,
            0.0,
        ).sum(
            axis=1
        ) / np.maximum(paired_count, 1)

        contrast_per_sample = contrast_per_sample[good]
        inside_good = in_frac[good]
        outside_good = out_frac[good]

        contrast = float(np.median(contrast_per_sample))

        contrast_min = float(p.boundary_contrast_min)

        contrast_full = max(
            float(p.boundary_contrast_full),
            contrast_min + 1e-6,
        )

        contrast_score = float(
            np.clip(
                (contrast - contrast_min) / (contrast_full - contrast_min),
                0.0,
                1.0,
            )
        )

        contrast_consistency = float(
            np.mean(contrast_per_sample >= float(p.boundary_contrast_consistency))
        )

        inside_score = float(np.mean(inside_good))

        outside_score = float(1.0 - np.mean(outside_good))

        separation_score = float(
            np.clip(
                np.mean(inside_good - outside_good),
                0.0,
                1.0,
            )
        )

        inside_weight = max(
            float(p.boundary_inside_weight),
            0.0,
        )

        contrast_weight = max(
            float(p.boundary_contrast_weight),
            0.0,
        )

        continuity_weight = max(
            float(p.boundary_continuity_weight),
            0.0,
        )

        weight_sum = inside_weight + contrast_weight + continuity_weight

        if weight_sum <= 1e-9:
            score = separation_score
        else:
            score = (
                inside_weight * separation_score
                + contrast_weight * contrast_score
                + continuity_weight * contrast_consistency
            ) / weight_sum

        if inside_score < float(p.boundary_min_inside_similarity):
            score *= inside_score / max(
                float(p.boundary_min_inside_similarity),
                1e-6,
            )

        max_out = float(p.boundary_max_outside_similarity)

        if outside_score < (1.0 - max_out):
            score *= 0.5 + 0.5 * outside_score / max(1.0 - max_out, 1e-6)

        return {
            "score": float(np.clip(score, 0.0, 1.0)),
            "inside": inside_score,
            "outside": float(np.mean(outside_good)),
            "separation": separation_score,
            "contrast": contrast,
            "contrast_score": contrast_score,
            "continuity": contrast_consistency,
            "samples": int(good.sum()),
        }

    # ------------------------------------------------------------------
    # Final candidate scoring
    # ------------------------------------------------------------------
    def _finalize_candidate(
        self,
        hyp,
        arrs,
        expected,
        mid_g,
        normal,
        length,
        lab,
        interior,
        offset,
    ) -> Optional[Dict[str, Any]]:
        p = self.p

        pts_pad = arrs["pts"]
        valid_pad = arrs["valid"]

        S = valid_pad.shape[0]
        rows = np.arange(S)

        fit = self._fit_line(
            hyp["pts"],
            expected,
        )

        if fit is None:
            return None

        pt, d, _ = fit
        d = self._orient(d, expected)

        dist = np.abs(
            (pts_pad[..., 0] - pt[0]) * d[1] - (pts_pad[..., 1] - pt[1]) * d[0]
        )

        dist = np.where(
            valid_pad,
            dist,
            np.inf,
        )

        amin = dist.argmin(axis=1)
        dmin = dist[rows, amin]
        sel = dmin <= p.inlier_px

        if int(sel.sum()) < p.min_points:
            return None

        sel_pts = pts_pad[rows, amin][sel]

        fit = self._fit_line(
            sel_pts,
            expected,
        )

        if fit is None:
            return None

        pt, d, _ = fit
        d = self._orient(d, expected)

        dist_f = self._point_line_distance(
            sel_pts,
            pt,
            d,
        )

        inl = dist_f <= p.inlier_px

        if int(inl.sum()) < p.min_points:
            return None

        final_pts = sel_pts[inl]

        sel_samples = np.nonzero(sel)[0][inl]
        peak_idx = amin[sel_samples]

        grads = arrs["grad"][sel_samples, peak_idx]
        pols = arrs["pol"][sel_samples, peak_idx]

        fit_err = float(np.median(dist_f[inl]))

        angle_err = self._line_angle_error_deg(
            d,
            expected,
        )

        support = float(len(final_pts) / max(S, 1))

        t = (final_pts - mid_g[None, :]) @ expected

        coverage = float(
            np.clip(
                (t.max() - t.min()) / max(length, 1e-6),
                0.0,
                1.0,
            )
        )

        grad_score = float(
            np.mean(
                np.minimum(
                    grads / max(p.grad_cap, 1e-6),
                    1.0,
                )
            )
        )

        pos = int(np.count_nonzero(pols > 0))

        neg = int(np.count_nonzero(pols < 0))

        pol_cons = max(pos, neg) / max(len(pols), 1)

        base = (
            support
            * (0.5 + 0.5 * coverage)
            * (0.5 + 0.5 * grad_score)
            * (1.0 - p.polarity_penalty * (1.0 - pol_cons))
            / (1.0 + fit_err)
        )

        s_mid = self._line_offset(
            pt,
            d,
            mid_g,
            normal,
        )

        prior = float(self._offset_prior(s_mid))

        boundary = None

        if p.boundary_enabled and interior is not None:
            tl = (final_pts - pt[None, :]) @ d

            boundary = self._boundary_stats(
                lab,
                interior,
                pt,
                d,
                normal,
                float(tl.min()),
                float(tl.max()),
                offset,
            )

        bfactor = 1.0

        if boundary is not None:
            bw = float(np.clip(p.boundary_weight, 0.0, 1.0))

            bfactor = (1.0 - bw) + bw * float(np.clip(boundary["score"], 0.0, 1.0))

        unary = float(base * prior * bfactor)

        reject = None

        if support < p.min_support_frac:
            reject = "support_too_low"
        elif coverage < p.min_line_coverage:
            reject = "coverage_too_low"
        elif fit_err > p.max_line_fit_error_px:
            reject = "fit_error_too_large"
        elif angle_err > p.max_edge_angle_deg:
            reject = "angle_rejected"
        elif boundary is not None and boundary["score"] < p.min_boundary_score:
            reject = "boundary_rejected"

        return {
            "accepted": reject is None,
            "reject": reject,
            "point": pt,
            "dir": d,
            "pts": final_pts,
            "line": (
                pt,
                d,
                int(len(final_pts)),
            ),
            "s_mid": float(s_mid),
            "angle_deg": float(
                np.degrees(
                    np.arctan2(
                        expected[0] * d[1] - expected[1] * d[0],
                        float(d @ expected),
                    )
                )
            ),
            "support": support,
            "coverage": coverage,
            "fit_err": fit_err,
            "angle_err": float(angle_err),
            "pol_cons": float(pol_cons),
            "grad_score": grad_score,
            "base": float(base),
            "prior": prior,
            "boundary": boundary,
            "bfactor": float(bfactor),
            "unary": unary,
        }

    # ------------------------------------------------------------------
    def _edge_candidates(
        self,
        gray,
        lab,
        interior,
        p0,
        p1,
        normal,
        wall_px,
        offset,
        expected,
        mid_g,
    ):
        p = self.p

        length = float(np.linalg.norm(p1 - p0))

        band_in = float(p.band_px)
        band_out = float(p.band_px)

        if p.wall_outer_ratio > 0 and abs(wall_px) > 4.0:
            wall_limit = float(
                np.clip(
                    p.wall_outer_ratio * abs(wall_px),
                    p.min_out_px,
                    p.band_px,
                )
            )

            if wall_px > 0:
                band_out = wall_limit
            else:
                band_in = wall_limit

        arrs, info = self._collect_gradient_candidates(
            gray,
            p0,
            p1,
            normal,
            band_in,
            band_out,
            offset,
        )

        info["n_candidates"] = 0

        if info["gradient_candidate_count"] < p.min_points:
            return [], info

        hyps = self._rank_hypotheses(
            arrs,
            expected,
            mid_g,
            normal,
            length,
        )

        if not hyps:
            return [], info

        cands = []

        for hyp in hyps:
            c = self._finalize_candidate(
                hyp,
                arrs,
                expected,
                mid_g,
                normal,
                length,
                lab,
                interior,
                offset,
            )

            if c is not None:
                cands.append(c)

        cands.sort(key=lambda c: -c["unary"])

        kept: List[Dict[str, Any]] = []

        for c in cands:
            if any(
                abs(c["s_mid"] - k["s_mid"]) < 0.5 * p.nms_offset_px
                and abs(c["angle_deg"] - k["angle_deg"]) < 0.5 * p.nms_angle_deg
                for k in kept
            ):
                continue

            kept.append(c)

        info["n_candidates"] = len(kept)

        return kept, info

    # ------------------------------------------------------------------
    # Corner evidence
    # ------------------------------------------------------------------
    def _corner_gradient_evidence(
        self,
        gray: np.ndarray,
        corner_global: np.ndarray,
        normal_a: np.ndarray,
        normal_b: np.ndarray,
        offset: np.ndarray,
    ) -> float:
        p = self.p

        if not p.corner_evidence_enabled:
            return 1.0

        q = corner_global - offset

        cx = int(round(float(q[0])))
        cy = int(round(float(q[1])))

        radius = max(
            1,
            int(p.corner_evidence_radius_px),
        )

        h, w = gray.shape

        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)

        if x1 - x0 < 3 or y1 - y0 < 3:
            return 0.0

        gx = cv2.Sobel(
            gray,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )[y0:y1, x0:x1]

        gy = cv2.Sobel(
            gray,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )[y0:y1, x0:x1]

        mag = np.sqrt(gx * gx + gy * gy) + 1e-6

        min_grad = float(p.corner_gradient_min)

        cap = max(
            float(p.grad_cap),
            min_grad + 1e-6,
        )

        def directional_score(normal):
            normal = self._normalize_direction(normal)

            projection = np.abs(gx * normal[0] + gy * normal[1])

            aligned = projection / mag

            strength = np.clip(
                (mag - min_grad) / (cap - min_grad),
                0.0,
                1.0,
            )

            score = aligned * strength

            if score.size == 0:
                return 0.0

            return float(np.percentile(score, 85))

        score_a = directional_score(normal_a)
        score_b = directional_score(normal_b)

        return float(np.sqrt(max(score_a, 0.0) * max(score_b, 0.0)))

    # ------------------------------------------------------------------
    @staticmethod
    def _intersect(l1, l2) -> Optional[np.ndarray]:
        (p1, d1, _), (p2, d2, _) = l1, l2

        cr = d1[0] * d2[1] - d1[1] * d2[0]

        if abs(cr) < 0.2:
            return None

        a = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cr

        return p1 + a * d1

    @staticmethod
    def _segment(line, pts: np.ndarray) -> np.ndarray:
        p0, direction, _ = line

        t = (pts.astype(np.float64) - p0[None, :]) @ direction

        return np.stack(
            [
                p0 + t.min() * direction,
                p0 + t.max() * direction,
            ]
        )

    # ------------------------------------------------------------------
    # Observed corner construction
    # ------------------------------------------------------------------
    def _build_observed_corners(
        self,
        lines: List[Optional[Tuple[np.ndarray, np.ndarray, int]]],
        segments: Dict[int, np.ndarray],
        pred: np.ndarray,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build four corners from line intersections and optional endpoints.

        Edge i connects corner i to corner (i + 1) % 4.

        When one adjacent edge is missing, the remaining edge's fitted
        segment endpoint nearest to the predicted corner can be used.
        """
        p = self.p

        obs = np.full(
            (4, 2),
            np.nan,
            dtype=np.float64,
        )

        sources = ["missing"] * 4

        n_lines = sum(line is not None for line in lines)

        for corner_id in range(4):
            prev_edge = (corner_id - 1) % 4
            next_edge = corner_id

            prev_line = lines[prev_edge]
            next_line = lines[next_edge]

            if prev_line is not None and next_line is not None:
                corner = self._intersect(
                    prev_line,
                    next_line,
                )

                if corner is not None:
                    obs[corner_id] = corner
                    sources[corner_id] = "intersection"

                continue

            if not p.edge_endpoint_as_corner:
                continue

            # Endpoint fallback is intended for exactly three detected
            # edges. It is not used to fabricate corners from fewer edges.
            if n_lines != 3:
                continue

            available_edge = (
                prev_edge
                if prev_line is not None
                else next_edge if next_line is not None else None
            )

            if available_edge is None:
                continue

            seg = segments.get(available_edge)

            if seg is None or seg.shape != (2, 2):
                continue

            distances = np.linalg.norm(
                seg - pred[corner_id][None, :],
                axis=1,
            )

            endpoint_idx = int(np.argmin(distances))

            endpoint = seg[endpoint_idx]

            if (
                not np.isfinite(endpoint).all()
                or distances[endpoint_idx] > p.corner_tol_px
            ):
                continue

            obs[corner_id] = endpoint
            sources[corner_id] = "edge_endpoint"

        return obs, sources

    # ------------------------------------------------------------------
    # Joint candidate selection
    # ------------------------------------------------------------------
    def _select_joint(
        self,
        cands,
        pred,
        geoms,
        gray=None,
        offset=None,
    ):
        p = self.p

        K = [len(c) for c in cands]

        unary = [[c["unary"] for c in cs] for cs in cands]

        adj = []
        corner_evidence = []

        for i in range(4):
            j = (i + 1) % 4

            tab = [[0.0] * K[j] for _ in range(K[i])]

            ev_tab = [[0.0] * K[j] for _ in range(K[i])]

            for a in range(K[i]):
                for b in range(K[j]):
                    ca = cands[i][a]
                    cb = cands[j][b]

                    corner = self._intersect(
                        ca["line"],
                        cb["line"],
                    )

                    if corner is None:
                        pen = p.corner_penalty
                        ev = 0.0
                    else:
                        dist = float(np.linalg.norm(corner - pred[j]))

                        pen = p.corner_penalty * min(
                            1.0,
                            (dist / max(p.corner_tol_px, 1e-6)) ** 2,
                        )

                        if (
                            gray is not None
                            and offset is not None
                            and p.corner_evidence_enabled
                        ):
                            na = np.array(
                                [
                                    -ca["dir"][1],
                                    ca["dir"][0],
                                ],
                                dtype=np.float64,
                            )

                            nb = np.array(
                                [
                                    -cb["dir"][1],
                                    cb["dir"][0],
                                ],
                                dtype=np.float64,
                            )

                            ev = self._corner_gradient_evidence(
                                gray,
                                corner,
                                na,
                                nb,
                                offset,
                            )

                            pen += p.corner_evidence_weight * (1.0 - ev)
                        else:
                            ev = 1.0

                    tab[a][b] = pen
                    ev_tab[a][b] = ev

            adj.append(tab)
            corner_evidence.append(ev_tab)

        # Opposite edges: parallelism and plausible separation.
        opp = []

        for i in (0, 1):
            k = i + 2

            tab = [[0.0] * K[k] for _ in range(K[i])]

            if K[i] and K[k]:
                gi, gk = geoms[i], geoms[k]

                axis = gi["normal"] - gk["normal"]
                axis = axis / (np.linalg.norm(axis) + 1e-12)

                d_pred = abs(float((gk["mid"] - gi["mid"]) @ axis))

                ang_pred = self._wrap_half_pi(
                    self._dir_angle_deg(gk["u"]) - self._dir_angle_deg(gi["u"])
                )

                if d_pred > 5.0:
                    for a in range(K[i]):
                        for b in range(K[k]):
                            ca = cands[i][a]
                            cb = cands[k][b]

                            ratio = (d_pred + ca["s_mid"] + cb["s_mid"]) / d_pred

                            pr = min(
                                1.0,
                                ((ratio - 1.0) / max(p.opposite_ratio_tol, 1e-6)) ** 2,
                            )

                            ang_c = self._wrap_half_pi(
                                self._dir_angle_deg(cb["dir"])
                                - self._dir_angle_deg(ca["dir"])
                            )

                            pa = min(
                                1.0,
                                (
                                    self._wrap_half_pi(ang_c - ang_pred)
                                    / max(
                                        p.opposite_angle_tol_deg,
                                        1e-6,
                                    )
                                )
                                ** 2,
                            )

                            tab[a][b] = p.opposite_penalty * min(1.0, pr + pa)

            opp.append(tab)

        required_edges = (
            max(3, min(4, int(p.min_valid_edges))) if p.edge_endpoint_as_corner else 4
        )

        best = None
        n_combos = 0

        for combo in product(*[range(k + 1) for k in K]):
            chosen = [e for e in range(4) if combo[e] < K[e]]

            if len(chosen) < required_edges:
                continue

            n_combos += 1

            total = sum(unary[e][combo[e]] for e in chosen)

            for i in range(4):
                j = (i + 1) % 4
                a, b = combo[i], combo[j]

                if a < K[i] and b < K[j]:
                    total -= adj[i][a][b]

            for i in (0, 1):
                a = combo[i]
                b = combo[i + 2]

                if a < K[i] and b < K[i + 2]:
                    total -= opp[i][a][b]

            if best is None or total > best[0]:
                best = (
                    float(total),
                    combo,
                )

        if best is None:
            return None

        return (
            best[1],
            best[0],
            n_combos,
            corner_evidence,
        )

    # ------------------------------------------------------------------
    # Pose optimization
    # ------------------------------------------------------------------
    def _solve(
        self,
        c3d_sel,
        P,
        Nn,
        K,
        R0,
        t0,
    ):
        if least_squares is None:
            return None

        rvec0, _ = cv2.Rodrigues(R0)

        sr = max(
            float(self.p.sigma_rot_deg),
            1e-6,
        )

        st = max(
            float(self.p.sigma_trans_mm),
            1e-6,
        )

        x0 = np.concatenate([rvec0.ravel(), t0])

        n_res = len(P)

        def resid(x):
            R, _ = cv2.Rodrigues(x[:3])

            cam = c3d_sel @ R.T + x[3:]
            uv = self._project(cam, K)

            r_img = np.einsum(
                "ij,ij->i",
                uv - P,
                Nn,
            )

            rv, _ = cv2.Rodrigues(R @ R0.T)

            r_rot = np.degrees(rv.ravel()) / sr
            r_t = (x[3:] - t0) / st

            return np.concatenate([r_img, r_rot, r_t])

        sol = least_squares(
            resid,
            x0,
            method="lm",
            x_scale="jac",
            max_nfev=60,
        )

        if not sol.success and not np.isfinite(sol.x).all():
            return None

        if not np.isfinite(sol.x).all():
            return None

        R, _ = cv2.Rodrigues(sol.x[:3])

        return (
            R,
            sol.x[3:],
            resid(sol.x)[:n_res],
        )

    @staticmethod
    def _quad_area(q: np.ndarray) -> float:
        x = q[:, 0]
        y = q[:, 1]

        return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @staticmethod
    def _fill_diag(
        diag: Dict[str, Any],
        c: Dict[str, Any],
    ) -> None:
        diag["line_inlier_count"] = int(len(c["pts"]))

        diag["line_support_ratio"] = float(c["support"])

        diag["line_coverage_ratio"] = float(c["coverage"])

        diag["line_fit_error_px"] = float(c["fit_err"])

        diag["line_angle_error_deg"] = float(c["angle_err"])

        diag["line_polarity_consistency"] = float(c["pol_cons"])

        diag["line_gradient_score"] = float(c["grad_score"])

        diag["line_ransac_score"] = float(c["base"] * c["prior"])

        diag["line_offset_px"] = float(c["s_mid"])

        diag["line_offset_prior"] = float(c["prior"])

        diag["line_unary_score"] = float(c["unary"])

        diag["line_accepted"] = bool(c["accepted"])

        b = c["boundary"]

        if b is not None:
            diag["line_boundary_score"] = float(b["score"])

            diag["line_inside_similarity"] = float(b["inside"])

            diag["line_outside_similarity"] = float(b["outside"])

            diag["line_boundary_separation"] = float(b["separation"])

            diag["line_boundary_contrast"] = float(b["contrast"])

            diag["line_boundary_contrast_score"] = float(b["contrast_score"])

            diag["line_boundary_continuity"] = float(b["continuity"])
        else:
            diag["line_boundary_score"] = None
            diag["line_boundary_contrast"] = None
            diag["line_boundary_continuity"] = None

    # ------------------------------------------------------------------
    # Main refine
    # ------------------------------------------------------------------
    def refine(
        self,
        bgr: np.ndarray,
        camera: np.ndarray,
        label: int,
        R: np.ndarray,
        t_mm: np.ndarray,
    ) -> RefineResult:
        p = self.p

        R0 = np.asarray(
            R,
            dtype=np.float64,
        )

        t0 = np.asarray(
            t_mm,
            dtype=np.float64,
        )

        K = np.asarray(
            camera,
            dtype=np.float64,
        )

        vis: Dict[str, Any] = {}

        shifts: List[Optional[float]] = [None] * 4
        support: List[Optional[float]] = [None] * 4
        walls: List[Optional[float]] = [None] * 4

        edge_diagnostics: List[Dict[str, Any]] = [
            {
                "edge_id": i,
                "gradient_candidate_count": 0,
                "n_candidates": 0,
                "line_inlier_count": 0,
                "line_support_ratio": 0.0,
                "line_coverage_ratio": 0.0,
                "line_fit_error_px": None,
                "line_angle_error_deg": None,
                "line_accepted": False,
                "reason": None,
            }
            for i in range(4)
        ]

        extra_info: Dict[str, Any] = {}

        def mk_info(
            ok: bool,
            reason: str,
            **kw,
        ) -> Dict[str, Any]:
            info = {
                "ok": ok,
                "reason": reason,
                "edge_shift_px": list(shifts),
                "edge_support": list(support),
                "edge_wall_px": list(walls),
                "edge_diagnostics": edge_diagnostics,
            }

            info.update(extra_info)
            info.update(kw)

            return info

        def fail(
            reason: str,
            **kw,
        ) -> RefineResult:
            return RefineResult(
                False,
                R0,
                t0,
                mk_info(False, reason, **kw),
                **vis,
            )

        box = self.boxes.get(int(label))

        if box is None:
            return fail("no_size_for_label")

        sign, c3d, cos = self._pick_visible_face(
            box,
            R0,
            t0,
        )

        if cos < p.min_face_cos:
            return fail(
                "face_grazing",
                cos=float(cos),
            )

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

        vis["corners_pred"] = pred
        vis["corners_bottom"] = pred_b

        # ROI around predicted top and bottom faces.
        h, w = bgr.shape[:2]

        allp = np.vstack([pred, pred_b])

        pad = int(p.band_px + 6)

        x0 = int(
            max(
                0,
                np.floor(allp[:, 0].min()) - pad,
            )
        )

        y0 = int(
            max(
                0,
                np.floor(allp[:, 1].min()) - pad,
            )
        )

        x1 = int(
            min(
                w,
                np.ceil(allp[:, 0].max()) + pad,
            )
        )

        y1 = int(
            min(
                h,
                np.ceil(allp[:, 1].max()) + pad,
            )
        )

        if x1 <= x0 or y1 <= y0 or (x1 - x0) < p.min_roi_px or (y1 - y0) < p.min_roi_px:
            return fail("roi_too_small")

        roi = bgr[y0:y1, x0:x1]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY,
        )

        lab = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)

        k = max(
            1,
            int(p.blur_ksize) | 1,
        )

        if k > 1:
            gray = cv2.GaussianBlur(
                gray,
                (k, k),
                0,
            )

            lab = cv2.GaussianBlur(
                lab,
                (k, k),
                0,
            )

        gray = gray.astype(np.float32)

        offset = np.array(
            [x0, y0],
            dtype=np.float64,
        )

        # Face interior colour model.
        interior = None

        if p.boundary_enabled:
            interior = self._interior_model(
                lab,
                pred - offset,
            )

            if interior is not None:
                extra_info["boundary_model"] = {
                    "lab_median": [float(v) for v in interior[0]],
                    "sigma": [float(v) for v in interior[1]],
                }
            else:
                extra_info["boundary_model"] = None

        # Candidate lines for every edge.
        centroid = pred.mean(axis=0)

        geoms: List[Dict[str, Any]] = []
        edge_cands: List[List[Dict[str, Any]]] = []

        for i in range(4):
            j = (i + 1) % 4

            u = pred[j] - pred[i]

            Lu = float(np.linalg.norm(u))

            u = u / Lu if Lu > 1e-9 else np.array([1.0, 0.0])

            normal = np.array(
                [-u[1], u[0]],
                dtype=np.float64,
            )

            mid = 0.5 * (pred[i] + pred[j])

            if normal @ (mid - centroid) < 0:
                normal = -normal

            mid_b = 0.5 * (pred_b[i] + pred_b[j])

            wall = float(normal @ (mid_b - mid))

            walls[i] = wall

            geoms.append(
                {
                    "u": u,
                    "normal": normal,
                    "mid": mid,
                }
            )

            if Lu < p.min_edge_len_px:
                edge_diagnostics[i]["reason"] = "edge_too_short"

                edge_cands.append([])
                continue

            cands, sinfo = self._edge_candidates(
                gray,
                lab,
                interior,
                pred[i] - offset,
                pred[j] - offset,
                normal,
                wall,
                offset,
                u,
                mid,
            )

            edge_diagnostics[i].update(sinfo)
            edge_cands.append(cands)

        # Joint candidate selection.
        accepted: List[List[Dict[str, Any]]] = []

        for i in range(4):
            cs = [
                c
                for c in edge_cands[i]
                if (c["accepted"] and c["unary"] >= p.min_candidate_unary)
            ]

            cs = cs[: max(1, int(p.joint_candidates))]

            for rank, c in enumerate(cs):
                c["rank"] = rank

            accepted.append(cs)

        joint = self._select_joint(
            accepted,
            pred,
            geoms,
            gray,
            offset,
        )

        chosen: Dict[int, Dict[str, Any]] = {}

        if joint is not None:
            (
                combo,
                joint_score,
                n_combos,
                corner_evidence,
            ) = joint

            extra_info["joint_score"] = float(joint_score)

            extra_info["joint_combos"] = int(n_combos)

            selected_corner_scores = []

            for i in range(4):
                if combo[i] < len(accepted[i]):
                    chosen[i] = accepted[i][combo[i]]

            for i in range(4):
                j = (i + 1) % 4

                a = combo[i]
                b = combo[j]

                if a < len(accepted[i]) and b < len(accepted[j]):
                    ev = float(corner_evidence[i][a][b])

                    selected_corner_scores.append(ev)

                    edge_diagnostics[i]["corner_evidence"] = ev

            extra_info["corner_evidence"] = selected_corner_scores

            extra_info["corner_evidence_mean"] = (
                float(np.mean(selected_corner_scores))
                if selected_corner_scores
                else None
            )

        # Per-edge diagnostics.
        for i in range(4):
            diag = edge_diagnostics[i]
            cands = edge_cands[i]

            top = None

            if i in chosen:
                top = chosen[i]
            elif accepted[i]:
                top = accepted[i][0]
            elif cands:
                top = cands[0]

            if top is not None:
                self._fill_diag(
                    diag,
                    top,
                )

            if i in chosen:
                diag["line_accepted"] = True
                diag["reason"] = "ok"
                diag["selected_rank"] = int(chosen[i]["rank"])
            elif accepted[i]:
                diag["line_accepted"] = False
                diag["reason"] = "not_selected_joint"
            elif cands:
                diag["line_accepted"] = False
                diag["reason"] = cands[0]["reject"] or "unary_too_low"
            elif diag.get("reason") is None:
                diag["reason"] = "line_detection_failed"

        lines: List[Optional[Tuple[np.ndarray, np.ndarray, int]]] = [None] * 4

        edge_pts: List[np.ndarray] = []
        edge_segs: List[np.ndarray] = []

        # Keep segments indexed by their corresponding edge ID.
        segments: Dict[int, np.ndarray] = {}

        for i in sorted(chosen):
            c = chosen[i]

            line = (
                c["point"],
                c["dir"],
                int(len(c["pts"])),
            )

            lines[i] = line

            shifts[i] = float(c["s_mid"])
            support[i] = float(c["support"])

            seg = self._segment(
                line,
                c["pts"],
            )

            segments[i] = seg

            edge_pts.append(c["pts"])
            edge_segs.append(seg)

        valid_idx = sorted(chosen)

        required_edges = (
            max(3, min(4, int(p.min_valid_edges))) if p.edge_endpoint_as_corner else 4
        )

        extra_info["edge_endpoint_as_corner"] = bool(p.edge_endpoint_as_corner)

        extra_info["required_edges"] = int(required_edges)

        if len(valid_idx) < required_edges:
            vis["edge_pts"] = edge_pts
            vis["edge_segs"] = edge_segs

            with_candidates = [i for i in range(4) if accepted[i]]

            return fail(
                "too_few_edges",
                n_edges=len(valid_idx),
                valid_edges=valid_idx,
                available_edges=with_candidates,
                required_edges=required_edges,
                joint_selected=False,
            )

        vis["edge_pts"] = edge_pts
        vis["edge_segs"] = edge_segs

        # Construct four corners.
        obs, corner_sources = self._build_observed_corners(
            lines,
            segments,
            pred,
        )

        vis["corners_obs"] = obs

        extra_info["corner_sources"] = corner_sources
        extra_info["corner_valid"] = [bool(np.isfinite(q).all()) for q in obs]

        # Four valid corners are mandatory before pose correction.
        if obs.shape != (4, 2) or not np.isfinite(obs).all():
            missing = [i for i in range(4) if not np.isfinite(obs[i]).all()]

            return fail(
                "incomplete_corners",
                n_edges=len(valid_idx),
                valid_edges=valid_idx,
                missing_corners=missing,
                corner_sources=corner_sources,
                quad_corners=obs.tolist(),
            )

        # Validate corner displacement.
        corner_shifts = np.linalg.norm(
            obs - pred,
            axis=1,
        )

        max_corner_shift = float(corner_shifts.max())

        mean_corner_shift = float(corner_shifts.mean())

        extra_info["corner_shift_px"] = mean_corner_shift

        extra_info["corner_shift_max_px"] = max_corner_shift

        if max_corner_shift > p.corner_tol_px:
            return fail(
                "corner_shift_too_large",
                corner_shift_px=mean_corner_shift,
                corner_shift_max_px=max_corner_shift,
                corner_tol_px=float(p.corner_tol_px),
                corner_sources=corner_sources,
            )

        # Validate quadrilateral area and convexity.
        area_pred = self._quad_area(pred)
        area_obs = self._quad_area(obs)

        ratio = float(area_obs / (area_pred + 1e-9))

        convex = bool(cv2.isContourConvex(obs.astype(np.float32).reshape(-1, 1, 2)))

        extra_info["area_ratio"] = ratio
        extra_info["quad_convex"] = convex

        if not (p.area_ratio_min <= ratio <= p.area_ratio_max):
            return fail(
                "quad_area_ratio",
                area_ratio=ratio,
                area_ratio_min=p.area_ratio_min,
                area_ratio_max=p.area_ratio_max,
                quad_corners=obs.tolist(),
                quad_convex=convex,
                corner_sources=corner_sources,
            )

        if not convex:
            return fail(
                "quad_non_convex",
                area_ratio=ratio,
                quad_corners=obs.tolist(),
                quad_convex=False,
                corner_sources=corner_sources,
            )

        # Reject degenerate or implausibly short quadrilateral edges.
        observed_edge_lengths = np.linalg.norm(
            np.roll(obs, -1, axis=0) - obs,
            axis=1,
        )

        min_observed_edge = float(observed_edge_lengths.min())

        extra_info["quad_edge_lengths_px"] = [float(v) for v in observed_edge_lengths]

        if min_observed_edge < p.min_edge_len_px:
            return fail(
                "quad_edge_too_short",
                min_edge_px=min_observed_edge,
                min_edge_threshold_px=float(p.min_edge_len_px),
                quad_corners=obs.tolist(),
                corner_sources=corner_sources,
            )

        # Pose solve uses only actual detected lines.
        # Endpoint-generated corners are validation evidence, not extra
        # independently detected line constraints.
        idxs: List[int] = []
        points: List[np.ndarray] = []
        normals: List[np.ndarray] = []

        for i in valid_idx:
            line_point, line_direction, _ = lines[i]

            n_line = np.array(
                [
                    -line_direction[1],
                    line_direction[0],
                ],
                dtype=np.float64,
            )

            for corner_id in (
                i,
                (i + 1) % 4,
            ):
                idxs.append(corner_id)
                points.append(line_point)
                normals.append(n_line)

        solved = self._solve(
            c3d[idxs],
            np.asarray(
                points,
                dtype=np.float64,
            ),
            np.asarray(
                normals,
                dtype=np.float64,
            ),
            K,
            R0,
            t0,
        )

        if solved is None:
            return fail("optimizer_failed")

        R1, t1, r_px = solved

        ref = self._project(
            c3d @ R1.T + t1,
            K,
        )

        vis["corners_ref"] = ref

        reproj = float(np.abs(r_px).mean())

        reproj_max = float(np.abs(r_px).max())

        dt = float(np.linalg.norm(t1 - t0))

        dang = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(R1 @ R0.T) - 1.0) / 2.0,
                        -1.0,
                        1.0,
                    )
                )
            )
        )

        extra: Dict[str, Any] = {
            "reproj_px": reproj,
            "reproj_max_px": reproj_max,
            "dt_mm": dt,
            "dang_deg": dang,
            "n_edges": len(valid_idx),
            "valid_edges": valid_idx,
            "edge_points": [
                (lines[i][2] if lines[i] is not None else None) for i in range(4)
            ],
            "edge_diagnostics": edge_diagnostics,
            "corner_sources": corner_sources,
            "corner_shift_px": mean_corner_shift,
            "corner_shift_max_px": max_corner_shift,
            "area_ratio": ratio,
            "quad_convex": convex,
            "quad_edge_lengths_px": [float(v) for v in observed_edge_lengths],
        }

        info = mk_info(
            True,
            "ok",
            **extra,
        )

        if reproj > p.max_reproj_px:
            info.update(
                ok=False,
                reason="reproj_too_large",
            )
        elif dt > p.max_trans_mm:
            info.update(
                ok=False,
                reason="trans_jump",
            )
        elif dang > p.max_rot_deg:
            info.update(
                ok=False,
                reason="rot_jump",
            )

        if not info["ok"]:
            return RefineResult(
                False,
                R0,
                t0,
                info,
                **vis,
            )

        return RefineResult(
            True,
            R1,
            t1,
            info,
            **vis,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def draw(
        debug: np.ndarray,
        res: RefineResult,
    ) -> None:
        """Draw fitted edge segments and their observed corners."""

        def ip(q):
            return (
                int(round(float(q[0]))),
                int(round(float(q[1]))),
            )

        for seg in res.edge_segs or []:
            if not np.isfinite(seg).all():
                continue

            cv2.line(
                debug,
                ip(seg[0]),
                ip(seg[1]),
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        if res.corners_obs is not None:
            for q in res.corners_obs:
                if not np.isfinite(q).all():
                    continue

                cv2.circle(
                    debug,
                    ip(q),
                    4,
                    (0, 255, 255),
                    -1,
                    cv2.LINE_AA,
                )
