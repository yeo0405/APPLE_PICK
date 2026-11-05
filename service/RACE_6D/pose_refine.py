"""RACE-6D pose refinement using vertices detected by VertexDetector.

Vertex detection helpers live in vertex_detector.py. This module retains
pose optimization, validation, result structures and the PoseRefiner API.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

try:
    from .vertex_detector import RefineParams, VertexDetector, VertexDetectionResult
except ImportError:
    from vertex_detector import RefineParams, VertexDetector, VertexDetectionResult

logger = logging.getLogger(__name__)


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
        detector = VertexDetector.from_config(cfg)

        if detector is None:
            return None

        return cls(detector.boxes, detector.p)

    @staticmethod
    def _project(
        pts_cam: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
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
    ):
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

    def _solve(
        self,
        c3d_sel: np.ndarray,
        P: np.ndarray,
        Nn: np.ndarray,
        K: np.ndarray,
        R0: np.ndarray,
        t0: np.ndarray,
    ):
        if least_squares is None:
            return None

        rvec0, _ = cv2.Rodrigues(R0)

        sr = max(float(self.p.sigma_rot_deg), 1e-6)
        st = max(float(self.p.sigma_trans_mm), 1e-6)

        x0 = np.concatenate(
            [
                rvec0.ravel(),
                t0,
            ]
        )

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

            return np.concatenate(
                [
                    r_img,
                    r_rot,
                    r_t,
                ]
            )

        sol = least_squares(
            resid,
            x0,
            method="lm",
            x_scale="jac",
            max_nfev=60,
        )

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

    def refine(
        self,
        bgr: np.ndarray,
        camera: np.ndarray,
        label: int,
        R: np.ndarray,
        t_mm: np.ndarray,
        detection: VertexDetectionResult,
    ) -> RefineResult:
        p = self.p

        R0 = np.asarray(R, dtype=np.float64)
        t0 = np.asarray(t_mm, dtype=np.float64)
        K = np.asarray(camera, dtype=np.float64)

        vis: Dict[str, Any] = {}

        shifts = detection.info.get(
            "edge_shift_px",
            [None] * 4,
        )

        support = detection.info.get(
            "edge_support",
            [None] * 4,
        )

        walls = detection.info.get(
            "edge_wall_px",
            [None] * 4,
        )

        edge_diagnostics = detection.info.get(
            "edge_diagnostics",
            [],
        )

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

        if not detection.ok:
            return fail(
                "vertex_detection_failed",
                vertex_info=detection.info,
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

        pred = detection.corners_pred
        pred_b = detection.corners_bottom
        obs = detection.corners_obs

        lines = detection.lines or [None] * 4
        valid_idx = detection.valid_idx or []

        edge_pts = detection.edge_pts or []
        edge_segs = detection.edge_segs or []

        vis.update(
            corners_pred=pred,
            corners_bottom=pred_b,
            corners_obs=obs,
            edge_pts=edge_pts,
            edge_segs=edge_segs,
        )

        if pred is None or obs is None or len(valid_idx) < 3:
            return fail("invalid_vertex_detection")

        corner_sources = detection.info.get(
            "corner_sources",
            [],
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
            np.asarray(points, dtype=np.float64),
            np.asarray(normals, dtype=np.float64),
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

    @staticmethod
    def draw(
        debug: np.ndarray,
        res: RefineResult,
    ) -> None:
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
                if np.isfinite(q).all():
                    cv2.circle(
                        debug,
                        ip(q),
                        4,
                        (0, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )

        if res.corners_ref is not None:
            for q in res.corners_ref:
                if np.isfinite(q).all():
                    cv2.circle(
                        debug,
                        ip(q),
                        3,
                        (255, 255, 0),
                        -1,
                        cv2.LINE_AA,
                    )
