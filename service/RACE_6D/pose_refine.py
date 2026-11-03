"""RGB-only post-processing for RACE-6D cuboid poses.

Pipeline (per detection):
  1. Project the box with the RACE pose, pick the camera-facing +Z/-Z face
     (top face) and also project the opposite face (to know where the side
     wall / silhouette is).
  2. For each of the 4 top-face edges, search ONE straight line in a small
     band around the prediction (Hough-like search over offset + tilt,
     polarity-consistent), then refit it from the supporting edge points.
  3. Solve the pose by minimizing the distance of the projected top-face
     corners to their detected lines + a prior pulling toward the RACE pose.
     At least `min_valid_edges` valid edges are needed (default 3).
  4. Accept only if all sanity checks pass, otherwise keep the RACE pose.

Units: translation in mm (same as RACE), sizes in config are meters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

logger = logging.getLogger(__name__)


@dataclass
class RefineParams:
    # edge search
    band_px: float = 24.0
    weight_ratio: float = 3.0  # prior width = weight_ratio * band_px (larger = flatter)
    samples_per_edge: int = 24
    edge_margin: float = 0.12
    min_grad: float = 4.0
    grad_cap: float = 20.0  # per-sample support saturates at this gradient
    search_angle_deg: float = 6.0  # max tilt of the line vs the predicted edge
    angle_step_deg: float = 0.5
    support_px: float = 2.0  # sample counts as support if within this of the line
    min_support_frac: float = 0.4
    wall_outer_ratio: float = (
        0.75  # outward search limit = ratio * visible wall width (0 = off)
    )
    min_out_px: float = 8.0
    min_valid_edges: int = 3
    # line fit
    inlier_px: float = 1.5
    min_points: int = 8
    max_edge_angle_deg: float = 8.0
    # sanity
    max_reproj_px: float = 2.0
    max_trans_mm: float = 20.0
    max_rot_deg: float = 8.0
    area_ratio_min: float = 0.7
    area_ratio_max: float = 1.4
    min_face_cos: float = 0.2
    min_edge_len_px: float = 20.0
    blur_ksize: int = 5
    # prior
    sigma_rot_deg: float = 5.0
    sigma_trans_mm: float = 10.0


@dataclass
class RefineResult:
    ok: bool
    R: np.ndarray
    t_mm: np.ndarray
    info: Dict[str, Any]
    corners_pred: Optional[np.ndarray] = None  # RACE top face (4,2)
    corners_obs: Optional[np.ndarray] = (
        None  # intersections of detected lines (4,2), NaN if missing
    )
    corners_ref: Optional[np.ndarray] = None  # refined top face (4,2)
    corners_bottom: Optional[np.ndarray] = None  # RACE bottom face (4,2)
    edge_pts: Optional[List[np.ndarray]] = None  # supporting edge points per valid edge
    edge_segs: Optional[List[np.ndarray]] = None  # fitted line segments per valid edge


class PoseRefiner:
    def __init__(self, boxes: Dict[int, Dict[str, np.ndarray]], params: RefineParams):
        self.boxes = boxes  # label -> {"size_mm": (3,), "offset_mm": (3,)}
        self.p = params

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["PoseRefiner"]:
        if not cfg or not cfg.get("ENABLED", False):
            return None

        boxes: Dict[int, Dict[str, np.ndarray]] = {}
        for label, item in (cfg.get("LABELS") or {}).items():
            s = item["SIZE_M"]
            size_mm = np.array([s["X"], s["Y"], s["Z"]], dtype=np.float64) * 1000.0
            offset_mm = (
                np.asarray(item.get("ORIGIN_OFFSET_M", [0, 0, 0]), dtype=np.float64)
                * 1000.0
            )
            boxes[int(label)] = {"size_mm": size_mm, "offset_mm": offset_mm}

        valid = {f.name for f in fields(RefineParams)}
        kwargs = {}
        for k, v in (cfg.get("PARAMS") or {}).items():
            if k.lower() in valid:
                kwargs[k.lower()] = v
            else:
                logger.warning("Unknown RGB_REFINE param ignored: %s", k)

        return cls(boxes, RefineParams(**kwargs))

    # ------------------------------------------------------------------
    @staticmethod
    def _project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
        z = np.maximum(pts_cam[:, 2], 1e-3)
        u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]
        return np.stack([u, v], axis=1)

    def _face_corners_obj(self, box, sign: int) -> np.ndarray:
        hx, hy, hz = box["size_mm"] / 2.0
        c = np.array(
            [
                [-hx, -hy, sign * hz],
                [+hx, -hy, sign * hz],
                [+hx, +hy, sign * hz],
                [-hx, +hy, sign * hz],
            ],
            dtype=np.float64,
        )
        return c + box["offset_mm"]

    def _pick_visible_face(self, box, R, t):
        """Return (sign, corners_obj, cos) of the camera-facing Z face."""
        best = None
        for sign in (+1, -1):
            normal = R @ np.array([0.0, 0.0, float(sign)])
            center = (
                R @ (box["offset_mm"] + np.array([0, 0, sign * box["size_mm"][2] / 2]))
                + t
            )
            cos = float(-normal @ center / (np.linalg.norm(center) + 1e-9))
            if best is None or cos > best[2]:
                best = (sign, self._face_corners_obj(box, sign), cos)
        return best

    # ------------------------------------------------------------------
    def _search_edge_line(self, gray, p0, p1, n, wall_px, offset):
        """Find ONE straight line along edge p0->p1 (local coords).

        n is the unit OUTWARD normal; s > 0 means outward.
        Returns (points_global (K,2), median_shift_px, support_frac) or None.
        """
        p = self.p
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < p.min_edge_len_px:
            return None

        step = 0.5
        band_in = p.band_px
        band_out = p.band_px

        if p.wall_outer_ratio > 0 and abs(wall_px) > 4.0:
            wall_limit = float(
                np.clip(p.wall_outer_ratio * abs(wall_px), p.min_out_px, p.band_px)
            )

            if wall_px > 0:
                band_out = wall_limit
            else:
                band_in = wall_limit

        s = np.arange(-band_in, band_out + 1e-6, step)  # (M,)
        M = len(s)

        ts = np.linspace(p.edge_margin, 1.0 - p.edge_margin, p.samples_per_edge)
        N = len(ts)
        base = p0[None, :] + ts[:, None] * d[None, :]  # (N,2)
        pos = base[:, None, :] + s[None, :, None] * n[None, None, :]  # (N,M,2)

        h, w = gray.shape
        mx = pos[..., 0].astype(np.float32)
        my = pos[..., 1].astype(np.float32)
        inside = (mx >= 1) & (mx < w - 2) & (my >= 1) & (my < h - 2)
        prof = cv2.remap(
            gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )

        grad = np.gradient(prof, step, axis=1)  # (N,M), per pixel along +s
        valid = inside & np.roll(inside, 1, axis=1) & np.roll(inside, -1, axis=1)

        G = {
            "pos": np.where(valid, np.maximum(grad, 0.0), 0.0),
            "neg": np.where(valid, np.maximum(-grad, 0.0), 0.0),
        }

        cap = max(p.grad_cap, p.min_grad + 1e-6)
        ang = np.deg2rad(
            np.arange(-p.search_angle_deg, p.search_angle_deg + 1e-6, p.angle_step_deg)
        )  # (K,)
        u_i = (ts - 0.5) * L  # (N,) arc offset along the edge

        # line position at every sample for every (offset, tilt): (M,K,N)
        shift = s[:, None, None] + np.tan(ang)[None, :, None] * u_i[None, None, :]
        idx = np.rint((shift - s[0]) / step).astype(np.int64)
        in_rng = (idx >= 0) & (idx < M)
        idx_c = np.clip(idx, 0, M - 1)
        rows = np.arange(N)[None, None, :]
        prior = np.exp(-0.5 * (s / (max(p.weight_ratio, 1e-3) * p.band_px)) ** 2)[
            :, None
        ]

        best = None
        for name, Gm in G.items():  # polarity must be consistent along the line
            Gc = np.where(Gm >= p.min_grad, np.minimum(Gm, cap) / cap, 0.0)
            sc = (Gc[rows, idx_c] * in_rng).sum(axis=2) * prior  # (M,K)
            m, k = np.unravel_index(int(np.argmax(sc)), sc.shape)
            if best is None or sc[m, k] > best[0]:
                best = (float(sc[m, k]), int(m), int(k), name)

        score, m, k, name = best
        if score <= 0:
            return None

        # re-pick the actual edge position of every sample near that line
        s_line = s[m] + np.tan(ang[k]) * u_i  # (N,)
        Gm = G[name]
        il = np.rint((s_line - s[0]) / step).astype(np.int64)
        W = max(1, int(round(p.support_px / step)))
        j = il[:, None] + np.arange(-W, W + 1)[None, :]
        jv = (j >= 0) & (j < M)
        jc = np.clip(j, 0, M - 1)
        r = np.arange(N)[:, None]
        vals = np.where(jv, Gm[r, jc], 0.0)
        a = np.argmax(vals, axis=1)
        best_v = vals[np.arange(N), a]
        ok = best_v >= p.min_grad

        support = float(ok.mean())
        if ok.sum() < p.min_points or support < p.min_support_frac:
            return None

        s_pt = s[jc[np.arange(N), a]]
        pts = base[ok] + s_pt[ok, None] * n[None, :]
        return pts + offset[None, :], float(np.median(s_pt[ok])), support

    def _fit_line(self, pts):
        p = self.p
        pts = pts.astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        dist = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        keep = dist < p.inlier_px
        if keep.sum() < p.min_points:
            return None
        vx, vy, x0, y0 = cv2.fitLine(pts[keep], cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        dist = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
        n_in = int((dist < p.inlier_px).sum())
        if n_in < p.min_points:
            return None
        return np.array([x0, y0]), np.array([vx, vy]), n_in

    @staticmethod
    def _segment(line, pts):
        p0, d, _ = line
        t = (pts - p0[None, :]) @ d
        return np.stack([p0 + t.min() * d, p0 + t.max() * d])

    @staticmethod
    def _intersect(l1, l2):
        (p1, d1, _), (p2, d2, _) = l1, l2
        cr = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(cr) < 0.2:  # nearly parallel
            return None
        a = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / cr
        return p1 + a * d1

    # ------------------------------------------------------------------
    def _solve(self, c3d_sel, P, Nn, K, R0, t0):
        """Corners must lie on their detected lines (signed point-to-line px)
        + prior toward the RACE pose. Returns (R, t, residual_px) or None."""
        if least_squares is None:
            return None

        rvec0, _ = cv2.Rodrigues(R0)
        sr, st = self.p.sigma_rot_deg, self.p.sigma_trans_mm
        x0 = np.concatenate([rvec0.ravel(), t0])
        n_res = len(P)

        def resid(x):
            R, _ = cv2.Rodrigues(x[:3])
            uv = self._project(c3d_sel @ R.T + x[3:], K)
            r_img = np.einsum("ij,ij->i", uv - P, Nn)
            rv, _ = cv2.Rodrigues(R @ R0.T)
            r_rot = np.degrees(rv.ravel()) / sr
            r_t = (x[3:] - t0) / st
            return np.concatenate([r_img, r_rot, r_t])

        sol = least_squares(resid, x0, method="lm", x_scale="jac", max_nfev=60)
        R, _ = cv2.Rodrigues(sol.x[:3])
        return R, sol.x[3:], resid(sol.x)[:n_res]

    # ------------------------------------------------------------------
    def refine(self, bgr, camera, label, R, t_mm) -> RefineResult:
        p = self.p
        R0 = np.asarray(R, dtype=np.float64)
        t0 = np.asarray(t_mm, dtype=np.float64)
        K = np.asarray(camera, dtype=np.float64)

        vis: Dict[str, Any] = {}
        shifts: List[Optional[float]] = [None] * 4
        support: List[Optional[float]] = [None] * 4
        walls: List[Optional[float]] = [None] * 4

        def mk_info(ok, reason, **kw):
            info = {
                "ok": ok,
                "reason": reason,
                "edge_shift_px": list(
                    shifts
                ),  # + = real edge outside the RACE prediction
                "edge_support": list(support),
                "edge_wall_px": list(walls),  # visible side-wall width per edge
            }
            info.update(kw)
            return info

        def fail(reason, **kw):
            return RefineResult(False, R0, t0, mk_info(False, reason, **kw), **vis)

        box = self.boxes.get(int(label))
        if box is None:
            return fail("no_size_for_label")

        sign, c3d, cos = self._pick_visible_face(box, R0, t0)
        if cos < p.min_face_cos:
            return fail("face_grazing", cos=float(cos))

        c3d_b = self._face_corners_obj(box, -sign)  # opposite (bottom) face
        cam_pts = c3d @ R0.T + t0
        cam_b = c3d_b @ R0.T + t0
        if (cam_pts[:, 2] <= 0).any() or (cam_b[:, 2] <= 0).any():
            return fail("behind_camera")

        pred = self._project(cam_pts, K)
        pred_b = self._project(cam_b, K)
        vis["corners_pred"] = pred
        vis["corners_bottom"] = pred_b

        # ROI around the predicted top + bottom face
        h, w = bgr.shape[:2]
        allp = np.vstack([pred, pred_b])
        pad = int(p.band_px + 6)
        x0 = int(max(0, np.floor(allp[:, 0].min()) - pad))
        y0 = int(max(0, np.floor(allp[:, 1].min()) - pad))
        x1 = int(min(w, np.ceil(allp[:, 0].max()) + pad))
        y1 = int(min(h, np.ceil(allp[:, 1].max()) + pad))
        if x1 - x0 < 30 or y1 - y0 < 30:
            return fail("roi_too_small")

        gray = cv2.cvtColor(bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        k = int(p.blur_ksize) | 1
        gray = cv2.GaussianBlur(gray, (k, k), 0).astype(np.float32)
        offset = np.array([x0, y0], dtype=np.float64)

        # edge i: corner i -> corner (i+1)%4
        centroid = pred.mean(axis=0)
        lines: List[Optional[tuple]] = [None] * 4
        edge_pts: List[np.ndarray] = []
        edge_segs: List[np.ndarray] = []

        for i in range(4):
            j = (i + 1) % 4

            u = pred[j] - pred[i]
            Lu = float(np.linalg.norm(u))
            if Lu < p.min_edge_len_px:
                continue
            u = u / Lu

            n = np.array([-u[1], u[0]])  # make it point OUTWARD
            mid = 0.5 * (pred[i] + pred[j])
            if n @ (mid - centroid) < 0:
                n = -n

            # width of the side wall that is visible on this edge (px, outward)
            mid_b = 0.5 * (pred_b[i] + pred_b[j])
            wall = float(n @ (mid_b - mid))
            walls[i] = wall

            found = self._search_edge_line(
                gray, pred[i] - offset, pred[j] - offset, n, wall, offset
            )
            if found is None:
                continue

            pts, shift, sup = found
            shifts[i] = float(shift)
            support[i] = float(sup)

            line = self._fit_line(pts)
            if line is None:
                continue

            sin = abs(line[1][0] * u[1] - line[1][1] * u[0])
            if np.degrees(np.arcsin(np.clip(sin, 0, 1))) > p.max_edge_angle_deg:
                continue

            lines[i] = line
            edge_pts.append(pts)
            edge_segs.append(self._segment(line, pts))

        vis["edge_pts"] = edge_pts
        vis["edge_segs"] = edge_segs

        valid_idx = [i for i in range(4) if lines[i] is not None]
        if len(valid_idx) < p.min_valid_edges:
            return fail("too_few_edges", n_edges=len(valid_idx))

        # corners from intersections (display + sanity only)
        obs = np.full((4, 2), np.nan)
        for i in range(4):
            l1, l2 = lines[(i - 1) % 4], lines[i]
            if l1 is not None and l2 is not None:
                c = self._intersect(l1, l2)
                if c is not None:
                    obs[i] = c
        vis["corners_obs"] = obs

        def area(q):
            x, y = q[:, 0], q[:, 1]
            return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

        ratio = None
        if np.isfinite(obs).all():
            ratio = float(area(obs) / (area(pred) + 1e-9))
            convex = bool(cv2.isContourConvex(obs.astype(np.float32).reshape(-1, 1, 2)))

            if not (p.area_ratio_min <= ratio <= p.area_ratio_max):
                return fail(
                    "quad_area_ratio",
                    area_ratio=ratio,
                    area_ratio_min=p.area_ratio_min,
                    area_ratio_max=p.area_ratio_max,
                    quad_corners=obs.tolist(),
                    quad_convex=convex,
                )
            if not convex:
                return fail(
                    "quad_non_convex",
                    area_ratio=ratio,
                    quad_corners=obs.tolist(),
                    quad_convex=False,
                )

        # pose solve: corner-on-line residuals
        idxs, Pl, Nl = [], [], []
        for i in valid_idx:
            pl, dl, _ = lines[i]
            nu = np.array([-dl[1], dl[0]])
            for c in (i, (i + 1) % 4):
                idxs.append(c)
                Pl.append(pl)
                Nl.append(nu)

        solved = self._solve(c3d[idxs], np.array(Pl), np.array(Nl), K, R0, t0)
        if solved is None:
            return fail("scipy_missing")
        R1, t1, r_px = solved

        ref = self._project(c3d @ R1.T + t1, K)
        vis["corners_ref"] = ref

        reproj = float(np.abs(r_px).mean())
        dt = float(np.linalg.norm(t1 - t0))
        dang = float(
            np.degrees(np.arccos(np.clip((np.trace(R1 @ R0.T) - 1) / 2, -1, 1)))
        )

        extra: Dict[str, Any] = {
            "reproj_px": reproj,
            "reproj_max_px": float(np.abs(r_px).max()),
            "dt_mm": dt,
            "dang_deg": dang,
            "n_edges": len(valid_idx),
            "edge_points": [
                lines[i][2] if lines[i] is not None else None for i in range(4)
            ],
        }
        if ratio is not None:
            extra["area_ratio"] = ratio
        m = np.isfinite(obs).all(axis=1)
        if m.any():
            extra["corner_shift_px"] = float(
                np.linalg.norm(obs[m] - pred[m], axis=1).mean()
            )

        info = mk_info(True, "ok", **extra)

        if reproj > p.max_reproj_px:
            info.update(ok=False, reason="reproj_too_large")
        elif dt > p.max_trans_mm:
            info.update(ok=False, reason="trans_jump")
        elif dang > p.max_rot_deg:
            info.update(ok=False, reason="rot_jump")

        if not info["ok"]:
            return RefineResult(False, R0, t0, info, **vis)
        return RefineResult(True, R1, t1, info, **vis)

    # ------------------------------------------------------------------
    @staticmethod
    def draw(debug: np.ndarray, res: RefineResult) -> None:
        """orange: RACE bottom face   yellow: RACE top face
        white dots: edge points used  magenta: fitted edge lines
        green dots: line intersections  cyan: refined top face"""

        def ip(q):
            return int(round(float(q[0]))), int(round(float(q[1])))

        def poly(q, color):
            cv2.polylines(
                debug,
                [np.rint(q).astype(np.int32).reshape(-1, 1, 2)],
                True,
                color,
                1,
                cv2.LINE_AA,
            )

        if res.corners_bottom is not None:
            poly(res.corners_bottom, (0, 165, 255))
        if res.corners_pred is not None:
            poly(res.corners_pred, (0, 255, 255))

        for pts in res.edge_pts or []:
            for q in pts:
                cv2.circle(debug, ip(q), 1, (255, 255, 255), -1)

        for seg in res.edge_segs or []:
            cv2.line(debug, ip(seg[0]), ip(seg[1]), (255, 0, 255), 1, cv2.LINE_AA)

        if res.corners_obs is not None:
            for q in res.corners_obs:
                if np.isfinite(q).all():
                    cv2.circle(debug, ip(q), 3, (0, 255, 0), -1)

        if res.corners_ref is not None and res.ok:
            poly(res.corners_ref, (255, 255, 0))
