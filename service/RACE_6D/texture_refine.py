from __future__ import annotations

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TextureParams:
    method: str = "sift"
    px_per_mm: float = 1.5
    margin_mm: float = 40.0
    gate_mm: float = 40.0
    ratio: float = 0.85
    n_features: int = 1500
    sift_contrast: float = 0.03
    clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 4
    sharpen: bool = True
    sharpen_sigma: float = 1.0
    sharpen_amount: float = 1.0
    h_thresh_px: float = 5.0
    pnp_thresh_px: float = 3.5
    min_matches: int = 10
    min_inliers: int = 12
    min_coverage: float = 0.15
    max_reproj_px: float = 2.5
    max_scale_dev: float = 0.25
    max_trans_mm: float = 40.0
    max_rot_deg: float = 10.0
    min_face_cos: float = 0.25
    grid_cols: int = 4
    grid_rows: int = 4
    min_grid_cells: int = 4
    min_quadrants: int = 3
    min_x_span: float = 0.20
    min_y_span: float = 0.20
    sigma_rot_deg: float = 10.0
    sigma_trans_mm: float = 20.0
    debug_matches: bool = False
    debug_orientation: bool = False


@dataclass
class _Target:
    size_mm: np.ndarray
    offset_mm: np.ndarray
    ref_bgr: np.ndarray
    kp: list
    desc: np.ndarray
    pts_local: np.ndarray
    exp_xy: np.ndarray
    canvas_wh: tuple
    A: np.ndarray
    variants: Optional[dict] = None


@dataclass
class TextureResult:
    ok: bool
    R: np.ndarray
    t_mm: np.ndarray
    info: Dict[str, Any]
    corners_pred: Optional[np.ndarray] = None
    corners_ref: Optional[np.ndarray] = None
    pts_good: Optional[np.ndarray] = None
    pts_bad: Optional[np.ndarray] = None


class TexturePoseRefiner:
    def __init__(
        self,
        label_cfgs: Dict[int, dict],
        params: TextureParams,
        base_dir: Path = PROJECT_ROOT,
    ):
        self.p = params
        self.base_dir = Path(base_dir)
        self.last_match_image: Optional[np.ndarray] = None
        self.last_preprocessed_images: Optional[Dict[str, np.ndarray]] = None
        self._make_detector()

        self.targets: Dict[int, _Target] = {}
        for label, item in label_cfgs.items():
            try:
                self.targets[int(label)] = self._load_target(item)
                logger.info(
                    "RGB_REFINE texture label %s: %d reference keypoints",
                    label,
                    len(self.targets[int(label)].kp),
                )
            except Exception as exc:
                logger.error("RGB_REFINE texture label %s disabled: %s", label, exc)

    @property
    def boxes(self) -> Dict[int, _Target]:
        return self.targets

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["TexturePoseRefiner"]:
        if not cfg:
            return None

        tex = cfg.get("TEXTURE") or {}
        if not tex.get("ENABLED", False):
            return None

        valid = {f.name for f in fields(TextureParams)}
        kwargs = {}

        for k, v in (tex.get("PARAMS") or {}).items():
            if k.lower() in valid:
                kwargs[k.lower()] = v
            else:
                logger.warning("Unknown RGB_REFINE.TEXTURE param ignored: %s", k)

        label_cfgs = {
            k: v for k, v in (cfg.get("LABELS") or {}).items() if v.get("REF_IMAGE")
        }

        if not label_cfgs:
            logger.error("RGB_REFINE.TEXTURE enabled but no label has REF_IMAGE")
            return None

        refiner = cls(label_cfgs, TextureParams(**kwargs))

        if not refiner.targets:
            logger.error("RGB_REFINE.TEXTURE enabled but no label could be loaded")
            return None

        return refiner

    def _make_detector(self):
        p = self.p
        method = p.method.lower()

        if method == "sift":
            self.det = cv2.SIFT_create(
                nfeatures=int(p.n_features),
                contrastThreshold=float(p.sift_contrast),
            )
            norm = cv2.NORM_L2

        elif method == "akaze":
            self.det = cv2.AKAZE_create()
            norm = cv2.NORM_HAMMING

        elif method == "orb":
            self.det = cv2.ORB_create(
                nfeatures=int(p.n_features),
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                patchSize=31,
            )
            norm = cv2.NORM_HAMMING

        else:
            raise ValueError(f"Unknown METHOD: {p.method}")

        self.matcher = cv2.BFMatcher(norm)

        self.clahe = (
            cv2.createCLAHE(
                clipLimit=float(p.clahe_clip_limit),
                tileGridSize=(
                    int(p.clahe_tile_grid),
                    int(p.clahe_tile_grid),
                ),
            )
            if p.clahe
            else None
        )

    def _prep(self, gray: np.ndarray) -> np.ndarray:
        out = gray.copy()

        if self.clahe is not None:
            out = self.clahe.apply(out)

        if self.p.sharpen:
            sigma = max(float(self.p.sharpen_sigma), 0.1)
            amount = max(float(self.p.sharpen_amount), 0.0)

            blurred = cv2.GaussianBlur(
                out,
                (0, 0),
                sigmaX=sigma,
                sigmaY=sigma,
            )

            out = cv2.addWeighted(
                out,
                1.0 + amount,
                blurred,
                -amount,
                0,
            )

        return np.clip(out, 0, 255).astype(np.uint8)

    def _load_target(self, item: dict) -> _Target:
        p = self.p
        s = float(p.px_per_mm)

        sz = item["SIZE_M"]
        size_mm = (
            np.array(
                [sz["X"], sz["Y"], sz["Z"]],
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

        path = Path(item["REF_IMAGE"])

        if not path.is_absolute():
            path = self.base_dir / path

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)

        if img is None:
            raise FileNotFoundError(f"Cannot read reference image: {path}")

        rot = int(item.get("REF_ROTATE_DEG", 0)) % 360

        codes = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }

        if rot in codes:
            img = cv2.rotate(img, codes[rot])
        elif rot != 0:
            raise ValueError("REF_ROTATE_DEG must be 0/90/180/270")

        if item.get("REF_FLIP_X", False):
            img = cv2.flip(img, 1)

        h0, w0 = img.shape[:2]

        if abs((w0 / h0) / (size_mm[0] / size_mm[1]) - 1.0) > 0.08:
            logger.warning(
                "Reference aspect %.3f differs from SIZE_M X/Y %.3f by >8%% "
                "(check REF_ROTATE_DEG / cropping)",
                w0 / h0,
                size_mm[0] / size_mm[1],
            )

        Wr = max(8, int(round(size_mm[0] * s)))
        Hr = max(8, int(round(size_mm[1] * s)))

        ref = cv2.resize(
            img,
            (Wr, Hr),
            interpolation=cv2.INTER_AREA if w0 > Wr else cv2.INTER_CUBIC,
        )

        gray = self._prep(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY))
        kp, desc = self.det.detectAndCompute(gray, None)

        n_kp = 0 if desc is None else len(kp)

        if n_kp < 20:
            raise ValueError(
                f"reference image has too few features ({n_kp}); "
                "use a more textured / higher-resolution image "
                "(or lower SIFT_CONTRAST)"
            )

        pts = np.array([k.pt for k in kp], dtype=np.float64)

        srx = Wr / size_mm[0]
        sry = Hr / size_mm[1]

        local = np.stack(
            [
                (pts[:, 0] - (Wr - 1) / 2) / srx,
                ((Hr - 1) / 2 - pts[:, 1]) / sry,
            ],
            axis=1,
        )

        m = float(p.margin_mm)

        Wc = int(round((size_mm[0] + 2 * m) * s))
        Hc = int(round((size_mm[1] + 2 * m) * s))

        exp_xy = np.stack(
            [
                local[:, 0] * s + (Wc - 1) / 2,
                -local[:, 1] * s + (Hc - 1) / 2,
            ],
            axis=1,
        )

        A = np.array(
            [
                [1 / s, 0, -(Wc - 1) / (2 * s)],
                [0, -1 / s, (Hc - 1) / (2 * s)],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        variants = None

        if p.debug_orientation:
            variants = {}

            for name, img_v in (
                ("as_is", ref),
                ("rot180", cv2.rotate(ref, cv2.ROTATE_180)),
                ("flipx", cv2.flip(ref, 1)),
                ("flipy", cv2.flip(ref, 0)),
            ):
                g = self._prep(cv2.cvtColor(img_v, cv2.COLOR_BGR2GRAY))

                kp_v, desc_v = self.det.detectAndCompute(g, None)

                if desc_v is not None and len(kp_v) >= 8:
                    variants[name] = (kp_v, desc_v)

        return _Target(
            size_mm,
            offset_mm,
            ref,
            kp,
            desc,
            local,
            exp_xy,
            (Wc, Hc),
            A,
            variants,
        )

    @staticmethod
    def _project(pts_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
        z = np.maximum(pts_cam[:, 2], 1e-3)

        u = K[0, 0] * pts_cam[:, 0] / z + K[0, 2]
        v = K[1, 1] * pts_cam[:, 1] / z + K[1, 2]

        return np.stack([u, v], axis=1)

    def _errors(self, X, uv, K, R, t):
        return np.linalg.norm(
            self._project(X @ R.T + t, K) - uv,
            axis=1,
        )

    def _solve(self, X, uv, K, R0, t0, init_R=None, init_t=None):
        init_R = R0 if init_R is None else np.asarray(init_R, dtype=np.float64)
        init_t = (
            t0 if init_t is None else np.asarray(init_t, dtype=np.float64).reshape(3)
        )
        rvec_init, _ = cv2.Rodrigues(init_R)

        if least_squares is None:
            ok, rvec, tvec = cv2.solvePnP(
                X.astype(np.float64),
                uv.astype(np.float64),
                K,
                None,
                rvec_init.copy(),
                init_t.reshape(3, 1).copy(),
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                raise RuntimeError("solvePnP_iterative_failed")
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec.ravel()

        sr = max(float(self.p.sigma_rot_deg), 1e-6)
        st = max(float(self.p.sigma_trans_mm), 1e-6)
        x0 = np.concatenate([rvec_init.ravel(), init_t])

        def resid(x):
            R, _ = cv2.Rodrigues(x[:3])
            r_img = (self._project(X @ R.T + x[3:], K) - uv).ravel()
            rv, _ = cv2.Rodrigues(R @ R0.T)
            r_rot = np.degrees(rv.ravel()) / sr
            r_t = (x[3:] - t0) / st
            return np.concatenate([r_img, r_rot, r_t])

        sol = least_squares(
            resid,
            x0,
            method="trf",
            loss="huber",
            f_scale=1.5,
            x_scale="jac",
            max_nfev=80,
        )
        R, _ = cv2.Rodrigues(sol.x[:3])
        return R, sol.x[3:]

    def _mutual_ratio_matches(self, desc_ref, desc_patch):
        p = self.p
        forward = self.matcher.knnMatch(desc_ref, desc_patch, k=2)
        reverse = self.matcher.knnMatch(desc_patch, desc_ref, k=2)

        reverse_best = {}
        for pair in reverse:
            if len(pair) == 2 and pair[0].distance < p.ratio * pair[1].distance:
                reverse_best[pair[0].queryIdx] = pair[0].trainIdx

        good = []
        used_patch = set()
        for pair in forward:
            if len(pair) != 2:
                continue
            m = pair[0]
            if m.distance >= p.ratio * pair[1].distance:
                continue
            if reverse_best.get(m.trainIdx) != m.queryIdx:
                continue
            if m.trainIdx in used_patch:
                continue
            good.append(m)
            used_patch.add(m.trainIdx)
        return good

    def _spatial_stats(self, points, size_x, size_y):
        p = self.p
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        cols = max(1, int(p.grid_cols))
        rows = max(1, int(p.grid_rows))

        if len(pts) == 0:
            return {
                "grid_cells": 0,
                "quadrants": 0,
                "x_span": 0.0,
                "y_span": 0.0,
            }

        x = np.clip((pts[:, 0] / max(size_x, 1e-9) + 0.5), 0.0, 1.0 - 1e-9)
        y = np.clip((pts[:, 1] / max(size_y, 1e-9) + 0.5), 0.0, 1.0 - 1e-9)
        cx = np.floor(x * cols).astype(int)
        cy = np.floor(y * rows).astype(int)
        cells = set(zip(cx.tolist(), cy.tolist()))
        quadrants = set(
            zip((x >= 0.5).astype(int).tolist(), (y >= 0.5).astype(int).tolist())
        )

        return {
            "grid_cells": int(len(cells)),
            "quadrants": int(len(quadrants)),
            "x_span": float((pts[:, 0].max() - pts[:, 0].min()) / max(size_x, 1e-9)),
            "y_span": float((pts[:, 1].max() - pts[:, 1].min()) / max(size_y, 1e-9)),
        }

    @staticmethod
    def _pose_delta(R_a, t_a, R_b, t_b):
        dt_mm = float(
            np.linalg.norm(np.asarray(t_b).reshape(3) - np.asarray(t_a).reshape(3))
        )
        R_delta = (
            np.asarray(R_b, dtype=np.float64) @ np.asarray(R_a, dtype=np.float64).T
        )
        rvec, _ = cv2.Rodrigues(R_delta)
        dang_deg = float(np.degrees(np.linalg.norm(rvec)))
        return dt_mm, dang_deg

    def _solve_ippe(self, X, uv, K, R0, t0, plane_origin):
        X_flat = np.asarray(X, dtype=np.float64) - np.asarray(
            plane_origin, dtype=np.float64
        )
        X_flat[:, 2] = 0.0

        result = cv2.solvePnPGeneric(
            X_flat,
            np.asarray(uv, dtype=np.float64),
            K,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
        if not result or not bool(result[0]):
            raise RuntimeError("ippe_failed")

        rvecs = result[1]
        tvecs = result[2]
        candidates = []
        candidate_info = []

        for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R_plane, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
            t_plane = np.asarray(tvec, dtype=np.float64).reshape(3)
            t_obj = t_plane - R_plane @ np.asarray(plane_origin, dtype=np.float64)
            cam_pts = X @ R_plane.T + t_obj
            positive_depth = bool(np.all(cam_pts[:, 2] > 1e-3))
            reproj_errors = self._errors(X, uv, K, R_plane, t_obj)
            reproj = float(reproj_errors.mean())

            cam_pts_plane = X_flat @ R_plane.T + t_plane
            plane_conversion_error_mm = float(
                np.max(np.linalg.norm(cam_pts_plane - cam_pts, axis=1))
            )
            plane_conversion_reproj_px = float(
                np.max(
                    np.linalg.norm(
                        self._project(cam_pts_plane, K) - self._project(cam_pts, K),
                        axis=1,
                    )
                )
            )
            dt, dang = self._pose_delta(R0, t0, R_plane, t_obj)
            center = R_plane @ np.asarray(plane_origin, dtype=np.float64) + t_obj
            face_cos = float(-R_plane[:, 2] @ center / (np.linalg.norm(center) + 1e-9))
            face_valid = bool(face_cos >= self.p.min_face_cos)
            score = float(
                reproj
                + 0.15 * dang / max(float(self.p.sigma_rot_deg), 1e-6)
                + 0.15 * dt / max(float(self.p.sigma_trans_mm), 1e-6)
            )
            valid = positive_depth and face_valid
            entry = {
                "index": int(index),
                "valid": bool(valid),
                "positive_depth": positive_depth,
                "face_cos": face_cos,
                "reproj_px": reproj,
                "delta_t_mm": dt,
                "delta_R_deg": dang,
                "score": score,
                "t_plane_mm": [float(v) for v in t_plane],
                "t_object_mm": [float(v) for v in t_obj],
                "plane_conversion_max_mm": plane_conversion_error_mm,
                "plane_conversion_max_px": plane_conversion_reproj_px,
            }
            candidate_info.append(entry)
            if valid:
                candidates.append((score, reproj, R_plane, t_obj, index))

        if not candidates:
            raise RuntimeError("ippe_no_valid_candidate")

        candidates.sort(key=lambda item: item[0])
        _, reproj, R_best, t_best, selected_index = candidates[0]
        for entry in candidate_info:
            entry["selected"] = bool(entry["index"] == selected_index)
        return R_best, t_best, reproj, len(candidates), candidate_info, selected_index

    def _h_stats(self, H, c_src, c_dst):
        pts = np.vstack(
            [
                c_src,
                c_src + [10.0, 0.0],
                c_src + [0.0, 10.0],
            ]
        )

        q = cv2.perspectiveTransform(
            pts.reshape(-1, 1, 2).astype(np.float64),
            H,
        ).reshape(-1, 2)

        J = np.column_stack(
            [
                (q[1] - q[0]) / 10.0,
                (q[2] - q[0]) / 10.0,
            ]
        )

        scale = float(np.sqrt(abs(np.linalg.det(J))))

        rot = float(np.degrees(np.arctan2(J[1, 0], J[0, 0])))

        d = q[0] - c_dst
        s = float(self.p.px_per_mm)

        return (
            scale,
            rot,
            [
                float(d[0] / s),
                float(-d[1] / s),
            ],
        )

    def _diagnose(self, tg: _Target, kp_p, desc_p) -> Dict[str, Any]:
        p = self.p

        Wc, Hc = tg.canvas_wh
        h_ref, w_ref = tg.ref_bgr.shape[:2]

        c_src = np.array(
            [
                (w_ref - 1) / 2.0,
                (h_ref - 1) / 2.0,
            ]
        )

        c_dst = np.array(
            [
                (Wc - 1) / 2.0,
                (Hc - 1) / 2.0,
            ]
        )

        out: Dict[str, Any] = {}

        for name, (kp_v, desc_v) in tg.variants.items():
            knn = self.matcher.knnMatch(
                desc_v,
                desc_p,
                k=2,
            )

            good = [
                pr[0]
                for pr in knn
                if len(pr) == 2 and pr[0].distance < p.ratio * pr[1].distance
            ]

            entry: Dict[str, Any] = {
                "matches": int(len(good)),
                "h_inliers": 0,
            }

            if len(good) >= 8:
                src = np.float32([kp_v[m.queryIdx].pt for m in good])

                dst = np.float32([kp_p[m.trainIdx].pt for m in good])

                H, mask = cv2.findHomography(
                    src,
                    dst,
                    cv2.RANSAC,
                    p.h_thresh_px,
                    maxIters=2000,
                    confidence=0.995,
                )

                if H is not None and mask is not None:
                    n_in = int(mask.sum())
                    entry["h_inliers"] = n_in

                    if n_in >= 8:
                        sc, rot, sh = self._h_stats(
                            H,
                            c_src,
                            c_dst,
                        )

                        entry.update(
                            scale=sc,
                            rot_deg=rot,
                            shift_mm=sh,
                        )

            out[name] = entry

        out["_best"] = max(
            tg.variants,
            key=lambda k: out[k]["h_inliers"],
        )

        return out

    def _draw_dbg(self, tg, patch, kp_p, good, keep, inl):
        p = self.p
        s = float(p.px_per_mm)

        Wc, Hc = tg.canvas_wh
        pv = patch.copy()

        m = int(round(p.margin_mm * s))

        cv2.rectangle(
            pv,
            (m, m),
            (Wc - 1 - m, Hc - 1 - m),
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        n = len(good)
        cat = np.zeros(n, dtype=np.int8)

        if keep is None:
            cat[:] = 1
        else:
            cat[keep] = 1

            if inl is not None:
                cat[np.nonzero(keep)[0][inl]] = 2

        colors = {
            0: (0, 0, 255),
            1: (0, 255, 255),
            2: (0, 255, 0),
        }

        out = None

        for c in (0, 1, 2):
            ms = [good[i] for i in range(n) if cat[i] == c]

            if not ms and out is not None:
                continue

            flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS

            if out is not None:
                flags |= cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG

            out = cv2.drawMatches(
                tg.ref_bgr,
                tg.kp,
                pv,
                kp_p,
                ms,
                out,
                matchColor=colors[c],
                singlePointColor=(255, 0, 0),
                flags=flags,
            )

        txt = (
            "ref | patch (green box = predicted face) "
            "red=gated out yellow=gated green=H inlier"
        )

        cv2.putText(
            out,
            txt,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            out,
            txt,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        self.last_match_image = out

    def refine(self, bgr, camera, label, R, t_mm) -> TextureResult:
        p = self.p

        R0 = np.asarray(R, dtype=np.float64)
        t0 = np.asarray(t_mm, dtype=np.float64)
        K = np.asarray(camera, dtype=np.float64)

        self.last_match_image = None
        self.last_preprocessed_images = None

        vis: Dict[str, Any] = {}
        common: Dict[str, Any] = {}

        def fail(reason, **kw):
            info = {
                "ok": False,
                "reason": reason,
            }

            info.update(common)
            info.update(kw)

            return TextureResult(
                False,
                R0,
                t0,
                info,
                **vis,
            )

        tg = self.targets.get(int(label))

        if tg is None:
            return fail("no_ref_for_label")

        sx, sy, sz = tg.size_mm
        hz = sz / 2.0

        c_top = R0 @ (tg.offset_mm + np.array([0.0, 0.0, hz])) + t0

        if c_top[2] <= 0:
            return fail("behind_camera")

        cos = float(-R0[:, 2] @ c_top / (np.linalg.norm(c_top) + 1e-9))

        common["face_cos"] = cos

        if cos < p.min_face_cos:
            return fail("top_face_not_visible")

        H_l2i = K @ np.column_stack(
            [
                R0[:, 0],
                R0[:, 1],
                c_top,
            ]
        )

        local_c = np.array(
            [
                [-sx / 2, -sy / 2],
                [sx / 2, -sy / 2],
                [sx / 2, sy / 2],
                [-sx / 2, sy / 2],
            ]
        )

        vis["corners_pred"] = cv2.perspectiveTransform(
            local_c.reshape(-1, 1, 2),
            H_l2i,
        ).reshape(-1, 2)

        s = float(p.px_per_mm)
        Wc, Hc = tg.canvas_wh

        H_c2i = H_l2i @ tg.A

        patch = cv2.warpPerspective(
            bgr,
            H_c2i,
            (Wc, Hc),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        gray_p = self._prep(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY))

        kp_p, desc_p = self.det.detectAndCompute(
            gray_p,
            None,
        )

        common.update(
            n_ref_kp=int(len(tg.kp)),
            n_patch_kp=int(len(kp_p)),
        )

        if desc_p is None or len(kp_p) < 8:
            return fail("no_features_in_patch")

        if p.debug_matches:
            ref_gray = self._prep(cv2.cvtColor(tg.ref_bgr, cv2.COLOR_BGR2GRAY))

            self.last_preprocessed_images = {
                "reference": ref_gray.copy(),
                "patch": gray_p.copy(),
            }

        if p.debug_orientation and tg.variants:
            common["orient"] = self._diagnose(
                tg,
                kp_p,
                desc_p,
            )

        good = self._mutual_ratio_matches(tg.desc, desc_p)

        common["n_matches"] = int(len(good))

        if len(good) < p.min_matches:
            if p.debug_matches:
                self._draw_dbg(
                    tg,
                    patch,
                    kp_p,
                    good,
                    None,
                    None,
                )

            return fail("too_few_matches")

        q = np.array([m.queryIdx for m in good])

        tr = np.array([m.trainIdx for m in good])

        pat_pts = np.array(
            [kp_p[i].pt for i in tr],
            dtype=np.float64,
        )

        disp = pat_pts - tg.exp_xy[q]

        keep = np.linalg.norm(disp, axis=1) < p.gate_mm * s

        n_gated = int(keep.sum())
        common["n_gated"] = n_gated

        if n_gated < p.min_matches:
            if p.debug_matches:
                self._draw_dbg(
                    tg,
                    patch,
                    kp_p,
                    good,
                    keep,
                    None,
                )

            return fail("too_few_after_gate")

        q = q[keep]
        pat_g = pat_pts[keep]
        local_g = tg.pts_local[q]

        ref_xy = np.array(
            [tg.kp[i].pt for i in q],
            dtype=np.float64,
        )

        H, mask = cv2.findHomography(
            ref_xy,
            pat_g,
            cv2.RANSAC,
            p.h_thresh_px,
            maxIters=2000,
            confidence=0.995,
        )

        if H is None or mask is None:
            if p.debug_matches:
                self._draw_dbg(
                    tg,
                    patch,
                    kp_p,
                    good,
                    keep,
                    None,
                )

            return fail("homography_failed")

        mask = mask.ravel().astype(bool)
        n_h = int(mask.sum())

        if p.debug_matches:
            self._draw_dbg(
                tg,
                patch,
                kp_p,
                good,
                keep,
                mask,
            )

        uv_g = cv2.perspectiveTransform(
            pat_g.reshape(-1, 1, 2),
            H_c2i,
        ).reshape(-1, 2)

        h_ref, w_ref = tg.ref_bgr.shape[:2]

        h_scale, h_rot, shift_mm = self._h_stats(
            H,
            np.array(
                [
                    (w_ref - 1) / 2.0,
                    (h_ref - 1) / 2.0,
                ]
            ),
            np.array(
                [
                    (Wc - 1) / 2.0,
                    (Hc - 1) / 2.0,
                ]
            ),
        )

        common.update(
            n_h_inliers=n_h,
            h_scale=h_scale,
            h_rot_deg=h_rot,
            shift_mm=shift_mm,
        )

        if n_h < p.min_inliers:
            vis["pts_bad"] = uv_g
            return fail("homography_few_inliers")

        X = (
            np.column_stack(
                [
                    local_g[mask],
                    np.full(n_h, hz),
                ]
            )
            + tg.offset_mm
        )

        uv = uv_g[mask]

        plane_origin = tg.offset_mm + np.array([0.0, 0.0, hz], dtype=np.float64)

        uv_race6d = self._project(X @ R0.T + t0, K)
        race6d_errors = np.linalg.norm(uv_race6d - uv, axis=1)
        patch_local_xy = cv2.perspectiveTransform(
            pat_g[mask].reshape(-1, 1, 2).astype(np.float64),
            tg.A,
        ).reshape(-1, 2)
        patch_object_points = (
            np.column_stack([patch_local_xy, np.full(n_h, hz)]) + tg.offset_mm
        )
        uv_from_local_chain = self._project(
            patch_object_points @ R0.T + t0,
            K,
        )
        chain_errors = np.linalg.norm(uv_from_local_chain - uv, axis=1)
        R_orthogonality_error = float(np.linalg.norm(R0.T @ R0 - np.eye(3), ord="fro"))
        R_determinant = float(np.linalg.det(R0))
        top_center_cam = R0 @ plane_origin + t0
        top_center_uv = self._project(top_center_cam.reshape(1, 3), K)[0]

        common["coordinate_diagnostics"] = {
            "object_origin_convention": "object_center; top_plane_z=+SIZE_Z/2",
            "size_mm": [float(sx), float(sy), float(sz)],
            "origin_offset_mm": [float(v) for v in tg.offset_mm],
            "plane_origin_object_mm": [float(v) for v in plane_origin],
            "plane_origin_camera_mm": [float(v) for v in top_center_cam],
            "plane_origin_image_uv": [float(v) for v in top_center_uv],
            "object_points_min_mm": [float(v) for v in X.min(axis=0)],
            "object_points_max_mm": [float(v) for v in X.max(axis=0)],
            "object_points_mean_mm": [float(v) for v in X.mean(axis=0)],
            "race6d_reproj_mean_px": float(race6d_errors.mean()),
            "race6d_reproj_median_px": float(np.median(race6d_errors)),
            "race6d_reproj_p90_px": float(np.percentile(race6d_errors, 90)),
            "race6d_reproj_max_px": float(race6d_errors.max()),
            "local_to_image_chain_mean_px": float(chain_errors.mean()),
            "local_to_image_chain_max_px": float(chain_errors.max()),
            "rotation_orthogonality_error": R_orthogonality_error,
            "rotation_determinant": R_determinant,
            "camera_matrix": K.tolist(),
            "observed_uv_min": [float(v) for v in uv.min(axis=0)],
            "observed_uv_max": [float(v) for v in uv.max(axis=0)],
            "local_xy_min_mm": [float(v) for v in local_g[mask].min(axis=0)],
            "local_xy_max_mm": [float(v) for v in local_g[mask].max(axis=0)],
            "local_xy_mean_mm": [float(v) for v in local_g[mask].mean(axis=0)],
        }
        ippe_R = R0
        ippe_t = t0
        try:
            (
                ippe_R,
                ippe_t,
                ippe_reproj,
                ippe_candidates,
                ippe_candidate_info,
                ippe_selected,
            ) = self._solve_ippe(
                X,
                uv,
                K,
                R0,
                t0,
                plane_origin,
            )
            common.update(
                pnp_initializer="ippe",
                ippe_reproj_px=float(ippe_reproj),
                ippe_candidates=int(ippe_candidates),
                ippe_selected_candidate=int(ippe_selected),
                ippe_candidate_details=ippe_candidate_info,
            )
        except Exception as exc:
            common.update(
                pnp_initializer="race6d_fallback",
                ippe_error=str(exc),
                ippe_candidate_details=[],
            )

        ippe_to_race6d_dt, ippe_to_race6d_dR = self._pose_delta(R0, t0, ippe_R, ippe_t)
        ippe_errors = self._errors(X, uv, K, ippe_R, ippe_t)
        common.update(
            ippe_to_race6d_dt_mm=ippe_to_race6d_dt,
            ippe_to_race6d_dR_deg=ippe_to_race6d_dR,
            race6d_to_ippe_reproj_mean_px=float(ippe_errors.mean()),
            race6d_to_ippe_reproj_median_px=float(np.median(ippe_errors)),
            race6d_to_ippe_reproj_p90_px=float(np.percentile(ippe_errors, 90)),
        )

        R1, t1 = self._solve(
            X,
            uv,
            K,
            R0,
            t0,
            init_R=ippe_R,
            init_t=ippe_t,
        )

        ls1_from_ippe_dt, ls1_from_ippe_dR = self._pose_delta(ippe_R, ippe_t, R1, t1)
        ls1_to_race6d_dt, ls1_to_race6d_dR = self._pose_delta(R0, t0, R1, t1)
        common.update(
            ls1_ippe_to_ls_dt_mm=ls1_from_ippe_dt,
            ls1_ippe_to_ls_dR_deg=ls1_from_ippe_dR,
            ls1_to_race6d_dt_mm=ls1_to_race6d_dt,
            ls1_to_race6d_dR_deg=ls1_to_race6d_dR,
        )

        err = self._errors(
            X,
            uv,
            K,
            R1,
            t1,
        )

        inl = err < p.pnp_thresh_px

        if p.min_inliers <= inl.sum() < n_h:
            R_before_inlier_refine = R1.copy()
            t_before_inlier_refine = t1.copy()
            R1, t1 = self._solve(
                X[inl],
                uv[inl],
                K,
                R0,
                t0,
                init_R=R1,
                init_t=t1,
            )
            inlier_ls_dt, inlier_ls_dR = self._pose_delta(
                R_before_inlier_refine,
                t_before_inlier_refine,
                R1,
                t1,
            )
            common.update(
                inlier_ls_change_dt_mm=inlier_ls_dt,
                inlier_ls_change_dR_deg=inlier_ls_dR,
            )

            err = self._errors(
                X,
                uv,
                K,
                R1,
                t1,
            )

            inl = err < p.pnp_thresh_px

        n_inl = int(inl.sum())

        vis["pts_good"] = uv[inl]

        vis["pts_bad"] = np.vstack(
            [
                uv_g[~mask],
                uv[~inl],
            ]
        )

        if n_inl < p.min_inliers:
            return fail(
                "pnp_few_inliers",
                n_inliers=n_inl,
            )

        reproj = float(err[inl].mean())

        final_from_ippe_dt, final_from_ippe_dR = self._pose_delta(
            ippe_R, ippe_t, R1, t1
        )
        final_to_race6d_dt, final_to_race6d_dR = self._pose_delta(R0, t0, R1, t1)
        common.update(
            final_ippe_to_ls_dt_mm=final_from_ippe_dt,
            final_ippe_to_ls_dR_deg=final_from_ippe_dR,
            final_to_race6d_dt_mm=final_to_race6d_dt,
            final_to_race6d_dR_deg=final_to_race6d_dR,
        )

        local_inliers = local_g[mask][inl]
        spatial = self._spatial_stats(local_inliers, sx, sy)
        common.update(spatial)

        hull = cv2.convexHull(local_inliers.astype(np.float32))

        coverage = float(cv2.contourArea(hull) / (sx * sy))

        dt = float(np.linalg.norm(t1 - t0))

        dang = float(
            np.degrees(
                np.arccos(
                    np.clip(
                        (np.trace(R1 @ R0.T) - 1) / 2,
                        -1,
                        1,
                    )
                )
            )
        )

        obj_c = (
            np.column_stack(
                [
                    local_c,
                    np.full(4, hz),
                ]
            )
            + tg.offset_mm
        )

        vis["corners_ref"] = self._project(
            obj_c @ R1.T + t1,
            K,
        )

        info = dict(common)

        info.update(
            ok=True,
            reason="ok",
            n_inliers=n_inl,
            reproj_px=reproj,
            coverage=coverage,
            dt_mm=dt,
            dang_deg=dang,
        )

        if coverage < p.min_coverage:
            info.update(
                ok=False,
                reason="low_coverage",
            )

        elif (
            spatial["grid_cells"] < int(p.min_grid_cells)
            or spatial["quadrants"] < int(p.min_quadrants)
            or spatial["x_span"] < float(p.min_x_span)
            or spatial["y_span"] < float(p.min_y_span)
        ):
            info.update(
                ok=False,
                reason="low_spatial_spread",
            )

        elif abs(h_scale - 1.0) > p.max_scale_dev:
            info.update(
                ok=False,
                reason="scale_mismatch",
            )

        elif reproj > p.max_reproj_px:
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
            return TextureResult(
                False,
                R0,
                t0,
                info,
                **vis,
            )

        return TextureResult(
            True,
            R1,
            t1,
            info,
            **vis,
        )

    @staticmethod
    def draw(
        debug: np.ndarray,
        res: TextureResult,
        draw_pred: bool = True,
    ) -> None:

        def ip(q):
            return (
                int(round(float(q[0]))),
                int(round(float(q[1]))),
            )

        def poly(q, color):
            cv2.polylines(
                debug,
                [np.rint(q).astype(np.int32).reshape(-1, 1, 2)],
                True,
                color,
                1,
                cv2.LINE_AA,
            )

        if draw_pred and res.corners_pred is not None:
            poly(
                res.corners_pred,
                (0, 255, 255),
            )

        if res.pts_bad is not None:
            for q in res.pts_bad:
                cv2.circle(
                    debug,
                    ip(q),
                    2,
                    (0, 0, 255),
                    -1,
                )

        if res.pts_good is not None:
            for q in res.pts_good:
                cv2.circle(
                    debug,
                    ip(q),
                    2,
                    (0, 255, 0),
                    -1,
                )

        if res.corners_ref is not None and res.ok:
            poly(
                res.corners_ref,
                (255, 255, 0),
            )
