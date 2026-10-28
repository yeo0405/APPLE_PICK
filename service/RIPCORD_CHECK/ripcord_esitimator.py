#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINOV3_ROOT = PROJECT_ROOT / "service" / "dinov3"
WEIGHTS = PROJECT_ROOT / "model" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
GT_PATH = PROJECT_ROOT / "service" / "RIPCORD_CHECK" / "GT" / "gt.json"

TARGET_NAME = "ripcord"

IMAGE_SIZE = 224
PATCH_SIZE = 16  # vits16
PATCH_GRID = IMAGE_SIZE // PATCH_SIZE  # 14, 参考侧网格
MASK_THRESHOLD = 0.25

SIM_THRESHOLD = 0.8
COVERAGE_THRESHOLD = 0.8

SEARCH_MARGIN_RATIO = 0.2
KEEP_SCALE = True
MAX_SEARCH_GRID = 32
PREPROCESS_MODE = "retinex"  # none | clahe | retinex | chroma
RETINEX_SIGMA_RATIO = 0.08
IMAGENET_NORM = False

USE_GT_MASK = True
MASK_ERODE_PX = 0

PRESENT_LABEL = "PRESENT"
ABSENT_LABEL = "ABSENT"

PRESENT_COLOR = (0, 255, 0)
ABSENT_COLOR = (0, 0, 255)
PATCH_COLOR = (255, 255, 0)
SEARCH_COLOR = (255, 128, 0)

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class DINOv3Estimator:
    def __init__(
        self,
        sim_threshold: float = SIM_THRESHOLD,
        coverage_threshold: float = COVERAGE_THRESHOLD,
        present_label: str = PRESENT_LABEL,
        absent_label: str = ABSENT_LABEL,
    ):
        self.sim_threshold = float(sim_threshold)
        self.coverage_threshold = float(coverage_threshold)
        self.present_label = present_label
        self.absent_label = absent_label
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = (
            torch.hub.load(
                str(DINOV3_ROOT), "dinov3_vits16", source="local", weights=str(WEIGHTS)
            )
            .to(self.device)
            .eval()
        )

        self.gt_data = json.loads(GT_PATH.read_text())
        gt_image_path = PROJECT_ROOT / self.gt_data["image"]
        gt_image = cv2.imread(str(gt_image_path))
        if gt_image is None:
            raise FileNotFoundError(f"Cannot read GT image: {gt_image_path}")
        self.gt_shape = gt_image.shape[:2]
        gt_proc = self._preprocess(gt_image)

        self.reference: Dict[str, Dict[str, Any]] = {}
        for obj in self.gt_data["objects"]:
            name = obj["side"]
            if name != TARGET_NAME:
                continue
            if "polygon" not in obj:
                raise KeyError(f"Missing 'polygon' in GT object: {name}")
            polygon = np.asarray(obj["polygon"], dtype=np.float32)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError(f"Invalid polygon for {name}: {obj['polygon']}")

            bbox = self._polygon_to_bbox(polygon, gt_proc.shape)
            gt_roi = self._crop_bbox(gt_proc, bbox)
            roi_mask = self._load_roi_mask(obj, polygon, bbox, gt_roi.shape[:2])
            patch_mask = self._get_patch_mask(roi_mask)
            if not patch_mask.any():
                raise ValueError(
                    f"Mask does not contain any valid DINOv3 patches: {name}"
                )

            patch_index = torch.from_numpy(np.flatnonzero(patch_mask.reshape(-1))).to(
                self.device
            )
            ref_feat, _ = self._extract_features(gt_roi)
            self.reference[name] = {
                "polygon": polygon.tolist(),
                "bbox": bbox,
                "roi_mask": roi_mask,
                "patch_index": patch_index,
                "feature": ref_feat[patch_index],
            }

        if TARGET_NAME not in self.reference:
            raise KeyError(f"GT object '{TARGET_NAME}' not found in: {GT_PATH}")

    # ------------------------------------------------------------------ mask
    def _load_roi_mask(
        self, obj: Dict[str, Any], polygon: np.ndarray, bbox, roi_shape
    ) -> np.ndarray:
        mask_file = (obj.get("mask") or {}).get("file")

        if USE_GT_MASK and mask_file:
            mask_path = PROJECT_ROOT / mask_file
            full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if full is None:
                raise FileNotFoundError(f"Cannot read GT mask: {mask_path}")
            if full.shape != self.gt_shape:
                raise ValueError(
                    f"Mask size {full.shape} != GT image size {self.gt_shape}: {mask_path}"
                )
            mask = np.where(self._crop_bbox(full, bbox) > 127, 255, 0).astype(np.uint8)
        else:
            mask = self._polygon_mask(
                polygon - np.array(bbox[:2], dtype=np.float32), roi_shape
            )

        if MASK_ERODE_PX > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * MASK_ERODE_PX + 1,) * 2
            )
            mask = cv2.erode(mask, kernel)
        if not mask.any():
            raise ValueError(f"Empty GT mask for: {obj['side']}")
        return mask

    # ------------------------------------------------------------ preprocess
    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        if PREPROCESS_MODE == "none":
            return bgr
        l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))
        if PREPROCESS_MODE == "clahe":
            l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        elif PREPROCESS_MODE == "retinex":
            l_f = l.astype(np.float32) + 1.0
            illum = cv2.GaussianBlur(
                l_f, (0, 0), max(bgr.shape[:2]) * RETINEX_SIGMA_RATIO
            )
            r = np.log(l_f) - np.log(illum)
            r = (r - r.mean()) / (r.std() + 1e-6)
            l = np.clip(r * 40.0 + 128.0, 0, 255).astype(np.uint8)
        elif PREPROCESS_MODE == "chroma":
            l = np.full_like(l, 128)
        else:
            raise ValueError(f"Unknown PREPROCESS_MODE: {PREPROCESS_MODE}")
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------ bbox
    def _polygon_to_bbox(self, polygon: np.ndarray, image_shape):
        h, w = image_shape[:2]
        x1 = int(np.floor(polygon[:, 0].min()))
        y1 = int(np.floor(polygon[:, 1].min()))
        x2 = int(np.ceil(polygon[:, 0].max()))
        y2 = int(np.ceil(polygon[:, 1].max()))
        x1, x2 = max(0, min(x1, w - 1)), max(0, min(x2, w - 1))
        y1, y2 = max(0, min(y1, h - 1)), max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid polygon bbox: {polygon.tolist()}")
        return [x1, y1, x2, y2]

    def _expand_bbox(self, bbox, shape, ratio: float):
        h, w = shape[:2]
        x1, y1, x2, y2 = bbox
        dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
        return [
            max(0, int(np.floor(x1 - dx))),
            max(0, int(np.floor(y1 - dy))),
            min(w, int(np.ceil(x2 + dx))),
            min(h, int(np.ceil(y2 + dy))),
        ]

    def _crop_bbox(self, image: np.ndarray, bbox):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {bbox}")
        return image[y1:y2, x1:x2]

    def _polygon_mask(self, polygon: np.ndarray, shape) -> np.ndarray:
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 255)
        return mask

    # --------------------------------------------------------------- feature
    def _search_grid(self, search_bbox, ref_bbox) -> Tuple[int, int]:
        """返回搜索框的 patch 网格 (gh, gw)。"""
        if not KEEP_SCALE:
            return PATCH_GRID, PATCH_GRID
        sx1, sy1, sx2, sy2 = search_bbox
        bx1, by1, bx2, by2 = ref_bbox
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        # 参考:bw 像素 -> PATCH_GRID 个 patch;搜索框按同一比例换算
        gw = int(round((sx2 - sx1) / bw * PATCH_GRID))
        gh = int(round((sy2 - sy1) / bh * PATCH_GRID))
        gw = int(np.clip(gw, 1, MAX_SEARCH_GRID))
        gh = int(np.clip(gh, 1, MAX_SEARCH_GRID))
        return gh, gw

    def _extract_features(
        self, image: np.ndarray, grid: Tuple[int, int] | None = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        gh, gw = grid if grid is not None else (PATCH_GRID, PATCH_GRID)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image, (gw * PATCH_SIZE, gh * PATCH_SIZE), interpolation=cv2.INTER_AREA
        )
        x = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)
        if IMAGENET_NORM:
            x = (x - MEAN) / STD
        x = x.unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feat = self.model.forward_features(x)["x_norm_patchtokens"][0]
        return F.normalize(feat, dim=-1), (gh, gw)

    def _get_patch_mask(self, mask: np.ndarray) -> np.ndarray:
        small = cv2.resize(
            mask.astype(np.uint8),
            (PATCH_GRID, PATCH_GRID),
            interpolation=cv2.INTER_AREA,
        )
        return small >= int(MASK_THRESHOLD * 255)

    # -------------------------------------------------------------- evaluate
    def _evaluate(
        self,
        ref_feat: torch.Tensor,
        ref_index: torch.Tensor,
        input_feat: torch.Tensor,
        input_grid: Tuple[int, int],
    ):
        gh, gw = input_grid
        best, best_idx = (ref_feat @ input_feat.T).max(dim=1)
        passed = best >= self.sim_threshold
        coverage = float(passed.float().mean())
        is_present = coverage >= self.coverage_threshold

        # pass_mask: 搜索框网格 (gh, gw)
        pass_mask = torch.zeros(gh * gw, dtype=torch.bool, device=self.device)
        pass_mask[best_idx[passed]] = True

        # similarity_map: 参考网格 (14, 14),与参考 patch 位置对应
        similarity = torch.zeros(PATCH_GRID * PATCH_GRID, device=self.device)
        similarity[ref_index] = best
        scores_np = best.detach().cpu().numpy()

        return {
            "label": self.present_label if is_present else self.absent_label,
            "present": is_present,
            "absent": not is_present,
            "coverage": coverage,
            "coverage_percent": coverage * 100.0,
            "mean_similarity": float(best.mean()),
            "median_similarity": float(best.median()),
            "p10_similarity": float(np.percentile(scores_np, 10)),
            "similarity_threshold": self.sim_threshold,
            "coverage_threshold": self.coverage_threshold,
            "patch_count": int(len(scores_np)),
            "pass_patch_count": int(passed.sum().item()),
            "search_grid": [gh, gw],
            "pass_mask": pass_mask.reshape(gh, gw).cpu().numpy(),
            "similarity_map": similarity.reshape(PATCH_GRID, PATCH_GRID).cpu().numpy(),
        }

    # ----------------------------------------------------------------- debug
    def _draw_debug(
        self,
        image: np.ndarray,
        name: str,
        polygon,
        bbox,
        roi_mask,
        evaluation: Dict[str, Any],
        search_bbox,
    ):
        polygon = np.asarray(polygon, dtype=np.int32)
        x1, y1, x2, y2 = map(int, bbox)
        sx1, sy1, sx2, sy2 = map(int, search_bbox)
        color = PRESENT_COLOR if evaluation["present"] else ABSENT_COLOR

        pass_mask = evaluation["pass_mask"]
        gh, gw = pass_mask.shape
        sw, sh = max(1, sx2 - sx1), max(1, sy2 - sy1)

        overlay = image.copy()
        for row, col in zip(*np.where(pass_mask)):
            p1 = (sx1 + int(col * sw / gw), sy1 + int(row * sh / gh))
            p2 = (sx1 + int((col + 1) * sw / gw), sy1 + int((row + 1) * sh / gh))
            cv2.rectangle(overlay, p1, p2, PATCH_COLOR, -1)
        debug = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)

        # 搜索框 + GT 位置
        cv2.rectangle(debug, (sx1, sy1), (sx2, sy2), SEARCH_COLOR, 1)
        contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(debug, contours, -1, color, 2, offset=(x1, y1))
        cv2.rectangle(debug, (x1, y1), (x2, y2), color, 1)

        text = f"{name.upper()}: {evaluation['label']}  cov={evaluation['coverage_percent']:.1f}%"
        font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
        tw, th = cv2.getTextSize(text, font, scale, thickness)[0]
        text_y = max(th + 10, int(polygon[:, 1].min()))
        bg_x1 = max(0, int(polygon[:, 0].min()))
        bg_x2 = min(debug.shape[1], bg_x1 + tw + 10)
        bg_y1 = max(0, text_y - th - 8)
        bg_y2 = min(debug.shape[0], text_y + 4)
        cv2.rectangle(debug, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            debug,
            text,
            (bg_x1 + 5, text_y - 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        return debug

    # --------------------------------------------------------------- predict
    def predict(self, rgb: np.ndarray) -> Dict[str, Any]:
        if rgb is None or not isinstance(rgb, np.ndarray):
            raise ValueError("rgb must be a numpy.ndarray")
        if rgb.shape[:2] != self.gt_shape:
            raise ValueError(
                f"Input size {rgb.shape[:2]} != GT image size {self.gt_shape}"
            )

        ref = self.reference[TARGET_NAME]
        proc = self._preprocess(rgb)

        search_bbox = self._expand_bbox(ref["bbox"], proc.shape, SEARCH_MARGIN_RATIO)
        grid = self._search_grid(search_bbox, ref["bbox"])
        input_feat, grid = self._extract_features(
            self._crop_bbox(proc, search_bbox), grid
        )

        evaluation = self._evaluate(
            ref["feature"], ref["patch_index"], input_feat, grid
        )
        debug_image = self._draw_debug(
            rgb.copy(),
            TARGET_NAME,
            ref["polygon"],
            ref["bbox"],
            ref["roi_mask"],
            evaluation,
            search_bbox,
        )

        return {
            TARGET_NAME: bool(evaluation["absent"]),
            f"{TARGET_NAME}_evaluation": evaluation,
            "search_bbox": search_bbox,
            "debug_image": debug_image,
        }


if __name__ == "__main__":
    import sys

    image_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT / "dino_test" / "hard.png"
    )
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    estimator = DINOv3Estimator()
    result = estimator.predict(image)

    print(f"Device: {estimator.device}")
    print(f"PREPROCESS_MODE: {PREPROCESS_MODE}")
    print(f"USE_GT_MASK: {USE_GT_MASK}")
    print(f"SEARCH_MARGIN_RATIO: {SEARCH_MARGIN_RATIO}  KEEP_SCALE: {KEEP_SCALE}")
    print(f"SIM_THRESHOLD: {estimator.sim_threshold}")
    print(f"COVERAGE_THRESHOLD: {estimator.coverage_threshold}")

    item = result[f"{TARGET_NAME}_evaluation"]
    print(
        f"{TARGET_NAME}: {item['label']} | absent={result[TARGET_NAME]} | "
        f"coverage={item['coverage_percent']:.1f}% | "
        f"mean={item['mean_similarity']:.4f} | p10={item['p10_similarity']:.4f} | "
        f"search_grid={item['search_grid']}"
    )

    debug_path = PROJECT_ROOT / "dino_test" / "debug_result.png"
    cv2.imwrite(str(debug_path), result["debug_image"])
    print(f"Debug image: {debug_path}")
