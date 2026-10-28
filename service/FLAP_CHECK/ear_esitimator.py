#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINOV3_ROOT = PROJECT_ROOT / "service" / "dinov3"
WEIGHTS = PROJECT_ROOT / "model" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
GT_PATH = PROJECT_ROOT / "service" / "FLAP_CHECK" / "GT" / "gt.json"

IMAGE_SIZE = 224
PATCH_GRID = 14
MASK_THRESHOLD = 0.25

SIM_THRESHOLD = 0.8
COVERAGE_THRESHOLD = 0.17

PREPROCESS_MODE = "retinex"
RETINEX_SIGMA_RATIO = 0.08
IMAGENET_NORM = False

USE_GT_MASK = True
MASK_ERODE_PX = 0

PRESENT_LABEL = "PRESENT"
ABSENT_LABEL = "ABSENT"

PRESENT_COLOR = (0, 0, 255)
ABSENT_COLOR = (0, 255, 0)
PATCH_COLOR = (255, 255, 0)

SIDES = ("left", "right")

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

        if not WEIGHTS.is_file():
            raise FileNotFoundError(f"DINOv3 weights not found: {WEIGHTS}")

        if not GT_PATH.is_file():
            raise FileNotFoundError(f"GT JSON not found: {GT_PATH}")

        self.model = (
            torch.hub.load(
                str(DINOV3_ROOT),
                "dinov3_vits16",
                source="local",
                weights=str(WEIGHTS),
            )
            .to(self.device)
            .eval()
        )

        self.gt_data = json.loads(GT_PATH.read_text(encoding="utf-8"))

        # groups: 按 id 排序，第一组优先，后面的组作为后备
        self.groups: List[Dict[str, Any]] = []
        self.gt_shape = None

        for group in self._parse_gt_groups(self.gt_data):
            self.groups.append(self._build_group(group))

        if not self.groups:
            raise ValueError(f"No GT groups found in: {GT_PATH}")

        self.sides = [s for s in SIDES if any(s in g["sides"] for g in self.groups)]

        if not self.sides:
            raise ValueError(f"No GT objects found in: {GT_PATH}")

        print(f"[DINOv3] Device: {self.device}")
        print(f"[DINOv3] GT: {GT_PATH}")
        print(f"[DINOv3] GT groups (priority order): {[g['id'] for g in self.groups]}")
        print(f"[DINOv3] Reference sides: {self.sides}")

    # ------------------------------------------------------------------
    # GT loading
    # ------------------------------------------------------------------
    def _parse_gt_groups(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "gts" in data:
            groups = sorted(data["gts"], key=lambda g: g.get("id", 0))

            expected = data.get("expected_gt_count")
            if expected is not None and len(groups) < int(expected):
                print(
                    f"[WARNING] gt.json has {len(groups)} GT group(s), "
                    f"expected {expected}"
                )

            return groups

        # 兼容旧格式：顶层直接是 image + objects
        if "image" in data and "objects" in data:
            return [
                {
                    "id": 0,
                    "image": data["image"],
                    "objects": data["objects"],
                }
            ]

        raise KeyError("gt.json must contain 'gts' (or legacy 'image' + 'objects')")

    def _build_group(self, group: Dict[str, Any]) -> Dict[str, Any]:
        gt_id = int(group.get("id", 0))

        gt_image_path = PROJECT_ROOT / group["image"]
        gt_image = cv2.imread(str(gt_image_path))

        if gt_image is None:
            raise FileNotFoundError(f"Cannot read GT image: {gt_image_path}")

        gt_shape = gt_image.shape[:2]

        if self.gt_shape is None:
            self.gt_shape = gt_shape
        elif gt_shape != self.gt_shape:
            raise ValueError(
                f"GT {gt_id} image size {gt_shape} != first GT image size "
                f"{self.gt_shape}; all GT images must have the same size"
            )

        gt_proc = self._preprocess(gt_image)

        sides: Dict[str, Dict[str, Any]] = {}

        for obj in group["objects"]:
            side = obj["side"]

            if side not in SIDES:
                raise ValueError(f"Unsupported GT side: {side}")

            if "bbox" not in obj:
                raise KeyError(f"Missing 'bbox' in GT {gt_id} object: {side}")

            bbox = self._validate_bbox(obj["bbox"], gt_shape)
            gt_roi = self._crop_bbox(gt_proc, bbox)

            roi_mask = self._load_roi_mask(
                obj=obj,
                bbox=bbox,
                roi_shape=gt_roi.shape[:2],
                gt_shape=gt_shape,
            )

            patch_mask = self._get_patch_mask(roi_mask)

            if not patch_mask.any():
                raise ValueError(
                    f"Mask does not contain any valid DINOv3 patches: "
                    f"GT {gt_id} {side}"
                )

            patch_index = torch.from_numpy(np.flatnonzero(patch_mask.reshape(-1))).to(
                self.device
            )

            features = self._extract_features(gt_roi)

            sides[side] = {
                "bbox": bbox,
                "roi_mask": roi_mask,
                "patch_index": patch_index,
                "feature": features[patch_index],
            }

        if not sides:
            raise ValueError(f"GT {gt_id} has no objects")

        return {
            "id": gt_id,
            "image": group["image"],
            "sides": sides,
        }

    def _validate_bbox(self, bbox, image_shape):
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox: expected [x1, y1, x2, y2], got {bbox}")

        values = np.asarray(bbox, dtype=np.float64)

        if not np.isfinite(values).all():
            raise ValueError(f"BBox contains non-finite coordinates: {bbox}")

        if not np.equal(values, np.round(values)).all():
            raise ValueError(f"BBox coordinates must be integers: {bbox}")

        x1, y1, x2, y2 = map(int, values)
        h, w = image_shape[:2]

        if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
            raise ValueError(
                f"BBox {bbox} is outside image bounds: width={w}, height={h}"
            )

        return [x1, y1, x2, y2]

    def _load_roi_mask(
        self,
        obj: Dict[str, Any],
        bbox,
        roi_shape,
        gt_shape,
    ) -> np.ndarray:
        mask_info = obj.get("mask") or {}
        mask_file = mask_info.get("file")

        if USE_GT_MASK and mask_file:
            mask_path = PROJECT_ROOT / mask_file
            full = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if full is None:
                raise FileNotFoundError(f"Cannot read GT mask: {mask_path}")

            if full.shape != tuple(gt_shape):
                raise ValueError(
                    f"Mask size {full.shape} != GT image size "
                    f"{tuple(gt_shape)}: {mask_path}"
                )

            mask = self._crop_bbox(full, bbox)
            mask = np.where(mask > 127, 255, 0).astype(np.uint8)

        else:
            mask = np.full(roi_shape, 255, dtype=np.uint8)

        if mask.shape != tuple(roi_shape):
            raise ValueError(f"ROI mask size {mask.shape} != expected size {roi_shape}")

        if MASK_ERODE_PX > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * MASK_ERODE_PX + 1,) * 2,
            )
            mask = cv2.erode(mask, kernel)

        if not mask.any():
            raise ValueError(f"Empty GT mask for side: {obj['side']}")

        return mask

    # ------------------------------------------------------------------
    # Image processing
    # ------------------------------------------------------------------
    def _preprocess(self, bgr: np.ndarray) -> np.ndarray:
        if PREPROCESS_MODE == "none":
            return bgr.copy()

        l, a, b = cv2.split(cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB))

        if PREPROCESS_MODE == "clahe":
            l = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8),
            ).apply(l)

        elif PREPROCESS_MODE == "retinex":
            l_f = l.astype(np.float32) + 1.0

            sigma = max(bgr.shape[:2]) * RETINEX_SIGMA_RATIO
            illum = cv2.GaussianBlur(l_f, (0, 0), sigma)

            r = np.log(l_f) - np.log(np.maximum(illum, 1e-6))
            r = (r - r.mean()) / (r.std() + 1e-6)

            l = np.clip(r * 40.0 + 128.0, 0, 255).astype(np.uint8)

        elif PREPROCESS_MODE == "chroma":
            l = np.full_like(l, 128)

        else:
            raise ValueError(f"Unknown PREPROCESS_MODE: {PREPROCESS_MODE}")

        return cv2.cvtColor(
            cv2.merge([l, a, b]),
            cv2.COLOR_LAB2BGR,
        )

    def _crop_bbox(self, image: np.ndarray, bbox):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid bbox: {bbox}")

        return image[y1:y2, x1:x2]

    def _extract_features(self, image: np.ndarray) -> torch.Tensor:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        x = torch.from_numpy(image).permute(2, 0, 1).float().div(255.0)

        if IMAGENET_NORM:
            x = (x - MEAN) / STD

        x = x.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            feat = self.model.forward_features(x)["x_norm_patchtokens"][0]

        return F.normalize(feat, dim=-1)

    def _get_patch_mask(self, mask: np.ndarray) -> np.ndarray:
        small = cv2.resize(
            mask.astype(np.uint8),
            (PATCH_GRID, PATCH_GRID),
            interpolation=cv2.INTER_AREA,
        )

        return small >= int(MASK_THRESHOLD * 255)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(
        self,
        ref_feat: torch.Tensor,
        ref_index: torch.Tensor,
        input_feat: torch.Tensor,
    ):
        best, best_idx = (ref_feat @ input_feat.T).max(dim=1)
        passed = best >= self.sim_threshold

        coverage = float(passed.float().mean())
        is_present = coverage >= self.coverage_threshold

        n_patch = PATCH_GRID * PATCH_GRID

        pass_mask = torch.zeros(
            n_patch,
            dtype=torch.bool,
            device=self.device,
        )
        pass_mask[best_idx[passed]] = True

        similarity = torch.zeros(
            n_patch,
            device=self.device,
        )
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
            "pass_mask": pass_mask.reshape(
                PATCH_GRID,
                PATCH_GRID,
            )
            .cpu()
            .numpy(),
            "similarity_map": similarity.reshape(
                PATCH_GRID,
                PATCH_GRID,
            )
            .cpu()
            .numpy(),
        }

    def _evaluate_side(self, proc: np.ndarray, side: str):
        """
        按 GT 优先级依次评估某一侧：
        第一组 PRESENT 就直接采用；否则用下一组；全部 ABSENT 时报告 coverage 最高的一组。
        """
        chosen = None
        best = None
        tried: List[int] = []

        for group in self.groups:
            ref = group["sides"].get(side)

            if ref is None:
                continue

            input_roi = self._crop_bbox(proc, ref["bbox"])
            input_feat = self._extract_features(input_roi)

            evaluation = self._evaluate(
                ref["feature"],
                ref["patch_index"],
                input_feat,
            )
            evaluation["gt_id"] = group["id"]
            tried.append(group["id"])

            if best is None or evaluation["coverage"] > best[0]["coverage"]:
                best = (evaluation, ref)

            if evaluation["present"]:
                chosen = (evaluation, ref)
                break

        if chosen is None:
            chosen = best

        if chosen is None:
            return None, None

        evaluation, ref = chosen
        evaluation["tried_gt_ids"] = tried
        evaluation["fallback_used"] = bool(
            evaluation["present"] and evaluation["gt_id"] != self.groups[0]["id"]
        )

        return evaluation, ref

    # ------------------------------------------------------------------
    # Debug drawing
    # ------------------------------------------------------------------
    def _draw_debug(
        self,
        image: np.ndarray,
        side: str,
        bbox,
        roi_mask: np.ndarray,
        evaluation: Dict[str, Any],
    ):
        x1, y1, x2, y2 = map(int, bbox)

        color = PRESENT_COLOR if evaluation["present"] else ABSENT_COLOR

        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)

        overlay = image.copy()

        for row, col in zip(*np.where(evaluation["pass_mask"])):
            p1 = (
                x1 + int(col * roi_w / PATCH_GRID),
                y1 + int(row * roi_h / PATCH_GRID),
            )
            p2 = (
                x1 + int((col + 1) * roi_w / PATCH_GRID),
                y1 + int((row + 1) * roi_h / PATCH_GRID),
            )

            cv2.rectangle(
                overlay,
                p1,
                p2,
                PATCH_COLOR,
                -1,
            )

        debug = cv2.addWeighted(
            overlay,
            0.28,
            image,
            0.72,
            0,
        )

        contours, _ = cv2.findContours(
            roi_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            debug,
            contours,
            -1,
            color,
            2,
            offset=(x1, y1),
        )

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        text = (
            f"{side.upper()}[GT{evaluation['gt_id']}]: {evaluation['label']} "
            f"cov={evaluation['coverage_percent']:.1f}%"
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2

        tw, th = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )[0]

        text_y = max(th + 10, y1)
        bg_x1 = max(0, x1)
        bg_x2 = min(debug.shape[1], bg_x1 + tw + 10)
        bg_y1 = max(0, text_y - th - 8)
        bg_y2 = min(debug.shape[0], text_y + 4)

        cv2.rectangle(
            debug,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1,
        )

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

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, rgb: np.ndarray) -> Dict[str, Any]:
        if rgb is None or not isinstance(rgb, np.ndarray):
            raise ValueError("rgb must be a numpy.ndarray")

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb must have shape (H, W, 3)")

        if rgb.shape[:2] != self.gt_shape:
            raise ValueError(
                f"Input size {rgb.shape[:2]} != GT image size {self.gt_shape}"
            )

        proc = self._preprocess(rgb)
        debug_image = rgb.copy()

        result = {
            "left": None,
            "right": None,
            "left_evaluation": None,
            "right_evaluation": None,
            "debug_image": debug_image,
        }

        for side in self.sides:
            evaluation, ref = self._evaluate_side(proc, side)

            if evaluation is None:
                continue

            result[side] = bool(evaluation["absent"])
            result[f"{side}_evaluation"] = evaluation

            debug_image = self._draw_debug(
                debug_image,
                side,
                ref["bbox"],
                ref["roi_mask"],
                evaluation,
            )

        result["debug_image"] = debug_image
        return result


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
    print(f"SIM_THRESHOLD: {estimator.sim_threshold}")
    print(f"COVERAGE_THRESHOLD: {estimator.coverage_threshold}")

    for side in SIDES:
        item = result[f"{side}_evaluation"]

        if item is None:
            print(f"{side}: No reference")
            continue

        print(
            f"{side}: {item['label']} | absent={result[side]} | "
            f"GT{item['gt_id']} (tried={item['tried_gt_ids']}, "
            f"fallback={item['fallback_used']}) | "
            f"coverage={item['coverage_percent']:.1f}% | "
            f"mean={item['mean_similarity']:.4f} | "
            f"p10={item['p10_similarity']:.4f}"
        )

    debug_path = PROJECT_ROOT / "dino_test" / "debug_result.png"
    debug_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(debug_path), result["debug_image"]):
        raise RuntimeError(f"Failed to save debug image: {debug_path}")

    print(f"Debug image: {debug_path}")
