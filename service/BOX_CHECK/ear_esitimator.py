#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DINOV3_ROOT = PROJECT_ROOT / "service" / "dinov3"
WEIGHTS = PROJECT_ROOT / "model" / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
GT_PATH = PROJECT_ROOT / "service" / "BOX_CHECK" / "GT" / "gt.json"

IMAGE_SIZE = 224
PATCH_GRID = 14
MASK_THRESHOLD = 0.25

SIM_THRESHOLD = 0.75
COVERAGE_THRESHOLD = 0.4

PRESENT_LABEL = "PRESENT"
ABSENT_LABEL = "ABSENT"

PRESENT_COLOR = (0, 0, 255)
ABSENT_COLOR = (0, 255, 0)
PATCH_COLOR = (255, 255, 0)


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

        self.model = torch.hub.load(
            str(DINOV3_ROOT),
            "dinov3_vits16",
            source="local",
            weights=str(WEIGHTS),
        ).to(self.device).eval()

        self.gt_data = json.loads(GT_PATH.read_text())
        gt_image_path = PROJECT_ROOT / self.gt_data["image"]
        gt_image = cv2.imread(str(gt_image_path))

        if gt_image is None:
            raise FileNotFoundError(f"Cannot read GT image: {gt_image_path}")

        self.reference: Dict[str, Dict[str, Any]] = {}

        for obj in self.gt_data["objects"]:
            side = obj["side"]
            bbox = obj["bbox"]
            gt_roi = self._crop_bbox(gt_image, bbox)

            mask_path = PROJECT_ROOT / obj["mask"]["file"]
            full_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            if full_mask is None:
                raise FileNotFoundError(f"Cannot read GT mask: {mask_path}")

            mask_roi = self._crop_bbox(full_mask, bbox)
            ear_mask = self._get_patch_mask(mask_roi)
            gt_feat = self._extract_features(gt_roi)

            self.reference[side] = {
                "bbox": bbox,
                "feature": gt_feat,
                "mask": ear_mask,
            }

    def _crop_bbox(self, image: np.ndarray, bbox):
        h, w = image.shape[:2]
        x1, y1, x2, y2 = map(int, bbox)
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

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
        return small >= MASK_THRESHOLD

    def _evaluate(
        self,
        gt_feat: torch.Tensor,
        input_feat: torch.Tensor,
        ear_mask: np.ndarray,
    ):
        gt_feat = gt_feat.reshape(PATCH_GRID, PATCH_GRID, -1)
        input_feat = input_feat.reshape(PATCH_GRID, PATCH_GRID, -1)

        similarity = (gt_feat * input_feat).sum(dim=-1)
        ear_mask_t = torch.from_numpy(ear_mask).to(self.device)
        scores = similarity[ear_mask_t]
        scores_np = scores.detach().cpu().numpy()

        pass_mask = (similarity >= self.sim_threshold) & ear_mask_t
        coverage = float(pass_mask[ear_mask_t].float().mean())
        mean_similarity = float(scores.mean())
        median_similarity = float(scores.median())
        p10_similarity = float(np.percentile(scores_np, 10))

        is_present = coverage >= self.coverage_threshold
        label = self.present_label if is_present else self.absent_label

        return {
            "label": label,
            "present": is_present,
            "coverage": coverage,
            "coverage_percent": coverage * 100.0,
            "mean_similarity": mean_similarity,
            "median_similarity": median_similarity,
            "p10_similarity": p10_similarity,
            "similarity_threshold": self.sim_threshold,
            "coverage_threshold": self.coverage_threshold,
            "patch_count": int(len(scores_np)),
            "pass_patch_count": int(pass_mask.sum().item()),
            "pass_mask": pass_mask.detach().cpu().numpy(),
            "similarity_map": similarity.detach().cpu().numpy(),
        }

    def _draw_debug(
        self,
        image: np.ndarray,
        side: str,
        bbox,
        evaluation: Dict[str, Any],
    ):
        x1, y1, x2, y2 = map(int, bbox)
        color = PRESENT_COLOR if evaluation["present"] else ABSENT_COLOR

        debug = image.copy()
        overlay = debug.copy()

        roi_w = max(1, x2 - x1)
        roi_h = max(1, y2 - y1)

        pass_mask = evaluation["pass_mask"]

        for row, col in zip(*np.where(pass_mask)):
            px1 = x1 + int(col * roi_w / PATCH_GRID)
            py1 = y1 + int(row * roi_h / PATCH_GRID)
            px2 = x1 + int((col + 1) * roi_w / PATCH_GRID)
            py2 = y1 + int((row + 1) * roi_h / PATCH_GRID)

            cv2.rectangle(
                overlay,
                (px1, py1),
                (px2, py2),
                PATCH_COLOR,
                -1,
            )

        debug = cv2.addWeighted(overlay, 0.28, debug, 0.72, 0)

        cv2.rectangle(
            debug,
            (x1, y1),
            (x2, y2),
            color,
            3,
        )

        label = evaluation["label"]
        coverage = evaluation["coverage_percent"]
        text = f"{side.upper()}: {label}  cov={coverage:.1f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2

        (tw, th), baseline = cv2.getTextSize(
            text,
            font,
            scale,
            thickness,
        )

        text_y = max(th + baseline + 4, y1)
        bg_y1 = max(0, text_y - th - baseline - 6)
        bg_y2 = min(debug.shape[0], text_y + 3)
        bg_x2 = min(debug.shape[1], x1 + tw + 10)

        cv2.rectangle(
            debug,
            (x1, bg_y1),
            (bg_x2, bg_y2),
            color,
            -1,
        )

        text_color = (255, 255, 255)

        cv2.putText(
            debug,
            text,
            (x1 + 5, text_y - 2),
            font,
            scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        return debug

    def predict(self, rgb: np.ndarray) -> Dict[str, Any]:
        if rgb is None or not isinstance(rgb, np.ndarray):
            raise ValueError("rgb must be a numpy.ndarray")

        debug_image = rgb.copy()

        result = {
            "left": None,
            "right": None,
            "debug_image": debug_image,
        }

        for side, ref in self.reference.items():
            input_roi = self._crop_bbox(rgb, ref["bbox"])
            input_feat = self._extract_features(input_roi)

            evaluation = self._evaluate(
                ref["feature"],
                input_feat,
                ref["mask"],
            )

            pass_mask = evaluation.pop("pass_mask")
            similarity_map = evaluation.pop("similarity_map")

            evaluation["pass_mask"] = pass_mask
            evaluation["similarity_map"] = similarity_map

            result[side] = evaluation

            debug_image = self._draw_debug(
                debug_image,
                side,
                ref["bbox"],
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
    print(f"SIM_THRESHOLD: {estimator.sim_threshold}")
    print(f"COVERAGE_THRESHOLD: {estimator.coverage_threshold}")

    for side in ("left", "right"):
        item = result[side]
        print(
            f"{side}: {item['label']} | "
            f"coverage={item['coverage_percent']:.1f}% | "
            f"mean={item['mean_similarity']:.4f}"
        )

    debug_path = PROJECT_ROOT / "dino_test" / "debug_result.png"
    cv2.imwrite(str(debug_path), result["debug_image"])
    print(f"Debug image: {debug_path}")