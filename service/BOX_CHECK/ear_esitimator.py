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

SIM_THRESHOLD = 0.9
COVERAGE_THRESHOLD = 0.2

PRESENT_LABEL = "PRESENT"
ABSENT_LABEL = "ABSENT"

PRESENT_COLOR = (0, 0, 255)
ABSENT_COLOR = (0, 255, 0)
PATCH_COLOR = (255, 255, 0)
POLYGON_COLOR = (255, 0, 0)


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

            if "polygon" not in obj:
                raise KeyError(
                    f"Missing 'polygon' in GT object: {side}"
                )

            polygon = np.asarray(
                obj["polygon"],
                dtype=np.float32,
            )

            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError(
                    f"Invalid polygon for {side}: {obj['polygon']}"
                )

            bbox = self._polygon_to_bbox(
                polygon,
                gt_image.shape,
            )

            gt_roi = self._crop_bbox(
                gt_image,
                bbox,
            )

            polygon_roi = polygon - np.array(
                [bbox[0], bbox[1]],
                dtype=np.float32,
            )

            polygon_mask = self._polygon_mask(
                polygon_roi,
                gt_roi.shape[:2],
            )

            ear_mask = self._get_patch_mask(
                polygon_mask,
            )

            gt_feat = self._extract_features(
                gt_roi,
            )

            self.reference[side] = {
                "polygon": polygon.tolist(),
                "bbox": bbox,
                "feature": gt_feat,
                "mask": ear_mask,
            }

    def _polygon_to_bbox(
        self,
        polygon: np.ndarray,
        image_shape,
    ):
        h, w = image_shape[:2]

        x1 = int(np.floor(np.min(polygon[:, 0])))
        y1 = int(np.floor(np.min(polygon[:, 1])))
        x2 = int(np.ceil(np.max(polygon[:, 0])))
        y2 = int(np.ceil(np.max(polygon[:, 1])))

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid polygon bbox: {polygon.tolist()}"
            )

        return [x1, y1, x2, y2]

    def _crop_bbox(
        self,
        image: np.ndarray,
        bbox,
    ):
        h, w = image.shape[:2]

        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1)
        x2 = min(w, x2)
        y1 = max(0, y1)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bbox: {bbox}"
            )

        return image[y1:y2, x1:x2]

    def _polygon_mask(
        self,
        polygon: np.ndarray,
        shape,
    ) -> np.ndarray:
        h, w = shape[:2]

        mask = np.zeros(
            (h, w),
            dtype=np.uint8,
        )

        points = np.round(
            polygon,
        ).astype(np.int32)

        cv2.fillPoly(
            mask,
            [points],
            255,
        )

        return mask

    def _extract_features(
        self,
        image: np.ndarray,
    ) -> torch.Tensor:
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        x = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )

        x = x.unsqueeze(0).to(
            self.device,
        )

        with torch.inference_mode():
            feat = self.model.forward_features(
                x,
            )["x_norm_patchtokens"][0]

        return F.normalize(
            feat,
            dim=-1,
        )

    def _get_patch_mask(
        self,
        mask: np.ndarray,
    ) -> np.ndarray:
        small = cv2.resize(
            mask.astype(np.uint8),
            (PATCH_GRID, PATCH_GRID),
            interpolation=cv2.INTER_AREA,
        )

        return small >= int(
            MASK_THRESHOLD * 255
        )

    def _evaluate(
        self,
        gt_feat: torch.Tensor,
        input_feat: torch.Tensor,
        roi_mask: np.ndarray,
    ):
        gt_feat = gt_feat.reshape(
            PATCH_GRID,
            PATCH_GRID,
            -1,
        )

        input_feat = input_feat.reshape(
            PATCH_GRID,
            PATCH_GRID,
            -1,
        )

        similarity = (
            gt_feat * input_feat
        ).sum(dim=-1)

        roi_mask_t = torch.from_numpy(
            roi_mask,
        ).to(self.device)

        scores = similarity[
            roi_mask_t
        ]

        if scores.numel() == 0:
            raise ValueError(
                "Polygon does not contain any valid DINOv3 patches"
            )

        scores_np = (
            scores
            .detach()
            .cpu()
            .numpy()
        )

        pass_mask = (
            (similarity >= self.sim_threshold)
            & roi_mask_t
        )

        coverage = float(
            pass_mask[roi_mask_t]
            .float()
            .mean()
        )

        mean_similarity = float(
            scores.mean()
        )

        median_similarity = float(
            scores.median()
        )

        p10_similarity = float(
            np.percentile(
                scores_np,
                10,
            )
        )

        is_present = (
            coverage
            >= self.coverage_threshold
        )

        label = (
            self.present_label
            if is_present
            else self.absent_label
        )

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
            "patch_count": int(
                len(scores_np)
            ),
            "pass_patch_count": int(
                pass_mask.sum().item()
            ),
            "pass_mask": pass_mask.detach().cpu().numpy(),
            "similarity_map": similarity.detach().cpu().numpy(),
        }

    def _draw_debug(
        self,
        image: np.ndarray,
        side: str,
        polygon,
        bbox,
        evaluation: Dict[str, Any],
    ):
        polygon = np.asarray(
            polygon,
            dtype=np.int32,
        )

        x1, y1, x2, y2 = map(
            int,
            bbox,
        )

        color = (
            PRESENT_COLOR
            if evaluation["present"]
            else ABSENT_COLOR
        )

        debug = image.copy()
        overlay = debug.copy()

        roi_w = max(
            1,
            x2 - x1,
        )

        roi_h = max(
            1,
            y2 - y1,
        )

        pass_mask = evaluation[
            "pass_mask"
        ]

        for row, col in zip(
            *np.where(pass_mask)
        ):
            px1 = (
                x1
                + int(
                    col
                    * roi_w
                    / PATCH_GRID
                )
            )

            py1 = (
                y1
                + int(
                    row
                    * roi_h
                    / PATCH_GRID
                )
            )

            px2 = (
                x1
                + int(
                    (col + 1)
                    * roi_w
                    / PATCH_GRID
                )
            )

            py2 = (
                y1
                + int(
                    (row + 1)
                    * roi_h
                    / PATCH_GRID
                )
            )

            cv2.rectangle(
                overlay,
                (px1, py1),
                (px2, py2),
                PATCH_COLOR,
                -1,
            )

        debug = cv2.addWeighted(
            overlay,
            0.28,
            debug,
            0.72,
            0,
        )

        cv2.polylines(
            debug,
            [polygon.reshape(-1, 1, 2)],
            True,
            color,
            3,
        )

        label = evaluation["label"]
        coverage = evaluation[
            "coverage_percent"
        ]

        text = (
            f"{side.upper()}: "
            f"{label}  "
            f"cov={coverage:.1f}%"
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

        anchor_x = int(
            np.min(polygon[:, 0])
        )

        anchor_y = int(
            np.min(polygon[:, 1])
        )

        text_y = max(
            th + 10,
            anchor_y,
        )

        bg_y1 = max(
            0,
            text_y - th - 8,
        )

        bg_y2 = min(
            debug.shape[0],
            text_y + 4,
        )

        bg_x1 = max(
            0,
            anchor_x,
        )

        bg_x2 = min(
            debug.shape[1],
            bg_x1 + tw + 10,
        )

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
            (
                bg_x1 + 5,
                text_y - 2,
            ),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

        return debug

    def predict(
        self,
        rgb: np.ndarray,
    ) -> Dict[str, Any]:
        if (
            rgb is None
            or not isinstance(
                rgb,
                np.ndarray,
            )
        ):
            raise ValueError(
                "rgb must be a numpy.ndarray"
            )

        debug_image = rgb.copy()

        result = {
            "left": None,
            "right": None,
            "debug_image": debug_image,
        }

        for side, ref in self.reference.items():
            polygon = np.asarray(
                ref["polygon"],
                dtype=np.float32,
            )

            bbox = self._polygon_to_bbox(
                polygon,
                rgb.shape,
            )

            input_roi = self._crop_bbox(
                rgb,
                bbox,
            )

            polygon_roi = polygon - np.array(
                [bbox[0], bbox[1]],
                dtype=np.float32,
            )

            polygon_mask = self._polygon_mask(
                polygon_roi,
                input_roi.shape[:2],
            )

            roi_mask = self._get_patch_mask(
                polygon_mask,
            )

            input_feat = self._extract_features(
                input_roi,
            )

            evaluation = self._evaluate(
                ref["feature"],
                input_feat,
                roi_mask,
            )

            pass_mask = evaluation.pop(
                "pass_mask"
            )

            similarity_map = evaluation.pop(
                "similarity_map"
            )

            evaluation["pass_mask"] = pass_mask
            evaluation["similarity_map"] = similarity_map

            result[side] = evaluation

            debug_image = self._draw_debug(
                debug_image,
                side,
                polygon,
                bbox,
                evaluation,
            )

        result["debug_image"] = debug_image

        return result


if __name__ == "__main__":
    import sys

    image_path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else PROJECT_ROOT
        / "dino_test"
        / "hard.png"
    )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {image_path}"
        )

    estimator = DINOv3Estimator()
    result = estimator.predict(
        image,
    )

    print(
        f"Device: {estimator.device}"
    )

    print(
        f"SIM_THRESHOLD: "
        f"{estimator.sim_threshold}"
    )

    print(
        f"COVERAGE_THRESHOLD: "
        f"{estimator.coverage_threshold}"
    )

    for side in ("left", "right"):
        item = result[side]

        print(
            f"{side}: "
            f"{item['label']} | "
            f"coverage="
            f"{item['coverage_percent']:.1f}% | "
            f"mean="
            f"{item['mean_similarity']:.4f}"
        )

    debug_path = (
        PROJECT_ROOT
        / "dino_test"
        / "debug_result.png"
    )

    cv2.imwrite(
        str(debug_path),
        result["debug_image"],
    )

    print(
        f"Debug image: {debug_path}"
    )