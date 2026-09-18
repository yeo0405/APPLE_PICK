#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from sam2.sam2_image_predictor import SAM2ImagePredictor


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"
GT_PATH = PROJECT_ROOT / "service" / "ALIGNMENT" / "GT" / "gt.json"

SAM2_CONFIG = MODEL_DIR / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = MODEL_DIR / "sam2.1_hiera_small.pt"

SHOW_SAM2_MASK = True
MASK_ALPHA = 0.35

SAM2_MASK_COLOR = (0, 255, 0)
GT_OBB_COLOR = (0, 0, 255)
INPUT_OBB_COLOR = (255, 0, 0)
CONTOUR_COLOR = (0, 255, 0)
DELTA_LINE_COLOR = (255, 255, 0)

OBB_ALPHA = 0.5
ORIGIN_ALPHA = 0.8
OBB_THICKNESS = 4
ORIGIN_RADIUS = 4
DELTA_LINE_THICKNESS = 2
DELTA_LINE_GAP = 10
MIN_MASK_AREA = 100

DEFAULT_COEF_X = 0.001
DEFAULT_COEF_Y = 0.001


class AlignmentEstimator:
    def __init__(
        self,
        coef_x: float = DEFAULT_COEF_X,
        coef_y: float = DEFAULT_COEF_Y,
    ):
        self.coef_x = float(coef_x)
        self.coef_y = float(coef_y)

        self.gt_data = self._load_gt()
        self.gt_bbox = self._get_gt_bbox()
        self.gt_mask = self._load_gt_mask()
        self.gt_obb = self._fit_obb(self.gt_mask)
        self.predictor = self._load_sam2()

    def _load_gt(self) -> Dict[str, Any]:
        with open(GT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _get_gt_bbox(self) -> Tuple[float, float, float, float]:
        return tuple(map(float, self.gt_data["objects"][0]["bbox"]))

    def _load_gt_mask(self) -> np.ndarray:
        mask_file = Path(self.gt_data["objects"][0]["mask"]["file"])

        if mask_file.is_absolute():
            mask_path = mask_file
        elif str(mask_file).startswith("service/"):
            mask_path = PROJECT_ROOT / mask_file
        else:
            mask_path = GT_PATH.parent / mask_file

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise FileNotFoundError(
                f"GT mask not found: {mask_path}"
            )

        mask = mask > 127

        if int(mask.sum()) < MIN_MASK_AREA:
            raise ValueError("GT mask area is too small")

        return mask

    def _load_sam2(self):
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(
            config_dir=str(MODEL_DIR.resolve()),
            version_base=None,
        ):
            cfg = compose(config_name=SAM2_CONFIG.stem)
            model = instantiate(
                cfg.model,
                _recursive_=True,
            )

        checkpoint = torch.load(
            str(SAM2_CHECKPOINT),
            map_location="cpu",
        )

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]

        model.load_state_dict(
            checkpoint,
            strict=False,
        )
        model.eval()

        if torch.cuda.is_available():
            model = model.cuda()

        return SAM2ImagePredictor(model)

    @staticmethod
    def _fit_obb(
        mask: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        ys, xs = np.where(mask)

        if len(xs) < 3:
            return None

        points = np.column_stack(
            (xs, ys)
        ).astype(np.float32)

        rect = cv2.minAreaRect(points)
        (cx, cy), (width, height), angle = rect

        if width < 1e-6 or height < 1e-6:
            return None

        box = cv2.boxPoints(rect).astype(np.float32)

        if width < height:
            width, height = height, width
            angle += 90.0

        angle = (angle + 180.0) % 180.0

        center = np.array(
            [cx, cy],
            dtype=np.float32,
        )

        return {
            "center": center.tolist(),
            "width": float(width),
            "height": float(height),
            "angle_deg": float(angle),
            "box": box.tolist(),
            "area": float(width * height),
        }

    def _predict_mask(
        self,
        rgb: np.ndarray,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[float],
        Tuple[float, float, float, float],
    ]:
        h, w = rgb.shape[:2]

        x1, y1, x2, y2 = self.gt_bbox

        x1 = max(0.0, min(float(w - 1), x1))
        y1 = max(0.0, min(float(h - 1), y1))
        x2 = max(0.0, min(float(w - 1), x2))
        y2 = max(0.0, min(float(h - 1), y2))

        bbox = np.array(
            [x1, y1, x2, y2],
            dtype=np.float32,
        )

        self.predictor.set_image(rgb)

        masks, scores, _ = self.predictor.predict(
            box=bbox,
            multimask_output=True,
        )

        best_mask = None
        best_score = None

        for mask, score in zip(masks, scores):
            mask = mask.astype(bool)

            if int(mask.sum()) < MIN_MASK_AREA:
                continue

            score = float(score)

            if best_score is None or score > best_score:
                best_mask = mask
                best_score = score

        return (
            best_mask,
            best_score,
            (x1, y1, x2, y2),
        )

    @staticmethod
    def _get_debug_obb(
        obb: Optional[Dict[str, Any]],
        target_width: Optional[float] = None,
        target_height: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        if obb is None:
            return None

        center = np.asarray(
            obb["center"],
            dtype=np.float32,
        )

        width = float(obb["width"])
        height = float(obb["height"])
        angle = float(obb["angle_deg"])

        if target_width is not None:
            width = min(width, target_width)

        if target_height is not None:
            height = min(height, target_height)

        rect = (
            tuple(center),
            (width, height),
            angle,
        )

        return cv2.boxPoints(rect).astype(np.int32)

    @staticmethod
    def _draw_obb(
        image: np.ndarray,
        obb: Optional[Dict[str, Any]],
        color: Tuple[int, int, int],
        target_width: Optional[float] = None,
        target_height: Optional[float] = None,
    ):
        box = AlignmentEstimator._get_debug_obb(
            obb,
            target_width,
            target_height,
        )

        if box is None:
            return

        if OBB_ALPHA > 0:
            overlay = image.copy()

            cv2.polylines(
                overlay,
                [box],
                True,
                color,
                OBB_THICKNESS,
                cv2.LINE_AA,
            )

            image[:] = cv2.addWeighted(
                overlay,
                OBB_ALPHA,
                image,
                1.0 - OBB_ALPHA,
                0,
            )

        center = tuple(
            np.round(
                np.asarray(
                    obb["center"],
                    dtype=np.float32,
                )
            ).astype(int)
        )

        if ORIGIN_ALPHA > 0:
            overlay = image.copy()

            cv2.circle(
                overlay,
                center,
                ORIGIN_RADIUS,
                color,
                -1,
                cv2.LINE_AA,
            )

            image[:] = cv2.addWeighted(
                overlay,
                ORIGIN_ALPHA,
                image,
                1.0 - ORIGIN_ALPHA,
                0,
            )

    @staticmethod
    def _normalize_angle_difference(
        gt_angle: float,
        input_angle: float,
    ) -> float:
        delta = gt_angle - input_angle
        return (delta + 90.0) % 180.0 - 90.0

    def _calculate_alignment(
        self,
        input_obb: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if self.gt_obb is None or input_obb is None:
            return None

        gt_center = np.asarray(
            self.gt_obb["center"],
            dtype=np.float64,
        )

        input_center = np.asarray(
            input_obb["center"],
            dtype=np.float64,
        )

        dx_px = float(gt_center[0] - input_center[0])
        dy_px = float(gt_center[1] - input_center[1])

        rotation_deg = self._normalize_angle_difference(
            self.gt_obb["angle_deg"],
            input_obb["angle_deg"],
        )

        dx_m = dx_px * self.coef_x
        dy_m = dy_px * self.coef_y

        return {
            "angle_deg": rotation_deg,
            "x_px": dx_px,
            "y_px": dy_px,
            "x_m": dx_m,
            "y_m": dy_m,
            "coef_x": self.coef_x,
            "coef_y": self.coef_y,
        }

    @staticmethod
    def _draw_dashed_line(
        image: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color: Tuple[int, int, int],
        thickness: int = 2,
        dash_length: int = 10,
    ):
        p1 = np.asarray(p1, dtype=np.float32)
        p2 = np.asarray(p2, dtype=np.float32)

        vector = p2 - p1
        length = float(np.linalg.norm(vector))

        if length < 1e-6:
            return

        direction = vector / length
        distance = 0.0

        while distance < length:
            start = p1 + direction * distance
            end = p1 + direction * min(
                distance + dash_length,
                length,
            )

            cv2.line(
                image,
                tuple(np.round(start).astype(int)),
                tuple(np.round(end).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )

            distance += dash_length * 2

    def _draw_delta(
        self,
        image: np.ndarray,
        alignment: Optional[Dict[str, Any]],
        input_obb: Optional[Dict[str, Any]],
    ):
        if alignment is None or input_obb is None:
            return

        gt_center = tuple(
            np.round(
                np.asarray(
                    self.gt_obb["center"],
                    dtype=np.float32,
                )
            ).astype(int)
        )

        input_center = tuple(
            np.round(
                np.asarray(
                    input_obb["center"],
                    dtype=np.float32,
                )
            ).astype(int)
        )

        self._draw_dashed_line(
            image,
            input_center,
            gt_center,
            DELTA_LINE_COLOR,
            DELTA_LINE_THICKNESS,
            DELTA_LINE_GAP,
        )

    def _draw_text(
        self,
        image: np.ndarray,
        alignment: Optional[Dict[str, Any]],
        score: Optional[float],
    ):
        lines = []

        if score is not None:
            lines.append(f"SAM2: {score:.3f}")

        if alignment is not None:
            lines.extend(
                [
                    f"Rotation: {alignment['angle_deg']:.2f} deg",
                    f"DX: {alignment['x_px']:.2f} px",
                    f"DY: {alignment['y_px']:.2f} px",
                    f"DX: {alignment['x_m']:.5f} m",
                    f"DY: {alignment['y_m']:.5f} m",
                ]
            )

        y = 35

        for text in lines:
            cv2.putText(
                image,
                text,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 30

    def _draw_debug(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        score: Optional[float],
        input_obb: Optional[Dict[str, Any]],
        alignment: Optional[Dict[str, Any]],
    ) -> np.ndarray:
        debug = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR,
        )

        if SHOW_SAM2_MASK and mask is not None:
            overlay = debug.copy()
            overlay[mask] = SAM2_MASK_COLOR

            debug = cv2.addWeighted(
                debug,
                1.0 - MASK_ALPHA,
                overlay,
                MASK_ALPHA,
                0,
            )

            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            cv2.drawContours(
                debug,
                contours,
                -1,
                CONTOUR_COLOR,
                2,
            )

        self._draw_obb(
            debug,
            self.gt_obb,
            GT_OBB_COLOR,
        )

        gt_width = (
            self.gt_obb["width"]
            if self.gt_obb is not None
            else None
        )

        gt_height = (
            self.gt_obb["height"]
            if self.gt_obb is not None
            else None
        )

        self._draw_obb(
            debug,
            input_obb,
            INPUT_OBB_COLOR,
            target_width=gt_width,
            target_height=gt_height,
        )

        self._draw_delta(
            debug,
            alignment,
            input_obb,
        )

        self._draw_text(
            debug,
            alignment,
            score,
        )

        return debug

    def predict(
        self,
        rgb: np.ndarray,
    ) -> Dict[str, Any]:
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                "Input image must be HxWx3 RGB"
            )

        mask, score, bbox = self._predict_mask(rgb)

        input_obb = (
            self._fit_obb(mask)
            if mask is not None
            else None
        )

        alignment = self._calculate_alignment(
            input_obb
        )

        debug_image = self._draw_debug(
            rgb,
            mask,
            score,
            input_obb,
            alignment,
        )

        return {
            "present": input_obb is not None,
            "mask": mask,
            "bbox": bbox,
            "sam_score": score,
            "gt_obb": self.gt_obb,
            "input_obb": input_obb,
            "alignment": alignment,
            "debug_image": debug_image,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--coef-x",
        type=float,
        default=DEFAULT_COEF_X,
    )
    parser.add_argument(
        "--coef-y",
        type=float,
        default=DEFAULT_COEF_Y,
    )
    args = parser.parse_args()

    image = cv2.imread(args.input)

    if image is None:
        raise FileNotFoundError(args.input)

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    estimator = AlignmentEstimator(
        coef_x=args.coef_x,
        coef_y=args.coef_y,
    )

    result = estimator.predict(rgb)

    cv2.imwrite(
        args.output,
        result["debug_image"],
    )

    print("GT OBB:")
    print(
        json.dumps(
            result["gt_obb"],
            indent=2,
        )
    )

    print("Input OBB:")
    print(
        json.dumps(
            result["input_obb"],
            indent=2,
        )
    )

    print("Alignment:")
    print(
        json.dumps(
            result["alignment"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
