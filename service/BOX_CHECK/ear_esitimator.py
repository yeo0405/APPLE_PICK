#!/usr/bin/env python3
"""SAM2 Left/Right Object Presence Estimator."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ============================================================
# CONFIGURATION & THRESHOLDS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
GT_PATH = SCRIPT_DIR / "GT" / "gt.json"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"

SAM2_CONFIG = str(MODEL_DIR / "sam2.1_hiera_s.yaml")
SAM2_CHECKPOINT = str(MODEL_DIR / "sam2.1_hiera_small.pt")

OBJECTS = ["left", "right"]

# Detection Thresholds
MIN_IOU = 0.30
MIN_AREA_RATIO = 0.30
MAX_AREA_RATIO = 2.00
MAX_CENTER_DISTANCE = 50.0
MIN_SAM_SCORE = 0.30
MAX_CONTOUR_DISTANCE = 3

# Debug Visualization Config
SHOW_INPUT_MASK = True
SHOW_BBOX = True
SHOW_TEXT = True

MASK_CONTOUR_THICKNESS = 3
BBOX_THICKNESS = 3
FONT_SCALE_STATUS = 0.8
FONT_SCALE_DETAIL = 0.55
FONT_THICKNESS = 2

# SAM2 Model Config
USE_BFLOAT16 = True
DYNAMIC_MULTIMASK_VIA_STABILITY = True
DYNAMIC_MULTIMASK_STABILITY_DELTA = 0.05
DYNAMIC_MULTIMASK_STABILITY_THRESH = 0.98


# ============================================================
# GROUND TRUTH LOADERS
# ============================================================

def load_gt(gt_path: str | Path) -> Dict[str, Any]:
    path = Path(gt_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"GT JSON not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "image" not in data or "objects" not in data:
        raise RuntimeError("GT JSON missing 'image' or 'objects' key.")

    if len(data["objects"]) != 2:
        raise RuntimeError("GT JSON must contain exactly 2 objects.")

    objects = sorted(data["objects"], key=lambda x: x["index"])

    for index, obj in enumerate(objects):
        expected_side = OBJECTS[index]
        if obj.get("side") != expected_side:
            raise RuntimeError(f"Expected object {index} to be '{expected_side}', got '{obj.get('side')}'.")
        if "bbox" not in obj:
            raise RuntimeError(f"Object {index} is missing 'bbox'.")
        if "mask" not in obj or "file" not in obj["mask"]:
            raise RuntimeError(f"Object {index} is missing 'mask.file'.")

    return data


def load_gt_masks(gt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    objects = sorted(gt_data["objects"], key=lambda x: x["index"])
    results = []

    for obj in objects:
        index = int(obj["index"])
        side = obj["side"]
        bbox = [int(v) for v in obj["bbox"]]

        mask_path = Path(obj["mask"]["file"]).resolve()
        if not mask_path.is_file():
            raise FileNotFoundError(f"GT mask not found: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot load GT mask: {mask_path}")

        mask = mask > 127
        area = int(np.count_nonzero(mask))
        if area == 0:
            raise RuntimeError(f"GT mask {index} is empty: {mask_path}")

        results.append({
            "index": index,
            "side": side,
            "bbox": bbox,
            "mask": mask,
            "score": float(obj["mask"].get("score", 0.0)),
            "area": area,
            "file": str(mask_path),
        })

    return results


# ============================================================
# MODEL & MASK METRICS HELPER FUNCTIONS
# ============================================================

def build_sam2_from_file(config_path: str, checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    cfg_path, ckpt_path = Path(config_path).resolve(), Path(checkpoint_path).resolve()

    if not cfg_path.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = []
    if DYNAMIC_MULTIMASK_VIA_STABILITY:
        overrides.extend([
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            f"++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta={DYNAMIC_MULTIMASK_STABILITY_DELTA}",
            f"++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh={DYNAMIC_MULTIMASK_STABILITY_THRESH}",
        ])

    with initialize_config_dir(version_base=None, config_dir=str(cfg_path.parent)):
        cfg = compose(config_name=cfg_path.stem, overrides=overrides)
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    _load_checkpoint(model, str(ckpt_path))
    model = model.to(device)
    model.eval()

    print("[SAM2] Model loaded successfully")
    return model


def calculate_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union > 0 else 0.0


def mask_center(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def get_largest_contour(mask: np.ndarray) -> Optional[np.ndarray]:
    binary = (np.asarray(mask, dtype=np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def calculate_contour_distance(gt_mask: np.ndarray, input_mask: np.ndarray) -> float:
    gt_contour = get_largest_contour(gt_mask)
    input_contour = get_largest_contour(input_mask)

    if gt_contour is None or input_contour is None:
        return float("inf")

    if cv2.contourArea(gt_contour) <= 0 or cv2.contourArea(input_contour) <= 0:
        return float("inf")

    return float(cv2.matchShapes(gt_contour, input_contour, cv2.CONTOURS_MATCH_I1, 0.0))


def compare_object(gt_mask: np.ndarray, input_mask: Optional[np.ndarray], input_score: float) -> Dict[str, Any]:
    gt_area = int(np.count_nonzero(gt_mask))

    if input_mask is None:
        return {
            "exists": False, "iou": 0.0, "gt_area": gt_area, "input_area": 0,
            "area_ratio": 0.0, "center_distance": float("inf"), "contour_distance": float("inf"),
            "input_score": 0.0, "passed_iou": False, "passed_area": False,
            "passed_center": False, "passed_contour": False, "passed_score": False,
        }

    if gt_mask.shape != input_mask.shape:
        input_mask = cv2.resize(
            input_mask.astype(np.uint8),
            (gt_mask.shape[1], gt_mask.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    input_area = int(np.count_nonzero(input_mask))
    iou = calculate_iou(gt_mask, input_mask)
    area_ratio = float(input_area) / float(gt_area) if gt_area > 0 else 0.0

    gt_center = mask_center(gt_mask)
    input_center = mask_center(input_mask)

    if gt_center is None or input_center is None:
        center_distance = float("inf")
    else:
        dx, dy = gt_center[0] - input_center[0], gt_center[1] - input_center[1]
        center_distance = float(np.sqrt(dx * dx + dy * dy))

    contour_distance = calculate_contour_distance(gt_mask, input_mask)

    passed_iou = iou >= MIN_IOU
    passed_area = MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO
    passed_center = center_distance <= MAX_CENTER_DISTANCE
    passed_contour = contour_distance <= MAX_CONTOUR_DISTANCE
    passed_score = input_score >= MIN_SAM_SCORE

    exists = passed_iou and passed_area and passed_center and passed_contour and passed_score

    return {
        "exists": exists, "iou": iou, "gt_area": gt_area, "input_area": input_area,
        "area_ratio": area_ratio, "center_distance": center_distance,
        "contour_distance": contour_distance, "input_score": input_score,
        "passed_iou": passed_iou, "passed_area": passed_area,
        "passed_center": passed_center, "passed_contour": passed_contour,
        "passed_score": passed_score,
    }


# ============================================================
# DEBUG ANNOTATION
# ============================================================

def draw_debug(
    input_image: np.ndarray,
    gt_results: List[Dict[str, Any]],
    input_results: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
) -> np.ndarray:
    debug = input_image.copy()
    image_height, image_width = input_image.shape[:2]

    for i, gt in enumerate(gt_results):
        side = gt["side"]
        x1, y1, x2, y2 = gt["bbox"]
        comparison = comparisons[i]
        input_mask = input_results[i]["mask"]
        exists = comparison["exists"]

        color = (0, 255, 0) if exists else (0, 0, 255)

        if SHOW_BBOX:
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, BBOX_THICKNESS)

        if SHOW_INPUT_MASK and exists and input_mask is not None:
            if input_mask.shape == (image_height, image_width):
                full_mask = input_mask.astype(np.uint8) * 255
            else:
                full_mask = cv2.resize(
                    input_mask.astype(np.uint8),
                    (image_width, image_height),
                    interpolation=cv2.INTER_NEAREST,
                ) * 255

            contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug, contours, -1, (0, 255, 0), MASK_CONTOUR_THICKNESS)

        if not SHOW_TEXT:
            continue

        status = "PRESENT" if exists else "ABSENT"
        text1 = f"{side.upper()}: {status}"
        text2 = f"IoU={comparison['iou']:.2f} Area={comparison['area_ratio']:.2f} D={comparison['center_distance']:.1f}"

        contour_distance = comparison["contour_distance"]
        contour_text = f"C={contour_distance:.3f}" if np.isfinite(contour_distance) else "C=INF"
        text3 = f"S={comparison['input_score']:.2f} {contour_text}"

        cv2.putText(debug, text1, (x1, max(30, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_STATUS, color, FONT_THICKNESS)
        cv2.putText(debug, text2, (x1, max(55, y1 + 20)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_DETAIL, (255, 255, 0), FONT_THICKNESS)
        cv2.putText(debug, text3, (x1, max(80, y1 + 45)), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_DETAIL, (255, 255, 0), FONT_THICKNESS)

    return debug


# ============================================================
# SAM2 ESTIMATOR CLASS
# ============================================================

class SAM2Estimator:
    def __init__(self, annotate: bool = True):
        self.annotate = annotate

        print("===================================")
        print("[SAM2Estimator] Initializing...")
        print("[SAM2Estimator] Loading GT...")

        self.gt_data = load_gt(GT_PATH)
        self.gt_results = load_gt_masks(self.gt_data)

        if len(self.gt_results) != 2:
            raise RuntimeError("Failed to load LEFT and RIGHT GT masks.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[SAM2Estimator] Device: {self.device}")

        model = build_sam2_from_file(SAM2_CONFIG, SAM2_CHECKPOINT, self.device)
        self.predictor = SAM2ImagePredictor(model)

        print("[SAM2Estimator] Ready")
        print("===================================")

    def _sam_predict(self, bbox: List[int]) -> Tuple[Optional[np.ndarray], float]:
        box = np.asarray(bbox, dtype=np.float32)

        try:
            with torch.inference_mode():
                if self.device.startswith("cuda") and USE_BFLOAT16:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, _ = self.predictor.predict(box=box, multimask_output=True, return_logits=False)
                else:
                    masks, scores, _ = self.predictor.predict(box=box, multimask_output=True, return_logits=False)
        except Exception as e:
            print(f"[ERROR] SAM2 prediction failed: {e}")
            return None, 0.0

        masks, scores = np.asarray(masks), np.asarray(scores)
        if masks.ndim == 4:
            masks = masks[0]
        if scores.ndim > 1:
            scores = scores[0]

        if len(masks) == 0:
            return None, 0.0

        best_idx = int(np.argmax(scores))
        return np.asarray(masks[best_idx], dtype=bool), float(scores[best_idx])

    def _generate_input_masks(self, rgb: np.ndarray) -> List[Dict[str, Any]]:
        input_rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        with torch.inference_mode():
            if self.device.startswith("cuda") and USE_BFLOAT16:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    self.predictor.set_image(input_rgb)
            else:
                self.predictor.set_image(input_rgb)

        height, width = rgb.shape[:2]
        results = []

        for obj in self.gt_results:
            index, side, bbox = obj["index"], obj["side"], obj["bbox"]
            x1, y1, x2, y2 = bbox

            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                print(f"[WARNING] Invalid bbox for {side}")
                results.append({"index": index, "side": side, "bbox": bbox, "mask": None, "score": 0.0})
                continue

            mask, score = self._sam_predict(bbox)
            results.append({"index": index, "side": side, "bbox": bbox, "mask": mask, "score": score})

        return results

    def predict(self, rgb: np.ndarray) -> Dict[str, Any]:
        if rgb is None:
            raise ValueError("SAM2Estimator.predict(): rgb is None")
        if not isinstance(rgb, np.ndarray):
            raise TypeError("SAM2Estimator.predict(): rgb must be numpy.ndarray")
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("SAM2Estimator.predict(): rgb must have shape (H, W, 3)")

        image_size = self.gt_data.get("image_size")
        if image_size is not None:
            expected_w, expected_h = int(image_size[0]), int(image_size[1])
            actual_h, actual_w = rgb.shape[:2]
            if actual_w != expected_w or actual_h != expected_h:
                raise ValueError(f"Input image resolution ({actual_w}x{actual_h}) does not match GT ({expected_w}x{expected_h})")

        input_results = self._generate_input_masks(rgb)
        comparisons = [
            compare_object(
                gt_mask=self.gt_results[i]["mask"],
                input_mask=input_results[i]["mask"],
                input_score=input_results[i]["score"],
            )
            for i in range(2)
        ]

        debug_image = draw_debug(rgb, self.gt_results, input_results, comparisons) if self.annotate else None

        return {
            "left": bool(comparisons[0]["exists"]),
            "right": bool(comparisons[1]["exists"]),
            "debug_image": debug_image,
            "left_result": comparisons[0],
            "right_result": comparisons[1],
        }


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input RGB image")
    parser.add_argument("--output", default="debug_image.jpg", help="Debug image output path")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise RuntimeError(f"Cannot load image: {args.input}")

    estimator = SAM2Estimator()
    result = estimator.predict(image)

    print(f"LEFT  : {'PRESENT' if result['left'] else 'ABSENT'}")
    print(f"RIGHT : {'PRESENT' if result['right'] else 'ABSENT'}")
    print(f"LEFT IoU: {result['left_result']['iou']:.4f}")
    print(f"RIGHT IoU: {result['right_result']['iou']:.4f}")

    if result["debug_image"] is not None:
        cv2.imwrite(args.output, result["debug_image"])
        print(f"Debug image saved to: {args.output}")