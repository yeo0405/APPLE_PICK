#!/usr/bin/env python3
"""SAM2 Ripcord GT Mask vs Input Comparison Script."""

import argparse
import json
import os
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
# CONFIG & PATHS
# ============================================================

GT_PATH = "/home/yeo/Downloads/ripcord/gt.json"
SAM2_CONFIG = (
    "/home/yeo/Downloads/SG/src/"
    "tomo2-imaging/implementation/"
    "tomo_apple_picking/model/"
    "sam2.1_hiera_s.yaml"
)
SAM2_CHECKPOINT = (
    "/home/yeo/Downloads/SG/src/"
    "tomo2-imaging/implementation/"
    "tomo_apple_picking/model/"
    "sam2.1_hiera_small.pt"
)
DEFAULT_OUTPUT = "sam2_compare_debug.jpg"

# ============================================================
# DETECTION THRESHOLDS
# ============================================================

MIN_IOU = 0.30
MIN_AREA_RATIO = 0.30
MAX_AREA_RATIO = 2.50
MAX_CENTER_DISTANCE = 50.0
MIN_SAM_SCORE = 0.30


# ============================================================
# DATA LOADING HELPERS
# ============================================================


def load_gt(gt_path: str) -> Dict[str, Any]:
    """載入 Ground Truth (GT) JSON 檔案並驗證必要欄位。"""
    resolved_path = Path(gt_path).resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"GT JSON not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for field in ["image", "roi", "ripcords"]:
        if field not in data:
            raise RuntimeError(f"GT JSON does not contain '{field}'.")

    if len(data["ripcords"]) != 4:
        raise RuntimeError("GT JSON must contain exactly 4 ripcords.")

    for ripcord in data["ripcords"]:
        if "mask" not in ripcord or "file" not in ripcord["mask"]:
            raise RuntimeError(
                f"Ripcord {ripcord.get('index', '?')} missing 'mask' or 'mask.file'. "
                f"Please regenerate gt.json using gt.py."
            )

    return data


def load_image(path: str) -> np.ndarray:
    """載入影像檔案。"""
    image = cv2.imread(path)
    if image is None:
        raise RuntimeError(f"Cannot load image: {path}")
    return image


def load_gt_masks(gt_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """載入 GT Masks 並確保其大小與 ROI 吻合。"""
    roi_x1, roi_y1, roi_x2, roi_y2 = gt_data["roi"]
    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1

    if roi_width <= 0 or roi_height <= 0:
        raise RuntimeError(f"Invalid ROI: {gt_data['roi']}")

    expected_shape = (roi_height, roi_width)
    results = []

    # 確保依 index 排序 (0: top, 1: second, 2: third, 3: bottom)
    sorted_ripcords = sorted(gt_data["ripcords"], key=lambda x: x["index"])

    for ripcord in sorted_ripcords:
        index = ripcord["index"]
        mask_path = Path(ripcord["mask"]["file"]).resolve()

        if not mask_path.is_file():
            raise FileNotFoundError(f"GT mask not found: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Cannot load GT mask: {mask_path}")

        mask = mask > 127

        if mask.shape != expected_shape:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (roi_width, roi_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)

        area = int(np.count_nonzero(mask))
        if area == 0:
            raise RuntimeError(f"GT mask {index} is empty: {mask_path}")

        score = float(ripcord["mask"].get("score", 0.0))
        center = ripcord["center"]

        results.append(
            {
                "index": index,
                "center": (int(center[0]), int(center[1])),
                "mask": mask,
                "score": score,
                "area": area,
                "file": str(mask_path),
            }
        )

    return results


# ============================================================
# SAM2 INITIALIZATION & PREDICTION
# ============================================================


def build_sam2_from_file(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
) -> torch.nn.Module:
    """從設定檔與權重建立 SAM2 模型。"""
    cfg_path = Path(config_path).resolve()
    ckpt_path = Path(checkpoint_path).resolve()

    if not cfg_path.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {cfg_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {ckpt_path}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(cfg_path.parent)):
        cfg = compose(
            config_name=cfg_path.stem,
            overrides=[
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ],
        )
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    _load_checkpoint(model, str(ckpt_path))
    model = model.to(device)
    model.eval()

    print("[SAM2] Model loaded successfully")
    return model


def load_sam2() -> Tuple[SAM2ImagePredictor, str]:
    """初始化 SAM2 Predictor 並自動選擇執行裝置。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = build_sam2_from_file(SAM2_CONFIG, SAM2_CHECKPOINT, device)
    predictor = SAM2ImagePredictor(model)
    print("[INFO] SAM2 predictor ready.")

    return predictor, device


def sam_predict(
    predictor: SAM2ImagePredictor,
    device: str,
    point: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], float]:
    """輸入單點位置，執行 SAM2 預測並傳回分數最高的 Mask。"""
    point_coords = np.array([[point[0], point[1]]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)

    try:
        with torch.inference_mode():
            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.startswith("cuda")
                else torch.no_grad()
            )
            with autocast_ctx:
                masks, scores, _ = predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=None,
                    mask_input=None,
                    multimask_output=True,
                    return_logits=False,
                )
    except Exception as e:
        print(f"[ERROR] SAM2 prediction failed: {e}")
        return None, 0.0

    masks = np.asarray(masks)
    scores = np.asarray(scores)

    if masks.ndim == 4:
        masks = masks[0]
    if scores.ndim > 1:
        scores = scores[0]

    if len(masks) == 0:
        return None, 0.0

    best_idx = int(np.argmax(scores))
    best_mask = np.asarray(masks[best_idx], dtype=bool)
    best_score = float(scores[best_idx])

    return best_mask, best_score


def generate_input_masks(
    predictor: SAM2ImagePredictor,
    device: str,
    input_image: np.ndarray,
    gt_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """依據 GT 點位在輸入影像的 ROI 內生成對應的 Masks。"""
    roi_x1, roi_y1, roi_x2, roi_y2 = gt_data["roi"]
    input_roi = input_image[roi_y1:roi_y2, roi_x1:roi_x2]

    if input_roi.size == 0:
        raise RuntimeError("Input ROI is empty.")

    input_rgb = cv2.cvtColor(input_roi, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device.startswith("cuda")
            else torch.no_grad()
        )
        with autocast_ctx:
            predictor.set_image(input_rgb)

    results = []
    sorted_ripcords = sorted(gt_data["ripcords"], key=lambda x: x["index"])

    for ripcord in sorted_ripcords:
        index = ripcord["index"]
        gx, gy = ripcord["center"]
        rx, ry = gx - roi_x1, gy - roi_y1

        print(f"[INPUT] Ripcord {index}: point=({rx}, {ry})")

        if rx < 0 or ry < 0 or rx >= input_rgb.shape[1] or ry >= input_rgb.shape[0]:
            results.append({"index": index, "mask": None, "score": 0.0})
            continue

        mask, score = sam_predict(predictor, device, (rx, ry))
        results.append({"index": index, "mask": mask, "score": score})

    return results


# ============================================================
# MASK EVALUATION & COMPARISON
# ============================================================


def calculate_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """計算兩張遮罩的 Intersection over Union (IoU)。"""
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    return float(intersection) / float(union) if union > 0 else 0.0


def mask_center(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """計算遮罩的重心中心點。"""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return float(np.mean(xs)), float(np.mean(ys))


def compare_ripcord(
    gt_mask: np.ndarray,
    input_mask: np.ndarray,
    input_score: float,
) -> Dict[str, Any]:
    """比較 GT Mask 與推論產生的 Input Mask，驗證各項指標是否通過。"""
    gt_area = int(np.count_nonzero(gt_mask))
    input_area = int(np.count_nonzero(input_mask))

    iou = calculate_iou(gt_mask, input_mask)
    area_ratio = float(input_area) / float(gt_area) if gt_area > 0 else 0.0

    gt_center = mask_center(gt_mask)
    input_center = mask_center(input_mask)

    if gt_center is None or input_center is None:
        center_distance = float("inf")
    else:
        dx = gt_center[0] - input_center[0]
        dy = gt_center[1] - input_center[1]
        center_distance = float(np.sqrt(dx * dx + dy * dy))

    passed_iou = iou >= MIN_IOU
    passed_area = MIN_AREA_RATIO <= area_ratio <= MAX_AREA_RATIO
    passed_center = center_distance <= MAX_CENTER_DISTANCE
    passed_score = input_score >= MIN_SAM_SCORE

    exists = passed_iou and passed_area and passed_center and passed_score

    return {
        "exists": exists,
        "iou": iou,
        "gt_area": gt_area,
        "input_area": input_area,
        "area_ratio": area_ratio,
        "center_distance": center_distance,
        "input_score": input_score,
        "passed_iou": passed_iou,
        "passed_area": passed_area,
        "passed_center": passed_center,
        "passed_score": passed_score,
    }


# ============================================================
# VISUALIZATION
# ============================================================


def draw_debug(
    input_image: np.ndarray,
    gt_data: Dict[str, Any],
    gt_results: List[Dict[str, Any]],
    input_results: List[Dict[str, Any]],
    comparisons: List[Dict[str, Any]],
) -> np.ndarray:
    """繪製並輸出標註結果的 Debug 影像。"""
    debug = input_image.copy()
    roi_x1, roi_y1, roi_x2, roi_y2 = gt_data["roi"]

    # 繪製 ROI
    cv2.rectangle(debug, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 0, 0), 3)

    sorted_ripcords = sorted(gt_data["ripcords"], key=lambda x: x["index"])

    for i in range(4):
        ripcord = sorted_ripcords[i]
        index = ripcord["index"]
        gx, gy = ripcord["center"]

        comparison = comparisons[i]
        input_mask = input_results[i]["mask"]

        # 僅當判定存在 (PRESENT) 時繪製邊框
        if comparison["exists"] and input_mask is not None:
            full_mask = np.zeros(input_image.shape[:2], dtype=np.uint8)
            full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = input_mask.astype(np.uint8) * 255

            contours, _ = cv2.findContours(
                full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(debug, contours, -1, (0, 255, 0), 3)  # GREEN

        # 標註 GT 中心點
        cv2.circle(debug, (gx, gy), 10, (255, 0, 255), 2)

        # 顯示狀態與數值資訊
        status = "PRESENT" if comparison["exists"] else "ABSENT"
        status_color = (0, 255, 0) if comparison["exists"] else (0, 0, 255)

        text1 = f"Ripcord {index}: {status}"
        text2 = (
            f"IoU={comparison['iou']:.2f} "
            f"Area={comparison['area_ratio']:.2f} "
            f"D={comparison['center_distance']:.1f}"
        )
        text3 = f"S={comparison['input_score']:.2f}"

        cv2.putText(
            debug,
            text1,
            (gx + 20, gy - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
        )
        cv2.putText(
            debug,
            text2,
            (gx + 20, gy + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            debug,
            text3,
            (gx + 20, gy + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

    # 標題外框
    cv2.rectangle(debug, (10, 10), (750, 80), (0, 0, 0), -1)
    cv2.putText(
        debug,
        "SAM2 GT MASK vs INPUT",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
    )

    return debug


def get_final_index(comparisons: List[Dict[str, Any]]) -> Optional[int]:
    """傳回第一個檢測為 PRESENT 的 Ripcord index。"""
    for index, comparison in enumerate(comparisons):
        if comparison["exists"]:
            return index
    return None


# ============================================================
# MAIN ENTRY POINT
# ============================================================


def main() -> Optional[int]:
    parser = argparse.ArgumentParser(
        description="SAM2 Ripcord GT Mask vs Input Comparison"
    )
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT, help="Output debug image path"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SAM2 RIPCORD COMPARISON")
    print("=" * 70)

    # 1. 載入 GT 與 Mask
    gt_data = load_gt(GT_PATH)
    gt_results = load_gt_masks(gt_data)
    if len(gt_results) != 4:
        raise RuntimeError("Failed to load all 4 GT masks.")

    # 2. 載入輸入影像與解析度檢查
    input_image = load_image(args.input)
    image_size = gt_data.get("image_size")
    if image_size is not None:
        expected_w, expected_h = int(image_size[0]), int(image_size[1])
        actual_h, actual_w = input_image.shape[:2]
        if actual_w != expected_w or actual_h != expected_h:
            raise RuntimeError(
                f"Input image resolution ({actual_w}x{actual_h}) "
                f"does not match GT ({expected_w}x{expected_h})."
            )

    # 3. 初始化 SAM2 並預測
    predictor, device = load_sam2()
    input_results = generate_input_masks(
        predictor=predictor,
        device=device,
        input_image=input_image,
        gt_data=gt_data,
    )
    if len(input_results) != 4:
        raise RuntimeError("Failed to generate all 4 input masks.")

    # 4. 比對遮罩
    comparisons = []
    for i in range(4):
        gt_mask = gt_results[i]["mask"]
        input_mask = input_results[i]["mask"]
        input_score = input_results[i]["score"]

        if input_mask is None:
            comparison = {
                "exists": False,
                "iou": 0.0,
                "gt_area": int(np.count_nonzero(gt_mask)),
                "input_area": 0,
                "area_ratio": 0.0,
                "center_distance": float("inf"),
                "input_score": 0.0,
                "passed_iou": False,
                "passed_area": False,
                "passed_center": False,
                "passed_score": False,
            }
        else:
            if gt_mask.shape != input_mask.shape:
                input_mask = cv2.resize(
                    input_mask.astype(np.uint8),
                    (gt_mask.shape[1], gt_mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                input_results[i]["mask"] = input_mask

            comparison = compare_ripcord(
                gt_mask=gt_mask,
                input_mask=input_mask,
                input_score=input_score,
            )

        comparisons.append(comparison)

    # 5. 輸出詳細紀錄
    print("\n" + "=" * 70)
    print("RIPCORD RESULTS")
    print("=" * 70)

    for i, comparison in enumerate(comparisons):
        print(f"Ripcord {i}: {'PRESENT' if comparison['exists'] else 'ABSENT'}")
        print(f"  IoU            : {comparison['iou']:.4f}")
        print(f"  Area ratio     : {comparison['area_ratio']:.4f}")
        print(f"  Center distance: {comparison['center_distance']:.2f}")
        print(f"  SAM score      : {comparison['input_score']:.4f}")
        print(f"  IoU check      : {comparison['passed_iou']}")
        print(f"  Area check     : {comparison['passed_area']}")
        print(f"  Center check   : {comparison['passed_center']}")
        print(f"  Score check    : {comparison['passed_score']}\n")

    # 6. 計算最終 Index
    final_index = get_final_index(comparisons)
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)
    print(f"Ripcord index: {final_index}")
    print(
        "No ripcord detected."
        if final_index is None
        else f"Detected ripcord: {final_index}"
    )
    print("=" * 70)

    # 7. 儲存 Debug 圖檔
    debug = draw_debug(
        input_image=input_image,
        gt_data=gt_data,
        gt_results=gt_results,
        input_results=input_results,
        comparisons=comparisons,
    )

    output_path = os.path.abspath(args.output)
    if not cv2.imwrite(output_path, debug):
        raise RuntimeError(f"Failed to save debug image: {output_path}")

    print(f"Debug image saved to: {output_path}")
    return final_index


if __name__ == "__main__":
    result = main()
    print(f"RESULT_INDEX={result}")
