#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"
SCRIPT_DIR = Path(__file__).resolve().parent
GT_DIR = SCRIPT_DIR / "GT"

SAM2_CONFIG = MODEL_DIR / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = MODEL_DIR / "sam2.1_hiera_small.pt"

WINDOW_NAME = "SAM2 Object GT"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

MIN_MASK_AREA = 100
MASK_ALPHA = 0.35
FONT_SCALE = 0.65
FONT_THICKNESS = 2

image = None
display = None
predictor = None
INPUT_IMAGE = ""

roi_start = None
roi_end = None
drawing_roi = False

detection_mask = None
detection_score = 0.0
sam_done = False
status_message = ""


def find_gt_image():
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        path = GT_DIR / f"GT{ext}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"GT image not found. Expected GT.* in: {GT_DIR}")


def prepare_predictor():
    global predictor

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(config_dir=str(SAM2_CONFIG.parent), version_base=None):
        cfg = compose(config_name=SAM2_CONFIG.stem)

    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, str(SAM2_CHECKPOINT))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image)


def get_roi():
    if roi_start is None or roi_end is None:
        return None

    x1, y1 = roi_start
    x2, y2 = roi_end

    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))

    x1 = max(0, min(x1, image.shape[1] - 1))
    x2 = max(0, min(x2, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    y2 = max(0, min(y2, image.shape[0] - 1))

    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    return x1, y1, x2, y2


def run_sam2():
    global detection_mask, detection_score, sam_done, status_message

    roi = get_roi()
    if roi is None:
        detection_mask = None
        detection_score = 0.0
        sam_done = False
        status_message = "Select a valid ROI first"
        return False

    x1, y1, x2, y2 = roi

    print(f"[SAM2] ROI: [{x1}, {y1}, {x2}, {y2}]")
    print("[SAM2] Running inference...")

    try:
        box = np.array([x1, y1, x2, y2], dtype=np.float32)
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
        )
    except Exception as error:
        print(f"[ERROR] SAM2 prediction failed: {error}")
        detection_mask = None
        detection_score = 0.0
        sam_done = False
        status_message = "SAM2 prediction failed"
        return False

    if masks is None or len(masks) == 0:
        detection_mask = None
        detection_score = 0.0
        sam_done = False
        status_message = "No SAM2 mask found"
        print("[SAM2] No masks detected")
        return False

    candidates = []

    for mask, score in zip(masks, scores):
        area = int(mask.sum())
        if area >= MIN_MASK_AREA:
            candidates.append((float(score), area, mask))

    if not candidates:
        detection_mask = None
        detection_score = 0.0
        sam_done = False
        status_message = "No valid SAM2 mask"
        print("[SAM2] No valid mask")
        return False

    detection_score, _, mask = max(candidates, key=lambda item: item[0])

    detection_mask = mask.astype(np.uint8)
    sam_done = True
    status_message = (
        f"SAM2 done | Score={detection_score:.4f} | "
        f"Area={int(mask.sum())}"
    )

    print(
        f"[SAM2] Selected score={detection_score:.4f}, "
        f"area={int(mask.sum())}"
    )

    return True


def draw_mask(frame):
    if detection_mask is None:
        return

    mask = detection_mask.astype(bool)
    if not np.any(mask):
        return

    overlay = frame.copy()
    overlay[mask] = (0, 255, 0)

    frame[:] = cv2.addWeighted(
        frame,
        1.0 - MASK_ALPHA,
        overlay,
        MASK_ALPHA,
        0,
    )


def draw_roi(frame):
    roi = get_roi()
    if roi is None:
        return

    x1, y1, x2, y2 = roi
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)


def render():
    global display

    if image is None:
        return

    frame = image.copy()

    draw_mask(frame)
    draw_roi(frame)

    cv2.putText(
        frame,
        "Drag ROI | S: SAM2 + Save | C: Clear | ESC: Save + Exit",
        (15, frame.shape[0] - 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )

    if status_message:
        cv2.putText(
            frame,
            status_message,
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 255, 255),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    scale = min(
        WINDOW_WIDTH / frame.shape[1],
        WINDOW_HEIGHT / frame.shape[0],
    )

    if scale < 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    display = frame
    cv2.imshow(WINDOW_NAME, display)


def save_gt():
    global status_message

    if not sam_done or detection_mask is None:
        status_message = "No SAM2 result to save"
        return False

    mask = detection_mask
    ys, xs = np.where(mask > 0)

    if len(xs) == 0:
        status_message = "Empty SAM2 mask"
        return False

    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max())
    y2 = int(ys.max())

    roi = get_roi()

    mask_dir = GT_DIR / "gt_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_path = mask_dir / "object_0.png"

    if not cv2.imwrite(
        str(mask_path),
        (mask * 255).astype(np.uint8),
    ):
        status_message = "Failed to save mask"
        return False

    data = {
        "image": str(Path(INPUT_IMAGE).relative_to(PROJECT_ROOT)),
        "image_size": {
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        },
        "object_count": 1,
        "objects": [
            {
                "index": 0,
                "side": "front",
                "bbox": [x1, y1, x2, y2],
                "roi": list(roi) if roi is not None else None,
                "mask": {
                    "file": str(mask_path.relative_to(GT_DIR)),
                    "area": int(mask.sum()),
                    "sam_score": float(detection_score),
                },
            }
        ],
        "sam2": {
            "config": str(SAM2_CONFIG.relative_to(PROJECT_ROOT)),
            "checkpoint": str(SAM2_CHECKPOINT.relative_to(PROJECT_ROOT)),
        },
    }

    json_path = GT_DIR / "gt.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 60)
    print("GT SAVED")
    print("=" * 60)
    print(f"Image      : {INPUT_IMAGE}")
    print(f"JSON       : {json_path}")
    print(f"Mask       : {mask_path}")
    print(f"ROI        : {roi}")
    print(f"BBox       : [{x1}, {y1}, {x2}, {y2}]")
    print(f"Mask area  : {int(mask.sum())}")
    print(f"SAM score  : {detection_score:.4f}")
    print("=" * 60)

    status_message = "GT saved"
    return True


def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, drawing_roi
    global detection_mask, detection_score, sam_done, status_message

    if display is None:
        return

    if display.shape[:2] != image.shape[:2]:
        scale_x = image.shape[1] / display.shape[1]
        scale_y = image.shape[0] / display.shape[0]
        x = int(x * scale_x)
        y = int(y * scale_y)

    x = max(0, min(x, image.shape[1] - 1))
    y = max(0, min(y, image.shape[0] - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        drawing_roi = True
        detection_mask = None
        detection_score = 0.0
        sam_done = False
        status_message = "Selecting ROI..."

    elif event == cv2.EVENT_MOUSEMOVE and drawing_roi:
        roi_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        drawing_roi = False

        if get_roi() is None:
            status_message = "Invalid ROI"
        else:
            status_message = "ROI selected | Press S to run SAM2"


def main():
    global image, display, INPUT_IMAGE, status_message
    global roi_start, roi_end, detection_mask, detection_score, sam_done

    if not GT_DIR.is_dir():
        raise FileNotFoundError(f"GT folder not found: {GT_DIR}")

    image_path = find_gt_image()
    INPUT_IMAGE = str(image_path)

    image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to load image: {INPUT_IMAGE}")

    print("=" * 60)
    print("SAM2 OBJECT GT")
    print("=" * 60)
    print(f"Image     : {image_path}")
    print(f"SAM2      : {SAM2_CHECKPOINT}")
    print("=" * 60)
    print("Drag ROI")
    print("S   = Run SAM2 + Save")
    print("C   = Clear")
    print("ESC = Save if SAM2 exists, then Exit")
    print("=" * 60)

    prepare_predictor()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    status_message = "Drag ROI | S: SAM2 + Save | C: Clear | ESC: Save + Exit"

    while True:
        render()
        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            if sam_done and detection_mask is not None:
                save_gt()
            break

        if key in (ord("s"), ord("S")):
            if not sam_done:
                run_sam2()
            if sam_done:
                save_gt()

        elif key in (ord("c"), ord("C")):
            roi_start = None
            roi_end = None
            detection_mask = None
            detection_score = 0.0
            sam_done = False
            status_message = "Cleared"

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
