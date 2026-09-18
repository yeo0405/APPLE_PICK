#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"
YOLO_CHECKPOINT = MODEL_DIR / "yolo_ripcord.pt"

SCRIPT_DIR = Path(__file__).resolve().parent
GT_DIR = SCRIPT_DIR / "GT"
INPUT_IMAGE = ""

WINDOW_NAME = "YOLO Ripcord GT"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

CONFIDENCE_THRESHOLD = 0.01
IOU_THRESHOLD = 0.7
KEEP_LARGEST_COMPONENT = True
MIN_MASK_AREA = 100

SELECTED_CONTOUR_THICKNESS = 4
MASK_ALPHA = 0.35
FONT_SCALE = 0.65
FONT_THICKNESS = 2

image = None
display = None
model = None
detection = None
roi_start = None
roi_end = None
drawing_roi = False
status_message = ""


def relative_to_project(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def find_gt_image() -> Path:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in GT_DIR.iterdir() if p.is_file() and p.suffix.lower() in extensions)
    if not images:
        raise FileNotFoundError(f"No image found in GT folder: {GT_DIR}")
    if len(images) > 1:
        names = "\n".join(f"  {p.name}" for p in images)
        raise RuntimeError(f"Expected exactly one image in {GT_DIR}, found {len(images)}:\n{names}")
    return images[0]


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    if int(stats[idx, cv2.CC_STAT_AREA]) < MIN_MASK_AREA:
        return np.zeros_like(mask, dtype=bool)
    return labels == idx


def get_largest_contour(mask: np.ndarray):
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None


def load_yolo():
    global model
    if not YOLO_CHECKPOINT.is_file():
        raise FileNotFoundError(f"YOLO checkpoint not found: {YOLO_CHECKPOINT}")
    print(f"[YOLO] Loading model: {YOLO_CHECKPOINT}")
    model = YOLO(str(YOLO_CHECKPOINT))
    print("[YOLO] Model loaded")


def get_roi():
    if roi_start is None or roi_end is None:
        return None
    x1, x2 = sorted((roi_start[0], roi_end[0]))
    y1, y2 = sorted((roi_start[1], roi_end[1]))
    x1 = max(0, min(x1, image.shape[1] - 1))
    x2 = max(0, min(x2, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    y2 = max(0, min(y2, image.shape[0] - 1))
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None
    return x1, y1, x2, y2


def run_yolo():
    global detection, status_message
    roi = get_roi()
    if model is None or roi is None:
        detection = None
        status_message = "Draw a valid ROI first"
        redraw()
        return

    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]

    print(f"[YOLO] ROI: [{x1}, {y1}, {x2}, {y2}]")
    print("[YOLO] Running inference...")

    try:
        results = model.predict(
            source=crop,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False,
        )
    except Exception as error:
        print(f"[ERROR] YOLO prediction failed: {error}")
        detection = None
        status_message = "YOLO prediction failed"
        redraw()
        return

    if not results or results[0].masks is None or results[0].boxes is None:
        detection = None
        status_message = "No YOLO mask found inside ROI"
        print("[YOLO] No masks detected")
        redraw()
        return

    result = results[0]
    masks = result.masks.data.cpu().numpy()
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    names = result.names

    candidates = []

    for i, mask_data in enumerate(masks):
        mask = cv2.resize(
            mask_data.astype(np.uint8),
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        if KEEP_LARGEST_COMPONENT:
            mask = keep_largest_component(mask)

        area = int(np.count_nonzero(mask))
        if area < MIN_MASK_AREA:
            continue

        rx1, ry1, rx2, ry2 = boxes[i]
        full_mask = np.zeros(image.shape[:2], dtype=bool)
        full_mask[y1:y2, x1:x2] = mask

        candidates.append({
            "mask": full_mask,
            "bbox": [float(rx1 + x1), float(ry1 + y1), float(rx2 + x1), float(ry2 + y1)],
            "confidence": float(confidences[i]),
            "class_id": int(classes[i]),
            "class_name": names.get(int(classes[i]), str(int(classes[i]))),
            "area": area,
        })

    if not candidates:
        detection = None
        status_message = "No valid YOLO mask inside ROI"
        redraw()
        return

    detection = max(candidates, key=lambda d: d["confidence"])

    print(
        f"[YOLO] Candidates={len(candidates)} | "
        f"Selected class={detection['class_name']} "
        f"conf={detection['confidence']:.4f} "
        f"area={detection['area']}"
    )

    status_message = (
        f"Ripcord | Conf={detection['confidence']:.3f} | "
        f"Area={detection['area']}"
    )
    redraw()


def overlay_mask(output: np.ndarray, mask: np.ndarray, color, alpha: float):
    mask_bool = mask.astype(bool)
    if not np.any(mask_bool):
        return
    output[mask_bool] = cv2.addWeighted(
        output[mask_bool], 1.0 - alpha,
        np.full_like(output[mask_bool], color), alpha, 0
    )


def redraw():
    global display
    if image is None:
        return

    display = image.copy()

    roi = get_roi()
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(
            display,
            "ROI",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 0, 0),
            FONT_THICKNESS,
        )

    if detection is not None:
        mask = detection["mask"]
        overlay_mask(display, mask, (0, 255, 0), MASK_ALPHA)

        contour = get_largest_contour(mask)
        if contour is not None:
            cv2.drawContours(
                display,
                [contour],
                -1,
                (0, 255, 0),
                SELECTED_CONTOUR_THICKNESS,
            )

        x1, y1, x2, y2 = map(int, detection["bbox"])
        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        label = (
            f"Ripcord {detection['confidence']:.3f} "
            f"Area={detection['area']}"
        )
        cv2.putText(
            display,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (0, 255, 0),
            FONT_THICKNESS,
        )

    cv2.putText(
        display,
        "Drag: ROI | R: Detect | S: Save | C: Clear | ESC: Exit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
    )

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.putText(
            display,
            f"ROI: [{x1}, {y1}, {x2}, {y2}]",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    if detection is not None:
        cv2.putText(
            display,
            f"Detected: Conf={detection['confidence']:.3f}",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    if status_message:
        cv2.putText(
            display,
            status_message,
            (20, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )


def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, drawing_roi, detection, status_message

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        drawing_roi = True
        detection = None
        status_message = "Release mouse to run YOLO"
        redraw()

    elif event == cv2.EVENT_MOUSEMOVE and drawing_roi:
        roi_end = (x, y)
        redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        drawing_roi = False
        if get_roi() is None:
            status_message = "Invalid ROI"
            redraw()
            return
        run_yolo()


def save_gt() -> bool:
    global status_message

    if detection is None:
        print("[ERROR] No YOLO detection to save")
        status_message = "No detection to save"
        redraw()
        return False

    roi = get_roi()
    if roi is None:
        print("[ERROR] No valid ROI")
        status_message = "No valid ROI"
        redraw()
        return False

    mask = detection["mask"]
    mask_area = int(np.count_nonzero(mask))

    if mask_area < MIN_MASK_AREA:
        print("[ERROR] Selected mask is too small")
        status_message = "Mask too small"
        redraw()
        return False

    output_path = GT_DIR / "gt.json"
    mask_dir = GT_DIR / "gt_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    mask_path = mask_dir / "object_0.png"

    if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
        print(f"[ERROR] Failed to save mask: {mask_path}")
        return False

    x1, y1, x2, y2 = detection["bbox"]
    rx1, ry1, rx2, ry2 = roi

    data = {
        "image": relative_to_project(INPUT_IMAGE),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "object_count": 1,
        "objects": [{
            "index": 0,
            "side": "object",
            "bbox": [
                int(round(x1)),
                int(round(y1)),
                int(round(x2)),
                int(round(y2)),
            ],
            "roi": [
                int(rx1),
                int(ry1),
                int(rx2),
                int(ry2),
            ],
            "class_id": int(detection["class_id"]),
            "class_name": detection["class_name"],
            "confidence": float(detection["confidence"]),
            "mask": {
                "file": relative_to_project(mask_path),
                "area": mask_area,
            },
        }],
        "yolo": {
            "checkpoint": relative_to_project(YOLO_CHECKPOINT),
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "roi_required": True,
            "keep_largest_component": KEEP_LARGEST_COMPONENT,
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("=" * 60)
    print("GT SAVED")
    print("=" * 60)
    print(f"Image      : {INPUT_IMAGE}")
    print(f"JSON       : {output_path}")
    print(f"Mask       : {mask_path}")
    print(f"ROI        : [{rx1}, {ry1}, {rx2}, {ry2}]")
    print(f"BBox       : [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
    print(f"Class      : {detection['class_name']} ({detection['class_id']})")
    print(f"Confidence : {detection['confidence']:.4f}")
    print(f"Mask area  : {mask_area}")
    print("=" * 60)

    status_message = "GT saved successfully"
    redraw()
    return True


def main():
    global image, display, INPUT_IMAGE

    if not GT_DIR.is_dir():
        raise FileNotFoundError(f"GT folder not found: {GT_DIR}")

    input_path = find_gt_image()
    INPUT_IMAGE = str(input_path)
    image = cv2.imread(INPUT_IMAGE)

    if image is None:
        raise RuntimeError(f"Cannot load image: {INPUT_IMAGE}")

    print("=" * 60)
    print("YOLO RIPCORD GT ANNOTATION")
    print("=" * 60)
    print(f"Image : {INPUT_IMAGE}")
    print(f"Model : {YOLO_CHECKPOINT}")
    print("=" * 60)
    print("Drag = Select ROI")
    print("R    = Re-run YOLO")
    print("S    = Save GT")
    print("C    = Clear ROI")
    print("ESC  = Exit")
    print("=" * 60)

    load_yolo()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, WINDOW_WIDTH, WINDOW_HEIGHT)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    redraw()

    while True:
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20)
        if key != -1:
            key &= 0xFF

        if key in (ord("s"), ord("S")):
            save_gt()
        elif key in (ord("r"), ord("R")):
            run_yolo()
        elif key in (ord("c"), ord("C")):
            roi_start = None
            roi_end = None
            detection = None
            status_message = "ROI cleared"
            redraw()
        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()