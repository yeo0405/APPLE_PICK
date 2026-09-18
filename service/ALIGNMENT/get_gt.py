#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

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
POINT_RADIUS = 8
POINT_THICKNESS = 2
FONT_SCALE = 0.65
FONT_THICKNESS = 2

image = None
display = None
predictor = None
INPUT_IMAGE = ""

roi_start = None
roi_end = None
drawing_roi = False

positive_points = []
negative_points = []

detection_mask = None
detection_score = 0.0
status_message = ""


def relative_to_project(path: str | Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def find_gt_image() -> Path:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in GT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    if not images:
        raise FileNotFoundError(f"No image found in GT folder: {GT_DIR}")
    if len(images) > 1:
        names = "\n".join(f"  {p.name}" for p in images)
        raise RuntimeError(
            f"Expected exactly one image in {GT_DIR}, found {len(images)}:\n{names}"
        )
    return images[0]


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


def load_sam2():
    global predictor

    if not SAM2_CONFIG.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {SAM2_CONFIG}")

    if not SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[SAM2] Device     : {device}")
    print(f"[SAM2] Config     : {SAM2_CONFIG}")
    print(f"[SAM2] Checkpoint : {SAM2_CHECKPOINT}")

    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(
        config_dir=str(SAM2_CONFIG.parent),
        version_base=None,
    ):
        cfg = compose(config_name=SAM2_CONFIG.stem)

    model = instantiate(cfg.model, _recursive_=True)

    checkpoint = torch.load(
        SAM2_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device)
    model.eval()

    predictor = SAM2ImagePredictor(model)

    print("[SAM2] Model loaded")


def prepare_predictor():
    roi = get_roi()

    if roi is None:
        return None

    x1, y1, x2, y2 = roi
    crop = image[y1:y2, x1:x2]

    predictor.set_image(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    )

    return roi


def run_sam2():
    global detection_mask
    global detection_score
    global status_message

    roi = prepare_predictor()

    if roi is None:
        detection_mask = None
        status_message = "Draw a valid ROI first"
        redraw()
        return

    if not positive_points:
        detection_mask = None
        status_message = "Add at least one positive point (+)"
        redraw()
        return

    x1, y1, _, _ = roi

    points = positive_points + negative_points

    labels = (
        [1] * len(positive_points)
        + [0] * len(negative_points)
    )

    local_points = np.array(
        [
            [x - x1, y - roi[1]]
            for x, y in points
        ],
        dtype=np.float32,
    )

    point_labels = np.array(
        labels,
        dtype=np.int32,
    )

    print(
        f"[SAM2] Positive={len(positive_points)} "
        f"Negative={len(negative_points)}"
    )

    try:
        masks, scores, _ = predictor.predict(
            point_coords=local_points,
            point_labels=point_labels,
            multimask_output=True,
        )
    except Exception as error:
        print(f"[ERROR] SAM2 prediction failed: {error}")
        detection_mask = None
        status_message = "SAM2 prediction failed"
        redraw()
        return

    valid = [
        i
        for i, mask in enumerate(masks)
        if int(np.count_nonzero(mask)) >= MIN_MASK_AREA
    ]

    if not valid:
        detection_mask = None
        status_message = "No valid SAM2 mask"
        redraw()
        return

    best = max(
        valid,
        key=lambda i: float(scores[i]),
    )

    detection_mask = masks[best].astype(bool)
    detection_score = float(scores[best])

    area = int(np.count_nonzero(detection_mask))

    print(
        f"[SAM2] Selected mask={best} "
        f"score={detection_score:.4f} "
        f"area={area}"
    )

    status_message = (
        f"SAM2 | Score={detection_score:.3f} | "
        f"Area={area} | "
        f"+{len(positive_points)} -{len(negative_points)}"
    )

    redraw()


def overlay_mask(
    output: np.ndarray,
    mask: np.ndarray,
    color,
    alpha: float,
):
    if mask is None or not np.any(mask):
        return

    output[mask] = cv2.addWeighted(
        output[mask],
        1.0 - alpha,
        np.full_like(output[mask], color),
        alpha,
        0,
    )


def get_full_mask() -> np.ndarray | None:
    roi = get_roi()

    if roi is None or detection_mask is None:
        return None

    x1, y1, x2, y2 = roi

    full_mask = np.zeros(
        image.shape[:2],
        dtype=bool,
    )

    full_mask[
        y1:y2,
        x1:x2
    ] = detection_mask

    return full_mask


def redraw():
    global display

    if image is None:
        return

    display = image.copy()

    roi = get_roi()

    if roi is not None:
        x1, y1, x2, y2 = roi

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2,
        )

        cv2.putText(
            display,
            "ROI",
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 0, 0),
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    full_mask = get_full_mask()

    if full_mask is not None:
        overlay_mask(
            display,
            full_mask,
            (0, 255, 0),
            MASK_ALPHA,
        )

        mask_u8 = (
            full_mask.astype(np.uint8) * 255
        )

        contours, _ = cv2.findContours(
            mask_u8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if contours:
            cv2.drawContours(
                display,
                contours,
                -1,
                (0, 255, 0),
                3,
            )

    for x, y in positive_points:
        cv2.circle(
            display,
            (x, y),
            POINT_RADIUS,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            display,
            "+",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    for x, y in negative_points:
        cv2.circle(
            display,
            (x, y),
            POINT_RADIUS,
            (0, 0, 255),
            -1,
        )

        cv2.line(
            display,
            (x - 7, y - 7),
            (x + 7, y + 7),
            (255, 255, 255),
            POINT_THICKNESS,
        )

        cv2.line(
            display,
            (x + 7, y - 7),
            (x - 7, y + 7),
            (255, 255, 255),
            POINT_THICKNESS,
        )

    cv2.putText(
        display,
        "Left Drag: ROI | Left Click: + | Middle Click: - | "
        "R: Refine | S: Save | C: Clear | ESC: Exit",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
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
            cv2.LINE_AA,
        )

    cv2.putText(
        display,
        f"Points: +{len(positive_points)} -{len(negative_points)}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if detection_mask is not None:
        cv2.putText(
            display,
            f"SAM2: Score={detection_score:.3f} "
            f"Area={int(np.count_nonzero(detection_mask))}",
            (20, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if status_message:
        cv2.putText(
            display,
            status_message,
            (20, 164),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end
    global drawing_roi
    global positive_points, negative_points
    global detection_mask
    global status_message

    if event == cv2.EVENT_LBUTTONDOWN:
        if get_roi() is None:
            roi_start = (x, y)
            roi_end = (x, y)
            drawing_roi = True
            detection_mask = None
            positive_points.clear()
            negative_points.clear()
            status_message = "Drawing ROI..."
            redraw()
            return

        roi = get_roi()

        if roi is None:
            return

        x1, y1, x2, y2 = roi

        if x1 <= x < x2 and y1 <= y < y2:
            positive_points.append((x, y))
            status_message = (
                "Positive point added - press R"
            )
            redraw()

    elif event == cv2.EVENT_MOUSEMOVE and drawing_roi:
        roi_end = (x, y)
        redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        if not drawing_roi:
            return

        roi_end = (x, y)
        drawing_roi = False

        roi = get_roi()

        if roi is None:
            roi_start = None
            roi_end = None
            status_message = "Invalid ROI"
        else:
            positive_points.clear()
            negative_points.clear()
            detection_mask = None
            status_message = (
                "ROI selected - add positive/negative points"
            )

        redraw()

    elif event == cv2.EVENT_MBUTTONDOWN:
        roi = get_roi()

        if roi is None:
            status_message = "Draw ROI first"
            redraw()
            return

        x1, y1, x2, y2 = roi

        if x1 <= x < x2 and y1 <= y < y2:
            negative_points.append((x, y))
            status_message = (
                "Negative point added - press R"
            )
            redraw()


def save_gt() -> bool:
    global status_message

    full_mask = get_full_mask()

    if full_mask is None:
        print("[ERROR] No valid SAM2 mask to save")
        status_message = "No mask to save"
        redraw()
        return False

    area = int(np.count_nonzero(full_mask))

    if area < MIN_MASK_AREA:
        print("[ERROR] Mask too small")
        status_message = "Mask too small"
        redraw()
        return False

    roi = get_roi()

    if roi is None:
        print("[ERROR] No valid ROI")
        status_message = "No valid ROI"
        redraw()
        return False

    mask_dir = GT_DIR / "gt_masks"
    mask_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_path = mask_dir / "object_0.png"
    json_path = GT_DIR / "gt.json"

    if not cv2.imwrite(
        str(mask_path),
        full_mask.astype(np.uint8) * 255,
    ):
        print(
            f"[ERROR] Failed to save mask: {mask_path}"
        )
        return False

    x1, y1, x2, y2 = roi

    ys, xs = np.where(full_mask)

    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    ]

    data = {
        "image": relative_to_project(INPUT_IMAGE),
        "image_size": [
            int(image.shape[1]),
            int(image.shape[0]),
        ],
        "object_count": 1,
        "objects": [{
            "index": 0,
            "side": "object",
            "bbox": bbox,
            "roi": [
                int(x1),
                int(y1),
                int(x2),
                int(y2),
            ],
            "positive_points": [
                [int(x), int(y)]
                for x, y in positive_points
            ],
            "negative_points": [
                [int(x), int(y)]
                for x, y in negative_points
            ],
            "mask": {
                "file": relative_to_project(mask_path),
                "area": area,
                "sam_score": detection_score,
            },
        }],
        "sam2": {
            "config": relative_to_project(
                SAM2_CONFIG
            ),
            "checkpoint": relative_to_project(
                SAM2_CHECKPOINT
            ),
        },
    }

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print("=" * 60)
    print("GT SAVED")
    print("=" * 60)
    print(f"Image     : {INPUT_IMAGE}")
    print(f"JSON      : {json_path}")
    print(f"Mask      : {mask_path}")
    print(f"ROI       : [{x1}, {y1}, {x2}, {y2}]")
    print(f"BBox      : {bbox}")
    print(f"Positive  : {len(positive_points)}")
    print(f"Negative  : {len(negative_points)}")
    print(f"SAM score : {detection_score:.4f}")
    print(f"Mask area : {area}")
    print("=" * 60)

    status_message = "GT saved successfully"
    redraw()

    return True


def main():
    global image, display, INPUT_IMAGE
    global roi_start, roi_end
    global positive_points, negative_points
    global detection_mask

    if not GT_DIR.is_dir():
        raise FileNotFoundError(
            f"GT folder not found: {GT_DIR}"
        )

    input_path = find_gt_image()
    INPUT_IMAGE = str(input_path)

    image = cv2.imread(INPUT_IMAGE)

    if image is None:
        raise RuntimeError(
            f"Cannot load image: {INPUT_IMAGE}"
        )

    print("=" * 60)
    print("SAM2 OBJECT GT ANNOTATION")
    print("=" * 60)
    print(f"Image : {INPUT_IMAGE}")
    print(f"Config: {SAM2_CONFIG}")
    print(f"Model : {SAM2_CHECKPOINT}")
    print("=" * 60)
    print("Left Drag    = Draw ROI")
    print("Left Click   = Positive point (+)")
    print("Middle Click = Negative point (-)")
    print("R            = Refine SAM2")
    print("S            = Save GT")
    print("C            = Clear")
    print("ESC          = Exit")
    print("=" * 60)

    load_sam2()

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_NAME,
        WINDOW_WIDTH,
        WINDOW_HEIGHT,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
    )

    redraw()

    while True:
        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        key = cv2.waitKey(20)

        if key != -1:
            key &= 0xFF

        if key in (ord("r"), ord("R")):
            run_sam2()

        elif key in (ord("s"), ord("S")):
            save_gt()

        elif key in (ord("c"), ord("C")):
            roi_start = None
            roi_end = None
            drawing_roi = False
            positive_points.clear()
            negative_points.clear()
            detection_mask = None
            status_message = "Cleared - draw ROI"
            redraw()

        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
