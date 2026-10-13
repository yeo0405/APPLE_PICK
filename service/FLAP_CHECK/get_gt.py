#!/usr/bin/env python3

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.build_sam import _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
GT_DIR = SCRIPT_DIR / "GT"

SAM2_CONFIG = MODEL_DIR / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = MODEL_DIR / "sam2.1_hiera_small.pt"
OUTPUT_PATH = GT_DIR / "gt.json"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

MAX_POLYGONS = 2

image = None
display = None
status_message = ""

gt_image_path = None

polygons = []
current_polygon = []

drawing_polygon = False


def relative_to_project(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def find_gt_image():
    images = sorted(
        p for p in GT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(f"No GT image found in: {GT_DIR}")

    if len(images) > 1:
        names = "\n".join(f"  - {p.name}" for p in images)
        raise RuntimeError(
            f"GT directory must contain exactly one image, but found {len(images)}:\n{names}"
        )

    return images[0]


def load_sam2():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {SAM2_CHECKPOINT}")

    if not SAM2_CONFIG.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {SAM2_CONFIG}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(
        version_base=None,
        config_dir=str(MODEL_DIR),
    ):
        cfg = compose(
            config_name=SAM2_CONFIG.name,
            overrides=[
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ],
        )

    OmegaConf.resolve(cfg)

    model = instantiate(cfg.model)
    _load_checkpoint(model, str(SAM2_CHECKPOINT))
    model = model.to(device)
    model.eval()

    predictor = SAM2ImagePredictor(model)

    print(f"[SAM2] Device: {device}")
    print(f"[SAM2] Config: {SAM2_CONFIG}")
    print(f"[SAM2] Checkpoint: {SAM2_CHECKPOINT}")

    return predictor, device


def polygon_to_bbox(polygon):
    points = np.asarray(polygon, dtype=np.float32)

    if len(points) < 3:
        raise ValueError("Polygon must contain at least 3 points")

    x_min = int(np.floor(points[:, 0].min()))
    y_min = int(np.floor(points[:, 1].min()))
    x_max = int(np.ceil(points[:, 0].max()))
    y_max = int(np.ceil(points[:, 1].max()))

    h, w = image.shape[:2]

    x_min = max(0, min(x_min, w - 1))
    y_min = max(0, min(y_min, h - 1))
    x_max = max(0, min(x_max, w - 1))
    y_max = max(0, min(y_max, h - 1))

    return [x_min, y_min, x_max, y_max]


def predict_bbox(predictor, bbox):
    box = np.asarray(bbox, dtype=np.float32)

    try:
        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True,
            return_logits=False,
        )
    except Exception as e:
        print(f"[ERROR] SAM2 prediction failed: {e}")
        return None

    if masks is None or len(masks) == 0:
        return None

    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(scores).astype(float)

    if masks.ndim == 2:
        masks = masks[None, ...]

    best_idx = int(np.argmax(scores))
    best_mask = masks[best_idx]
    best_score = float(scores[best_idx])

    return {
        "mask": best_mask,
        "score": best_score,
        "area": int(np.count_nonzero(best_mask)),
        "selected_index": best_idx,
        "num_masks": len(masks),
        "all_scores": [float(v) for v in scores],
    }


def generate_sam_masks(predictor):
    global image, polygons

    if len(polygons) != MAX_POLYGONS:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        predictor.set_image(image_rgb)
    except Exception as e:
        print(f"[ERROR] SAM2 set_image failed: {e}")
        return None

    results = []

    for index, polygon in enumerate(polygons):
        bbox = polygon_to_bbox(polygon)

        result = predict_bbox(predictor, bbox)

        if result is None:
            return None

        side = "left" if index == 0 else "right"

        results.append({
            "index": index,
            "side": side,
            "polygon": polygon,
            "bbox": bbox,
            "mask": result["mask"],
            "score": result["score"],
            "area": result["area"],
            "selected_index": result["selected_index"],
            "num_masks": result["num_masks"],
            "all_scores": result["all_scores"],
        })

        print(
            f"[SAM2] {side.upper()}: "
            f"bbox={bbox}, "
            f"mask={result['selected_index']}/"
            f"{result['num_masks'] - 1}, "
            f"score={result['score']:.4f}, "
            f"area={result['area']}"
        )

        print(f"       scores={result['all_scores']}")

    return results


def save_sam_masks(results, output_path):
    output_path = Path(output_path)

    mask_dir = output_path.parent / f"{output_path.stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_infos = []

    for result in results:
        index = result["index"]
        mask_path = mask_dir / f"ear_{index}.png"

        mask = result["mask"].astype(np.uint8) * 255

        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to save mask: {mask_path}")

        mask_infos.append({
            "index": index,
            "side": result["side"],
            "file": relative_to_project(mask_path),
            "score": result["score"],
            "area": result["area"],
        })

    return mask_infos


def draw_polygon(image, polygon, color, thickness=3, closed=True):
    points = np.asarray(polygon, dtype=np.int32).reshape(-1, 1, 2)

    if len(points) >= 2:
        cv2.polylines(
            image,
            [points],
            closed,
            color,
            thickness,
        )

    for x, y in np.asarray(polygon, dtype=np.int32):
        cv2.circle(image, (int(x), int(y)), 5, color, -1)


def draw_sam_result(results):
    result_image = image.copy()

    for result in results:
        side = result["side"]
        polygon = result["polygon"]
        bbox = result["bbox"]
        mask = result["mask"]
        score = result["score"]

        mask_uint8 = mask.astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            result_image,
            contours,
            -1,
            (0, 255, 0),
            3,
        )

        draw_polygon(
            result_image,
            polygon,
            (255, 0, 0),
            3,
        )

        x1, y1, x2, y2 = bbox

        cv2.rectangle(
            result_image,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            1,
        )

        cv2.putText(
            result_image,
            f"{side} S={score:.2f}",
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            result_image,
            f"Area={result['area']}",
            (
                x1,
                min(result_image.shape[0] - 10, y2 + 30),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    return result_image


def redraw():
    global display

    if image is None:
        return

    display = image.copy()

    for index, polygon in enumerate(polygons):
        label = "Left" if index == 0 else "Right"

        draw_polygon(
            display,
            polygon,
            (255, 0, 0),
            3,
        )

        x, y = polygon[0]

        cv2.putText(
            display,
            label,
            (x, max(30, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 0, 0),
            2,
        )

    if current_polygon:
        draw_polygon(
            display,
            current_polygon,
            (0, 255, 255),
            2,
            closed=False,
        )

    if current_polygon and len(current_polygon) >= 2:
        pts = np.asarray(current_polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            display,
            [pts],
            False,
            (0, 255, 255),
            2,
        )

    if len(polygons) == 0:
        text = "LEFT: click points, ENTER to finish"
    elif len(polygons) == 1:
        text = "RIGHT: click points, ENTER to finish"
    else:
        text = "2 polygons selected - press S"

    cv2.putText(
        display,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    if status_message:
        cv2.putText(
            display,
            status_message,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )


def finish_polygon():
    global current_polygon
    global polygons
    global status_message

    if len(current_polygon) < 3:
        status_message = "Polygon needs at least 3 points"
        redraw()
        return

    if len(polygons) >= MAX_POLYGONS:
        status_message = "Maximum polygons reached"
        redraw()
        return

    polygons.append(current_polygon.copy())
    current_polygon = []

    status_message = (
        "LEFT polygon saved"
        if len(polygons) == 1
        else "RIGHT polygon saved"
    )

    redraw()


def mouse_callback(event, x, y, flags, param):
    global current_polygon
    global status_message

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(polygons) >= MAX_POLYGONS:
            return

        current_polygon.append((x, y))
        status_message = f"Points: {len(current_polygon)}"
        redraw()

    elif event == cv2.EVENT_RBUTTONDOWN:
        if current_polygon:
            current_polygon.pop()
            status_message = f"Points: {len(current_polygon)}"
            redraw()


def save_gt():
    global status_message
    global display

    if len(polygons) != MAX_POLYGONS:
        status_message = "ERROR: select LEFT and RIGHT polygon"
        redraw()
        return False

    if current_polygon:
        status_message = "Finish current polygon first"
        redraw()
        return False

    status_message = "Loading SAM2..."
    redraw()

    try:
        predictor, device = load_sam2()
    except Exception as e:
        status_message = f"SAM2 load failed: {e}"
        redraw()
        return False

    status_message = "Running SAM2..."
    redraw()

    results = generate_sam_masks(predictor)

    if results is None or len(results) != MAX_POLYGONS:
        status_message = "SAM2 failed"
        redraw()
        return False

    try:
        mask_infos = save_sam_masks(
            results,
            OUTPUT_PATH,
        )
    except Exception as e:
        status_message = f"Mask save failed: {e}"
        redraw()
        return False

    objects = []

    for result in results:
        objects.append({
            "index": result["index"],
            "side": result["side"],
            "polygon": [
                [int(x), int(y)]
                for x, y in result["polygon"]
            ],
            "bbox": result["bbox"],
            "mask": mask_infos[result["index"]],
        })

    data = {
        "image": relative_to_project(gt_image_path),
        "image_size": [
            int(image.shape[1]),
            int(image.shape[0]),
        ],
        "object_count": MAX_POLYGONS,
        "objects": objects,
        "sam2": {
            "checkpoint": relative_to_project(SAM2_CHECKPOINT),
            "config": relative_to_project(SAM2_CONFIG),
            "prompt_type": "polygon_to_bbox",
            "selection": "highest_score",
            "multimask_output": True,
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    }

    try:
        with open(
            OUTPUT_PATH,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                data,
                f,
                indent=4,
                ensure_ascii=False,
            )
    except Exception as e:
        status_message = f"JSON save failed: {e}"
        redraw()
        return False

    display = draw_sam_result(results)
    status_message = "GT SAVED!"

    cv2.imshow("GT Annotation", display)
    cv2.waitKey(1)

    print("===================================")
    print(f"Image : {gt_image_path}")
    print(f"JSON  : {OUTPUT_PATH}")
    print(
        f"Masks : "
        f"{OUTPUT_PATH.with_suffix('')}_masks"
    )
    print("===================================")

    return True


def main():
    global image
    global display
    global polygons
    global current_polygon
    global status_message
    global gt_image_path

    gt_image_path = find_gt_image()

    image = cv2.imread(str(gt_image_path))

    if image is None:
        raise RuntimeError(f"Cannot load GT image: {gt_image_path}")

    window_name = "GT Annotation"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
    )

    redraw()

    print("===================================")
    print("GT Polygon Annotation")
    print(f"Image: {gt_image_path}")
    print("===================================")
    print("LEFT: click polygon points first.")
    print("Press ENTER or SPACE to finish LEFT.")
    print("RIGHT: click polygon points second.")
    print("Press ENTER or SPACE to finish RIGHT.")
    print("Right click: remove last point.")
    print("Press S to run SAM2 and save.")
    print("Press R to reset.")
    print("Press ESC to exit.")
    print("===================================")

    while True:
        cv2.imshow(
            window_name,
            display,
        )

        key = cv2.waitKey(20)

        if key != -1:
            key &= 0xFF

        if key in (13, 32):
            finish_polygon()

        elif key in (ord("s"), ord("S")):
            save_gt()

        elif key in (ord("r"), ord("R")):
            polygons = []
            current_polygon = []
            status_message = ""
            redraw()

        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()