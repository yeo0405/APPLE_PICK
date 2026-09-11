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
MODEL_DIR = PROJECT_ROOT / "model"

SAM2_CONFIG = MODEL_DIR / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = MODEL_DIR / "sam2.1_hiera_small.pt"

SCRIPT_DIR = Path(__file__).resolve().parent
GT_DIR = SCRIPT_DIR / "GT"

image = None
display = None
INPUT_IMAGE = ""
status_message = ""

MAX_BBOXES = 2
bboxes = []
drawing_bbox = False
bbox_start = None
bbox_end = None


def relative_to_project(path):
    """Return path relative to PROJECT_ROOT for JSON output."""
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def find_gt_image():
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in GT_DIR.iterdir() if p.is_file() and p.suffix.lower() in extensions)

    if len(images) == 0:
        raise FileNotFoundError(f"No image found in GT folder: {GT_DIR}")

    if len(images) > 1:
        names = "\n".join(f"  {p.name}" for p in images)
        raise RuntimeError(f"Expected exactly one image in {GT_DIR}, found {len(images)}:\n{names}")

    return images[0]


def build_sam2_from_file(config_path, checkpoint_path, device="cuda"):
    config_path = Path(config_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"SAM2 config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_path.parent)):
        cfg = compose(
            config_name=config_path.stem,
            overrides=[
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            ],
        )
        OmegaConf.resolve(cfg)
        model = instantiate(cfg.model, _recursive_=True)

    _load_checkpoint(model, str(checkpoint_path))
    model = model.to(device)
    model.eval()

    return model


def load_sam2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2_from_file(SAM2_CONFIG, SAM2_CHECKPOINT, device)
    return SAM2ImagePredictor(model), device


def generate_sam_masks(predictor, device):
    global image, bboxes

    if len(bboxes) != MAX_BBOXES:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with torch.inference_mode():
        if device == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictor.set_image(image_rgb)
        else:
            predictor.set_image(image_rgb)

    results = []

    for index, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        box = np.array([x1, y1, x2, y2], dtype=np.float32)

        try:
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, _ = predictor.predict(
                            box=box, multimask_output=True, return_logits=False
                        )
                else:
                    masks, scores, _ = predictor.predict(
                        box=box, multimask_output=True, return_logits=False
                    )
        except Exception as e:
            print(f"SAM2 prediction failed for bbox {index}: {e}")
            return None

        masks = np.asarray(masks)
        scores = np.asarray(scores)

        if masks.ndim == 4:
            masks = masks[0]
        if scores.ndim > 1:
            scores = scores[0]
        if len(masks) == 0:
            return None

        best_idx = int(np.argmax(scores))
        best_mask = np.asarray(masks[best_idx], dtype=bool)

        results.append({
            "index": index,
            "side": "left" if index == 0 else "right",
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "mask": best_mask,
            "score": float(scores[best_idx]),
            "area": int(np.count_nonzero(best_mask)),
        })

    return results


def save_sam_masks(results, output_path):
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    mask_dir = output_dir / f"{output_path.stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_infos = []

    for result in results:
        index = result["index"]
        mask = result["mask"]
        mask_path = mask_dir / f"ripcord_{index}.png"

        if not cv2.imwrite(str(mask_path), mask.astype(np.uint8) * 255):
            raise RuntimeError(f"Failed to save SAM2 mask: {mask_path}")

        mask_infos.append({
            "index": index,
            "side": result["side"],
            "file": relative_to_project(mask_path),
            "score": result["score"],
            "area": result["area"],
        })

    return mask_infos


def draw_sam_result(results):
    global image

    result_image = image.copy()

    for result in results:
        side = result["side"]
        mask = result["mask"]
        score = result["score"]
        x1, y1, x2, y2 = result["bbox"]

        mask_uint8 = mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(result_image, contours, -1, (0, 255, 0), 3)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv2.putText(
            result_image, f"{side} S={score:.2f}", (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2
        )
        cv2.putText(
            result_image, f"Area={result['area']}",
            (x1, min(result_image.shape[0] - 10, y2 + 30)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

    return result_image


def mouse_callback(event, x, y, flags, param):
    global bbox_start, bbox_end, drawing_bbox, bboxes

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(bboxes) < MAX_BBOXES and not drawing_bbox:
            bbox_start = (x, y)
            bbox_end = (x, y)
            drawing_bbox = True
            redraw()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_bbox:
            bbox_end = (x, y)
            redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_bbox:
            drawing_bbox = False
            bbox_end = (x, y)

            x1, x2 = sorted([bbox_start[0], bbox_end[0]])
            y1, y2 = sorted([bbox_start[1], bbox_end[1]])

            if x2 - x1 < 5 or y2 - y1 < 5:
                bbox_start = None
                bbox_end = None
                redraw()
                return

            bboxes.append((x1, y1, x2, y2))
            bbox_start = None
            bbox_end = None
            redraw()


def redraw():
    global display

    if image is None:
        return

    display = image.copy()

    for index, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = bbox
        label = "Left" if index == 0 else "Right"

        cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(
            display, label, (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2
        )

    if drawing_bbox and bbox_start and bbox_end:
        x1, x2 = sorted([bbox_start[0], bbox_end[0]])
        y1, y2 = sorted([bbox_start[1], bbox_end[1]])
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)

    if len(bboxes) == 0:
        text = "Drag mouse to select LEFT bbox"
    elif len(bboxes) == 1:
        text = "Drag mouse to select RIGHT bbox"
    else:
        text = "2 bboxes selected - press S to run SAM2"

    cv2.putText(
        display, text, (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
    )

    if status_message:
        cv2.putText(
            display, status_message, (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )


def save_gt(output_path):
    global status_message, display

    if len(bboxes) != MAX_BBOXES:
        status_message = "ERROR: Need LEFT and RIGHT bboxes"
        redraw()
        return False

    status_message = "Loading SAM2..."
    redraw()

    try:
        predictor, device = load_sam2()
    except Exception as e:
        status_message = f"ERROR: SAM2 load failed: {e}"
        redraw()
        return False

    status_message = "Running SAM2..."
    redraw()

    results = generate_sam_masks(predictor, device)

    if results is None or len(results) != MAX_BBOXES:
        status_message = "ERROR: SAM2 failed"
        redraw()
        return False

    try:
        mask_infos = save_sam_masks(results, output_path)
    except Exception as e:
        status_message = f"ERROR: Mask save failed: {e}"
        redraw()
        return False

    objects = [
        {
            "index": result["index"],
            "side": result["side"],
            "bbox": result["bbox"],
            "mask": mask_infos[result["index"]],
        }
        for result in results
    ]

    data = {
        "image": relative_to_project(INPUT_IMAGE),
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "object_count": MAX_BBOXES,
        "objects": objects,
        "sam2": {
            "config": relative_to_project(SAM2_CONFIG),
            "checkpoint": relative_to_project(SAM2_CHECKPOINT),
            "prompt_type": "box",
        },
    }

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e:
        status_message = f"ERROR: JSON save failed: {e}"
        redraw()
        return False

    display = draw_sam_result(results)
    status_message = "SAM2 SAVED!"

    cv2.imshow("GT Annotation", display)
    cv2.waitKey(1)

    print(f"Image: {INPUT_IMAGE}")
    print(f"JSON: {output_path}")
    print(f"Masks: {Path(output_path).with_suffix('')}_masks")

    return True


def main():
    global image, display, INPUT_IMAGE
    global bboxes, drawing_bbox, bbox_start, bbox_end, status_message

    if not GT_DIR.is_dir():
        raise FileNotFoundError(f"GT folder not found: {GT_DIR}")

    input_path = find_gt_image()
    INPUT_IMAGE = str(input_path)

    output_path = GT_DIR / "gt.json"

    image = cv2.imread(INPUT_IMAGE)
    if image is None:
        raise RuntimeError(f"Cannot load image: {INPUT_IMAGE}")

    window_name = "GT Annotation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    redraw()

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20)

        if key != -1:
            key &= 0xFF

        if key in (ord("s"), ord("S")):
            save_gt(str(output_path))
        elif key in (ord("r"), ord("R")):
            bboxes = []
            drawing_bbox = False
            bbox_start = None
            bbox_end = None
            status_message = ""
            redraw()
        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()