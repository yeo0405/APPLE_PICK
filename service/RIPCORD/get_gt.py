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
MAX_RIPCORDS = 4

image = None
display = None
status_message = ""

gt_image_path = None
roi = None
roi_start = None
roi_end = None
drawing_roi = False
ripcord_points = []


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
    model = instantiate(cfg.model, _recursive_=True)
    _load_checkpoint(model, str(SAM2_CHECKPOINT))
    model = model.to(device).eval()

    print(f"[SAM2] Device: {device}")
    print(f"[SAM2] Config: {SAM2_CONFIG}")
    print(f"[SAM2] Checkpoint: {SAM2_CHECKPOINT}")

    return SAM2ImagePredictor(model), device


def generate_sam_masks(predictor, device):
    global image, roi, ripcord_points

    if roi is None or len(ripcord_points) != MAX_RIPCORDS:
        return None

    x1, y1, x2, y2 = roi
    roi_image = image[y1:y2, x1:x2]

    if roi_image.size == 0:
        return None

    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)

    try:
        with torch.inference_mode():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    predictor.set_image(roi_rgb)
            else:
                predictor.set_image(roi_rgb)
    except Exception as e:
        print(f"[ERROR] SAM2 set_image failed: {e}")
        return None

    sorted_points = sorted(ripcord_points, key=lambda p: p[1])
    results = []

    for index, (gx, gy) in enumerate(sorted_points):
        rx, ry = gx - x1, gy - y1

        point_coords = np.array([[rx, ry]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        try:
            with torch.inference_mode():
                if device == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        masks, scores, _ = predictor.predict(
                            point_coords=point_coords,
                            point_labels=point_labels,
                            multimask_output=True,
                            return_logits=False,
                        )
                else:
                    masks, scores, _ = predictor.predict(
                        point_coords=point_coords,
                        point_labels=point_labels,
                        multimask_output=True,
                        return_logits=False,
                    )
        except Exception as e:
            print(f"[ERROR] Ripcord {index} SAM2 prediction failed: {e}")
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
        best_mask = masks[best_idx].astype(bool)
        best_score = float(scores[best_idx])

        results.append({
            "index": index,
            "global_center": [int(gx), int(gy)],
            "roi_center": [int(rx), int(ry)],
            "mask": best_mask,
            "score": best_score,
            "area": int(np.count_nonzero(best_mask)),
            "selected_index": best_idx,
            "all_scores": [float(v) for v in scores],
        })

    return results


def save_sam_masks(results):
    mask_dir = GT_DIR / "gt_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    infos = []

    for result in results:
        index = result["index"]
        mask_path = mask_dir / f"ripcord_{index}.png"
        mask = result["mask"].astype(np.uint8) * 255

        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to save mask: {mask_path}")

        infos.append({
            "index": index,
            "file": relative_to_project(mask_path),
            "score": result["score"],
            "area": result["area"],
        })

    return infos


def draw_sam_debug(results):
    global image, roi

    debug = image.copy()
    x1, y1, x2, y2 = roi

    overlay = debug.copy()

    for result in results:
        mask = result["mask"]
        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask.astype(np.uint8) * 255

        overlay[full_mask > 0] = (0, 180, 255)

    debug = cv2.addWeighted(overlay, 0.30, debug, 0.70, 0)

    cv2.rectangle(
        debug,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        3,
    )

    for result in results:
        index = result["index"]
        gx, gy = result["global_center"]
        score = result["score"]

        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = result["mask"].astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            full_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        cv2.drawContours(
            debug,
            contours,
            -1,
            (0, 255, 0),
            2,
        )

        cv2.circle(
            debug,
            (gx, gy),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            debug,
            f"Ripcord {index} S={score:.2f}",
            (gx + 12, gy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return debug


def save_gt():
    global status_message, display

    if roi is None:
        status_message = "ERROR: ROI not selected"
        redraw()
        return False

    if len(ripcord_points) != MAX_RIPCORDS:
        status_message = f"ERROR: Need {MAX_RIPCORDS} ripcord points"
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

    results = generate_sam_masks(predictor, device)

    if results is None or len(results) != MAX_RIPCORDS:
        status_message = "ERROR: SAM2 failed"
        redraw()
        return False

    try:
        mask_infos = save_sam_masks(results)
    except Exception as e:
        status_message = f"Mask save failed: {e}"
        redraw()
        return False

    sorted_points = sorted(ripcord_points, key=lambda p: p[1])

    ripcords = []

    for index, ((x, y), mask_info) in enumerate(zip(sorted_points, mask_infos)):
        ripcords.append({
            "index": index,
            "center": [int(x), int(y)],
            "mask": mask_info,
        })

    data = {
        "image": relative_to_project(gt_image_path),
        "image_size": [
            int(image.shape[1]),
            int(image.shape[0]),
        ],
        "roi": [int(v) for v in roi],
        "ripcord_count": MAX_RIPCORDS,
        "ripcords": ripcords,
        "sam2": {
            "checkpoint": relative_to_project(SAM2_CHECKPOINT),
            "config": relative_to_project(SAM2_CONFIG),
            "prompt_type": "point",
            "multimask_output": True,
            "selection": "highest_score",
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    }

    try:
        GT_DIR.mkdir(parents=True, exist_ok=True)

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

    debug = draw_sam_debug(results)
    debug_path = GT_DIR / "gt_debug.jpg"

    if not cv2.imwrite(str(debug_path), debug):
        print(f"[WARNING] Failed to save debug image: {debug_path}")

    display = debug
    status_message = "GT SAVED!"

    cv2.imshow("GT Annotation", display)
    cv2.waitKey(1)

    print("===================================")
    print("GT Annotation Saved")
    print(f"Image : {gt_image_path}")
    print(f"JSON  : {OUTPUT_PATH}")
    print(f"Masks : {GT_DIR / 'gt_masks'}")
    print(f"Debug : {debug_path}")
    print("===================================")

    for result in results:
        print(
            f"Ripcord {result['index']}: "
            f"center={result['global_center']} "
            f"score={result['score']:.4f} "
            f"area={result['area']}"
        )

    return True


def mouse_callback(event, x, y, flags, param):
    global roi_start, roi_end, drawing_roi, roi, ripcord_points

    if event == cv2.EVENT_LBUTTONDOWN:
        if roi is None:
            roi_start = (x, y)
            roi_end = (x, y)
            drawing_roi = True
            redraw()
        elif len(ripcord_points) < MAX_RIPCORDS:
            ripcord_points.append((x, y))
            redraw()

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_roi:
            roi_end = (x, y)
            redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_roi:
            drawing_roi = False

            x1, x2 = sorted([
                roi_start[0],
                roi_end[0],
            ])

            y1, y2 = sorted([
                roi_start[1],
                roi_end[1],
            ])

            if x2 - x1 < 5 or y2 - y1 < 5:
                roi = None
                roi_start = None
                roi_end = None
                redraw()
                return

            roi = (x1, y1, x2, y2)
            roi_start = None
            roi_end = None
            redraw()


def redraw():
    global display

    if image is None:
        return

    display = image.copy()

    if drawing_roi and roi_start and roi_end:
        x1, x2 = sorted([
            roi_start[0],
            roi_end[0],
        ])

        y1, y2 = sorted([
            roi_start[1],
            roi_end[1],
        ])

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 255, 0),
            2,
        )

    if roi:
        x1, y1, x2, y2 = roi

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3,
        )

        cv2.putText(
            display,
            "ROI",
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    for i, (x, y) in enumerate(ripcord_points):
        cv2.circle(
            display,
            (x, y),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            display,
            str(i),
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if roi is None:
        text = "Drag mouse to select ROI"
    elif len(ripcord_points) < MAX_RIPCORDS:
        text = f"Click ripcord {len(ripcord_points)}/{MAX_RIPCORDS}"
    else:
        text = "4 ripcords selected - press S to save"

    cv2.putText(
        display,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    if status_message:
        cv2.putText(
            display,
            status_message,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


def main():
    global image, display, gt_image_path
    global roi_start, roi_end, drawing_roi, roi
    global ripcord_points, status_message

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
    print("Ripcord GT Annotation")
    print(f"Image: {gt_image_path}")
    print("===================================")
    print("1. Drag mouse to select ROI.")
    print("2. Click the 4 ripcords.")
    print("3. Ripcords are sorted top-to-bottom.")
    print("4. Press S to run SAM2 and save.")
    print("5. Press R to reset.")
    print("6. Press ESC to exit.")
    print("===================================")

    while True:
        cv2.imshow(
            window_name,
            display,
        )

        key = cv2.waitKey(20)

        if key != -1:
            key &= 0xFF

        if key in (ord("s"), ord("S")):
            save_gt()

        elif key in (ord("r"), ord("R")):
            roi_start = None
            roi_end = None
            drawing_roi = False
            roi = None
            ripcord_points = []
            status_message = ""
            redraw()

        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
