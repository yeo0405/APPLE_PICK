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

NUM_GTS = 2  # GT 图片数量
BOXES_PER_GT = 2  # 每张图的 BBox 数量（left / right）
MIN_BOX_SIZE = 5
SIDES = ("left", "right")
WINDOW_NAME = "GT Annotation"

# ---- 全局状态 ----
image = None  # 当前显示的图像
display = None
status_message = ""

gt_image_paths = []  # [Path, Path]
gt_images = []  # [ndarray, ndarray]
all_boxes = [[] for _ in range(NUM_GTS)]
gt_entries = [
    None
] * NUM_GTS  # 每张图保存后: {"entry": json 用 dict, "results": SAM2 结果}
current_idx = 0

sam_predictor = None  # SAM2 只加载一次

drawing_box = False
start_point = None
current_box = None


def relative_to_project(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def find_gt_images():
    images = sorted(
        p
        for p in GT_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) != NUM_GTS:
        names = "\n".join(f"  - {p.name}" for p in images) or "  (none)"
        raise RuntimeError(
            f"GT directory must contain exactly {NUM_GTS} images, "
            f"but found {len(images)}:\n{names}"
        )

    return images


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

    return predictor


def normalize_bbox(x1, y1, x2, y2):
    h, w = image.shape[:2]

    x1, x2 = sorted((int(x1), int(x2)))
    y1, y2 = sorted((int(y1), int(y2)))

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))

    return [x1, y1, x2, y2]


def bbox_is_valid(bbox):
    x1, y1, x2, y2 = bbox

    return x2 - x1 >= MIN_BOX_SIZE and y2 - y1 >= MIN_BOX_SIZE


def current_saved():
    return gt_entries[current_idx] is not None


def switch_image(new_idx):
    """切换当前显示的 GT 图。"""
    global current_idx, image, drawing_box, start_point, current_box

    current_idx = new_idx
    image = gt_images[current_idx]
    drawing_box = False
    start_point = None
    current_box = None


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

    if len(scores) != len(masks):
        print("[ERROR] SAM2 mask and score counts do not match")
        return None

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


def generate_sam_masks(predictor, gt_idx):
    """对第 gt_idx 张图的所有 BBox 运行 SAM2。"""
    gt_image = gt_images[gt_idx]
    boxes = all_boxes[gt_idx]

    if len(boxes) != BOXES_PER_GT:
        return None

    image_rgb = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

    try:
        predictor.set_image(image_rgb)
    except Exception as e:
        print(f"[ERROR] SAM2 set_image failed (GT {gt_idx}): {e}")
        return None

    results = []

    for index, bbox in enumerate(boxes):
        result = predict_bbox(predictor, bbox)

        if result is None:
            return None

        side = SIDES[index]

        results.append(
            {
                "gt_id": gt_idx,
                "index": index,
                "side": side,
                "bbox": bbox.copy(),
                "mask": result["mask"],
                "score": result["score"],
                "area": result["area"],
                "selected_index": result["selected_index"],
                "num_masks": result["num_masks"],
                "all_scores": result["all_scores"],
            }
        )

        print(
            f"[SAM2] GT{gt_idx} {side.upper()}: "
            f"bbox={bbox}, "
            f"mask={result['selected_index']}/"
            f"{result['num_masks'] - 1}, "
            f"score={result['score']:.4f}, "
            f"area={result['area']}"
        )
        print(f"       scores={result['all_scores']}")

    return results


def save_sam_masks(results, gt_idx, output_path):
    output_path = Path(output_path)

    mask_dir = output_path.parent / f"{output_path.stem}_masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    mask_infos = []

    for result in results:
        index = result["index"]
        side = result["side"]
        mask_path = mask_dir / f"gt{gt_idx}_{side}.png"

        mask = result["mask"].astype(np.uint8) * 255

        if not cv2.imwrite(str(mask_path), mask):
            raise RuntimeError(f"Failed to save mask: {mask_path}")

        mask_infos.append(
            {
                "index": index,
                "side": side,
                "file": relative_to_project(mask_path),
                "score": result["score"],
                "area": result["area"],
            }
        )

    return mask_infos


def write_json():
    """把已完成的 GT 写入 gt.json（按 id 排序）。"""
    gts = [e["entry"] for e in gt_entries if e is not None]
    gts.sort(key=lambda g: g["id"])

    data = {
        "gt_count": len(gts),
        "expected_gt_count": NUM_GTS,
        "gts": gts,
        "sam2": {
            "checkpoint": relative_to_project(SAM2_CHECKPOINT),
            "config": relative_to_project(SAM2_CONFIG),
            "prompt_type": "bbox",
            "selection": "highest_score",
            "multimask_output": True,
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


def draw_bbox(
    canvas,
    bbox,
    color,
    thickness=2,
    label=None,
):
    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        canvas,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    if label:
        cv2.putText(
            canvas,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )


def draw_sam_result(base_image, results):
    result_image = base_image.copy()

    for result in results:
        side = result["side"]
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

        draw_bbox(
            result_image,
            bbox,
            (255, 0, 0),
            2,
            f"{side} S={score:.2f}",
        )

        x1, y1, x2, y2 = bbox

        cv2.putText(
            result_image,
            f"Area={result['area']}",
            (
                x1,
                min(result_image.shape[0] - 10, y2 + 25),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

    return result_image


def put_text(canvas, text, y, scale=0.8, color=(0, 255, 0)):
    cv2.putText(
        canvas,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
    )


def redraw():
    global display

    if image is None:
        return

    title = f"[GT {current_idx + 1}/{NUM_GTS}] {gt_image_paths[current_idx].name}"
    boxes = all_boxes[current_idx]
    last = NUM_GTS - 1

    # 当前这张已保存：显示 SAM2 结果
    if current_saved():
        display = draw_sam_result(image, gt_entries[current_idx]["results"])
        put_text(display, title, 40)

        if all(e is not None for e in gt_entries):
            hint = "ALL GT SAVED - B: review, R: redo this image, ESC: exit"
        elif current_idx < last:
            hint = "SAVED - press N for next image, R: redo this image"
        else:
            hint = "SAVED - B: previous image, R: redo this image"

        put_text(display, hint, 80, 0.7)

        if status_message:
            put_text(display, status_message, 120, 0.7)
        return

    # 未保存：显示 BBox 标注界面
    display = image.copy()

    for index, bbox in enumerate(boxes):
        label = "Left" if index == 0 else "Right"

        draw_bbox(
            display,
            bbox,
            (255, 0, 0),
            2,
            label,
        )

    if drawing_box and start_point is not None and current_box is not None:
        draw_bbox(
            display,
            current_box,
            (0, 255, 255),
            2,
            "Selecting",
        )

    if len(boxes) == 0:
        text = "LEFT: drag mouse to draw BBox"
    elif len(boxes) == 1:
        text = "RIGHT: drag mouse to draw BBox"
    else:
        text = f"2 BBoxes ready - press S to run SAM2 and save GT {current_idx}"

    put_text(display, title, 40)
    put_text(display, text, 80)

    if status_message:
        put_text(display, status_message, 120, 0.7)


def mouse_callback(event, x, y, flags, param):
    global drawing_box
    global start_point
    global current_box
    global status_message

    if current_saved():
        if event == cv2.EVENT_LBUTTONDOWN:
            status_message = "This GT is saved - press R to redo it"
            redraw()
        return

    boxes = all_boxes[current_idx]

    h, w = image.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(boxes) >= BOXES_PER_GT:
            status_message = "Maximum BBoxes reached - press S to save or R to reset"
            redraw()
            return

        drawing_box = True
        start_point = (x, y)
        current_box = [x, y, x, y]
        status_message = "Drag to select BBox"
        redraw()

    elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
        current_box = normalize_bbox(
            start_point[0],
            start_point[1],
            x,
            y,
        )
        redraw()

    elif event == cv2.EVENT_LBUTTONUP and drawing_box:
        drawing_box = False

        if start_point is None:
            current_box = None
            redraw()
            return

        bbox = normalize_bbox(
            start_point[0],
            start_point[1],
            x,
            y,
        )

        if not bbox_is_valid(bbox):
            status_message = f"BBox too small: minimum {MIN_BOX_SIZE}px"
        else:
            boxes.append(bbox)

            status_message = (
                "LEFT BBox saved" if len(boxes) == 1 else "RIGHT BBox saved"
            )

        start_point = None
        current_box = None
        redraw()


def save_current_gt():
    """对当前这张图运行 SAM2，保存 mask，并更新 gt.json。"""
    global status_message
    global sam_predictor

    if current_saved():
        status_message = "This GT is already saved - press R to redo it"
        redraw()
        return False

    if drawing_box:
        status_message = "Finish drawing BBox first"
        redraw()
        return False

    boxes = all_boxes[current_idx]

    if len(boxes) != BOXES_PER_GT:
        status_message = f"ERROR: need {BOXES_PER_GT} BBoxes ({len(boxes)} drawn)"
        redraw()
        return False

    if sam_predictor is None:
        status_message = "Loading SAM2..."
        redraw()
        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(1)

        try:
            sam_predictor = load_sam2()
        except Exception as e:
            status_message = f"SAM2 load failed: {e}"
            redraw()
            print(f"[ERROR] SAM2 load failed: {e}")
            return False

    status_message = "Running SAM2..."
    redraw()
    cv2.imshow(WINDOW_NAME, display)
    cv2.waitKey(1)

    results = generate_sam_masks(sam_predictor, current_idx)

    if results is None or len(results) != BOXES_PER_GT:
        status_message = "SAM2 failed"
        redraw()
        return False

    try:
        mask_infos = save_sam_masks(
            results,
            current_idx,
            OUTPUT_PATH,
        )
    except Exception as e:
        status_message = f"Mask save failed: {e}"
        redraw()
        print(f"[ERROR] Mask save failed: {e}")
        return False

    gt_image = gt_images[current_idx]

    objects = []
    for result in results:
        objects.append(
            {
                "index": result["index"],
                "side": result["side"],
                "bbox": [int(v) for v in result["bbox"]],
                "mask": mask_infos[result["index"]],
            }
        )

    entry = {
        "id": current_idx,
        "image": relative_to_project(gt_image_paths[current_idx]),
        "image_size": [
            int(gt_image.shape[1]),
            int(gt_image.shape[0]),
        ],
        "object_count": BOXES_PER_GT,
        "objects": objects,
    }

    gt_entries[current_idx] = {
        "entry": entry,
        "results": results,
    }

    try:
        write_json()
    except Exception as e:
        gt_entries[current_idx] = None
        status_message = f"JSON save failed: {e}"
        redraw()
        print(f"[ERROR] JSON save failed: {e}")
        return False

    done = sum(e is not None for e in gt_entries)

    if done == NUM_GTS:
        status_message = "ALL GT SAVED!"
    elif current_idx < NUM_GTS - 1:
        status_message = f"GT {current_idx} SAVED - press N for next image"
    else:
        status_message = f"GT {current_idx} SAVED - go back (B) to finish the others"

    redraw()

    print("===================================")
    print(f"GT{current_idx} image: {gt_image_paths[current_idx]}")
    print(f"JSON  : {OUTPUT_PATH} ({done}/{NUM_GTS} GT saved)")
    print(f"Masks : {OUTPUT_PATH.with_suffix('')}_masks")
    print("===================================")

    return True


def main():
    global image
    global display
    global all_boxes
    global status_message
    global gt_image_paths
    global gt_images

    gt_image_paths = find_gt_images()

    gt_images = []
    for path in gt_image_paths:
        img = cv2.imread(str(path))

        if img is None:
            raise RuntimeError(f"Cannot load GT image: {path}")

        gt_images.append(img)

    switch_image(0)

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        WINDOW_NAME,
        mouse_callback,
    )

    redraw()

    print("===================================")
    print(f"GT BBox Annotation ({NUM_GTS} GT images, one at a time)")
    for i, path in enumerate(gt_image_paths):
        print(f"  GT{i}: {path}")
    print("===================================")
    print("1. Drag LEFT BBox, then RIGHT BBox on the current image.")
    print("2. Press S to run SAM2 and save this GT.")
    print("3. Press N to move on to the next image (only after saving).")
    print("Press B to go back, R to redo the current image, ESC to exit.")
    print("===================================")

    while True:
        cv2.imshow(
            WINDOW_NAME,
            display,
        )

        key = cv2.waitKey(20)

        if key != -1:
            key &= 0xFF

        if key in (ord("s"), ord("S")):
            save_current_gt()

        elif key in (ord("n"), ord("N")):
            if current_idx >= NUM_GTS - 1:
                status_message = "Already the last image"
            elif not current_saved():
                status_message = "Press S to save this GT first"
            else:
                switch_image(current_idx + 1)
                status_message = ""
            redraw()

        elif key in (ord("b"), ord("B")):
            if current_idx > 0:
                switch_image(current_idx - 1)
                status_message = ""
            else:
                status_message = "Already the first image"
            redraw()

        elif key in (ord("r"), ord("R")):
            all_boxes[current_idx] = []
            gt_entries[current_idx] = None
            switch_image(current_idx)
            status_message = "Reset - redraw BBoxes, then press S to save"
            redraw()

        elif key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
