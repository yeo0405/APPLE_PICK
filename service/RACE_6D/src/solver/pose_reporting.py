"""Detection-aware pose statistics and projected-CAD evaluation overlays."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def _box_cxcywh_to_xyxy(boxes: torch.Tensor, width: float, height: float) -> torch.Tensor:
    scale = boxes.new_tensor([width, height, width, height])
    boxes = boxes * scale
    cx, cy, bw, bh = boxes.unbind(-1)
    return torch.stack((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), -1)


def _box_xyxy_to_normalized_cxcywh(
    boxes: torch.Tensor, width: float, height: float
) -> torch.Tensor:
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack(
        (
            (x0 + x1) / (2 * width),
            (y0 + y1) / (2 * height),
            (x1 - x0) / width,
            (y1 - y0) / height,
        ),
        -1,
    )


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((len(boxes1), len(boxes2)))
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(-1)
    return intersection / (area1[:, None] + area2[None, :] - intersection).clamp(min=1e-8)


def _greedy_matches(
    ious: torch.Tensor,
    pred_labels: torch.Tensor,
    target_labels: torch.Tensor,
    threshold: float,
) -> list[tuple[int, int, float]]:
    candidates = []
    for pred_index in range(len(pred_labels)):
        for target_index in range(len(target_labels)):
            if int(pred_labels[pred_index]) != int(target_labels[target_index]):
                continue
            iou = float(ious[pred_index, target_index])
            if iou >= threshold:
                candidates.append((iou, pred_index, target_index))
    candidates.sort(reverse=True)
    used_predictions: set[int] = set()
    used_targets: set[int] = set()
    matches = []
    for iou, pred_index, target_index in candidates:
        if pred_index in used_predictions or target_index in used_targets:
            continue
        used_predictions.add(pred_index)
        used_targets.add(target_index)
        matches.append((pred_index, target_index, iou))
    return matches


class PoseEvaluationReporter:
    """Accumulate thresholded pose metrics and optionally save CAD overlays."""

    def __init__(
        self,
        criterion,
        data_loader,
        overlay_dir: str | None = None,
        overlay_limit: int = 100,
        score_threshold: float = 0.25,
        iou_threshold: float = 0.5,
    ) -> None:
        self.criterion = criterion
        self.score_threshold = float(score_threshold)
        self.iou_threshold = float(iou_threshold)
        self.overlay_limit = max(0, int(overlay_limit))
        self.overlay_count = 0
        self.overlay_dir = Path(overlay_dir) if overlay_dir else None
        if self.overlay_dir:
            self.overlay_dir.mkdir(parents=True, exist_ok=True)

        dataset = getattr(data_loader, "dataset", None)
        self.coco = getattr(dataset, "coco", None)
        self.img_folder = Path(getattr(dataset, "img_folder", "."))
        self.counts = {"images": 0, "ground_truth": 0, "predictions": 0, "tp": 0, "fp": 0, "fn": 0}
        self.values = {
            "score": [],
            "bbox_iou": [],
            "tx_error_mm": [],
            "ty_error_mm": [],
            "tz_error_mm": [],
            "translation_error_mm": [],
            "rotation_error_deg": [],
            "add_r_error_mm": [],
            "add_r_error_fraction_diameter": [],
        }

    def update(self, outputs: dict, results: list[dict], targets: list[dict]) -> None:
        for batch_index, (result, target) in enumerate(zip(results, targets)):
            width, height = (float(value) for value in target["orig_size"].tolist())
            scores_all = result["scores"]
            keep = scores_all >= self.score_threshold
            scores = scores_all[keep]
            pred_labels = result["labels"][keep]
            pred_boxes = result["boxes"][keep]
            pred_rotations = result["rotations"][keep]
            pred_translation_codes = result["translations"][keep]

            target_labels = target["labels"]
            target_boxes = _box_cxcywh_to_xyxy(target["boxes"], width, height)
            target_poses = target["poses"].reshape(-1, 12)
            target_translations = target_poses[:, :3]
            target_rotations = target_poses[:, 3:].reshape(-1, 3, 3)

            if len(target["cam_K"]):
                camera = target["cam_K"][0].reshape(3, 3)
            else:
                continue
            pred_boxes_normalized = _box_xyxy_to_normalized_cxcywh(pred_boxes, width, height)
            pred_translations = self.criterion._c2t_pred(
                pred_translation_codes,
                camera,
                pred_boxes_normalized,
                width,
                height,
            )

            ious = _box_iou(pred_boxes, target_boxes)
            matches = _greedy_matches(
                ious, pred_labels, target_labels, self.iou_threshold
            )

            self.counts["images"] += 1
            self.counts["ground_truth"] += len(target_labels)
            self.counts["predictions"] += len(pred_labels)
            self.counts["tp"] += len(matches)
            self.counts["fp"] += len(pred_labels) - len(matches)
            self.counts["fn"] += len(target_labels) - len(matches)

            match_details = []
            for pred_index, target_index, iou in matches:
                translation_error = (
                    pred_translations[pred_index] - target_translations[target_index]
                ).abs()
                translation_norm = float(torch.linalg.vector_norm(translation_error))
                label = int(target_labels[target_index])
                rotation_error, add_r_error, add_r_fraction = self._pose_errors(
                    label,
                    pred_rotations[pred_index],
                    pred_translations[pred_index],
                    target_rotations[target_index],
                    target_translations[target_index],
                )
                detail = {
                    "pred_index": pred_index,
                    "target_index": target_index,
                    "score": float(scores[pred_index]),
                    "bbox_iou": iou,
                    "tx_error_mm": float(translation_error[0]),
                    "ty_error_mm": float(translation_error[1]),
                    "tz_error_mm": float(translation_error[2]),
                    "translation_error_mm": translation_norm,
                    "rotation_error_deg": rotation_error,
                    "add_r_error_mm": add_r_error,
                    "add_r_error_fraction_diameter": add_r_fraction,
                }
                match_details.append(detail)
                for key in self.values:
                    self.values[key].append(detail[key])

            if self.overlay_dir and self.overlay_count < self.overlay_limit:
                self._save_overlay(
                    target,
                    scores,
                    pred_labels,
                    pred_boxes,
                    pred_rotations,
                    pred_translations,
                    target_labels,
                    target_boxes,
                    target_rotations,
                    target_translations,
                    camera,
                    matches,
                    match_details,
                )

    def _pose_errors(
        self,
        label: int,
        pred_rotation: torch.Tensor,
        pred_translation: torch.Tensor,
        target_rotation: torch.Tensor,
        target_translation: torch.Tensor,
    ) -> tuple[float, float, float]:
        symmetries = self.criterion.sym_cache.get(label)
        if symmetries is None or len(symmetries) == 0:
            symmetries = torch.eye(3, device=pred_rotation.device).unsqueeze(0)
        equivalent_rotations = torch.matmul(target_rotation.unsqueeze(0), symmetries)
        relative = torch.matmul(
            pred_rotation.unsqueeze(0), equivalent_rotations.transpose(-1, -2)
        )
        traces = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
        angles = torch.acos(((traces - 1.0) / 2.0).clamp(-1.0, 1.0))
        rotation_error = float(torch.rad2deg(angles).min())

        model_points = self.criterion.points_3d_cache.get(label)
        if model_points is None or len(model_points) == 0:
            return rotation_error, float("nan"), float("nan")
        pred_points = torch.matmul(pred_rotation, model_points.T).T + pred_translation
        target_points = torch.einsum(
            "sij,pj->spi", equivalent_rotations, model_points
        ) + target_translation[None, None, :]
        add_r = torch.linalg.vector_norm(
            pred_points.unsqueeze(0) - target_points, dim=-1
        ).mean(-1).min()
        diameter = float(self.criterion.diameter_cache.get(label, 0.0))
        fraction = float(add_r) / diameter if diameter > 0 else float("nan")
        return rotation_error, float(add_r), fraction

    def _load_image(self, image_id: int) -> tuple[np.ndarray, str]:
        file_name = f"{image_id:06d}.png"
        if self.coco is not None:
            records = self.coco.loadImgs([image_id])
            if records:
                file_name = records[0]["file_name"]
        image = np.asarray(Image.open(self.img_folder / file_name).convert("RGB"))
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), file_name

    @staticmethod
    def _project(
        points: torch.Tensor,
        rotation: torch.Tensor,
        translation: torch.Tensor,
        camera: torch.Tensor,
    ) -> np.ndarray:
        camera_points = torch.matmul(rotation, points.T).T + translation
        valid = camera_points[:, 2] > 1e-3
        camera_points = camera_points[valid]
        pixels_h = torch.matmul(camera, camera_points.T).T
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
        return pixels.detach().cpu().numpy()

    @staticmethod
    def _draw_projected_points(
        image: np.ndarray, points: np.ndarray, color: tuple[int, int, int]
    ) -> None:
        height, width = image.shape[:2]
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        inside = (
            (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height)
        )
        for x, y in np.rint(points[inside]).astype(np.int32):
            cv2.circle(image, (int(x), int(y)), 1, color, -1, cv2.LINE_AA)

    @staticmethod
    def _draw_box(
        image: np.ndarray,
        box: torch.Tensor,
        color: tuple[int, int, int],
        text: str,
    ) -> None:
        values = box.detach().cpu().numpy().round().astype(int)
        x0, y0, x1, y1 = values.tolist()
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.putText(
            image,
            text,
            (max(0, x0), max(16, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def _save_overlay(
        self,
        target,
        scores,
        pred_labels,
        pred_boxes,
        pred_rotations,
        pred_translations,
        target_labels,
        target_boxes,
        target_rotations,
        target_translations,
        camera,
        matches,
        match_details,
    ) -> None:
        image_id = int(target["image_id"].item())
        image, file_name = self._load_image(image_id)
        cloud_layer = image.copy()
        matched_predictions = {pred_index for pred_index, _, _ in matches}

        for target_index, label_tensor in enumerate(target_labels):
            label = int(label_tensor)
            self._draw_box(image, target_boxes[target_index], (0, 220, 0), "GT")
            points = self.criterion.points_3d_cache.get(label)
            if points is not None:
                projected = self._project(
                    points,
                    target_rotations[target_index],
                    target_translations[target_index],
                    camera,
                )
                self._draw_projected_points(cloud_layer, projected, (0, 255, 0))

        details_by_prediction = {item["pred_index"]: item for item in match_details}
        for pred_index in range(len(scores)):
            if pred_index not in matched_predictions:
                self._draw_box(
                    image,
                    pred_boxes[pred_index],
                    (0, 165, 255),
                    f"unmatched {float(scores[pred_index]):.2f}",
                )
                continue
            detail = details_by_prediction[pred_index]
            self._draw_box(
                image,
                pred_boxes[pred_index],
                (0, 0, 255),
                f"P {detail['score']:.2f} IoU {detail['bbox_iou']:.2f}",
            )
            label = int(pred_labels[pred_index])
            points = self.criterion.points_3d_cache.get(label)
            if points is not None:
                projected = self._project(
                    points,
                    pred_rotations[pred_index],
                    pred_translations[pred_index],
                    camera,
                )
                self._draw_projected_points(cloud_layer, projected, (0, 0, 255))

        image = cv2.addWeighted(image, 0.55, cloud_layer, 0.45, 0)
        cv2.putText(
            image,
            "green=GT CAD/box  red=matched prediction  orange=unmatched",
            (8, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        output_name = f"{image_id:06d}_{Path(file_name).stem}.jpg"
        cv2.imwrite(str(self.overlay_dir / output_name), image)
        self.overlay_count += 1

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float | None]:
        array = np.asarray(values, dtype=np.float64)
        array = array[np.isfinite(array)]
        if len(array) == 0:
            return {"mean": None, "median": None, "p95": None}
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
        }

    def summarize(self) -> dict:
        tp, fp, fn = self.counts["tp"], self.counts["fp"], self.counts["fn"]
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        relative_add = np.asarray(self.values["add_r_error_fraction_diameter"], dtype=float)
        finite_add = relative_add[np.isfinite(relative_add)]
        stats = {
            "pose_score_threshold": self.score_threshold,
            "pose_iou_threshold": self.iou_threshold,
            **{f"pose_{key}": value for key, value in self.counts.items()},
            "pose_precision": precision,
            "pose_recall": recall,
            "pose_f1": f1,
            "pose_add_r_0.05d_accuracy": float((finite_add <= 0.05).mean()) if len(finite_add) else None,
            "pose_add_r_0.10d_accuracy": float((finite_add <= 0.10).mean()) if len(finite_add) else None,
            "pose_overlays_written": self.overlay_count,
        }
        for name, values in self.values.items():
            for statistic, value in self._distribution(values).items():
                stats[f"pose_{name}_{statistic}"] = value
        return stats
