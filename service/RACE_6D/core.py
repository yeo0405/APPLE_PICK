"""Reusable in-process RACE-6D pose estimator for ROS."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch


RACE_ROOT = Path(__file__).resolve().parent

if str(RACE_ROOT) not in sys.path:
    sys.path.insert(0, str(RACE_ROOT))

import src.zoo  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402


def _load_checkpoint(model: torch.nn.Module, path: str) -> None:
    state = torch.load(path, map_location="cpu", weights_only=False)

    if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
        weights = state["ema"]["module"]
    elif "model" in state:
        weights = state["model"]
    else:
        weights = state

    missing, unexpected = model.load_state_dict(weights, strict=False)

    if missing or unexpected:
        raise RuntimeError(
            "RACE checkpoint/model mismatch: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)

    if m.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {m.shape}")

    trace = float(np.trace(m))

    if trace > 0:
        s = 2 * np.sqrt(trace + 1.0)
        q = [
            0.25 * s,
            (m[2, 1] - m[1, 2]) / s,
            (m[0, 2] - m[2, 0]) / s,
            (m[1, 0] - m[0, 1]) / s,
        ]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(
            1.0 + m[0, 0] - m[1, 1] - m[2, 2]
        )
        q = [
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(
            1.0 + m[1, 1] - m[0, 0] - m[2, 2]
        )
        q = [
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ]
    else:
        s = 2 * np.sqrt(
            1.0 + m[2, 2] - m[0, 0] - m[1, 1]
        )
        q = [
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ]

    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)

    if norm < 1e-8:
        raise RuntimeError("RACE-6D produced an invalid rotation")

    return q / norm


class PoseEstimator:
    """RACE-6D model that takes BGR/depth camera arrays and returns pose detections."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str = "",
        score_threshold: float = 0.25,
        max_per_class: int = 1,
        max_detections: Optional[int] = None,
        depth_z_max_mm: Optional[float] = None,
        invalid_depth_value: int = 65535,
        class_id: Optional[int] = None,
        crop_x: int = 240,
        crop_y: int = 0,
        crop_width: Optional[int] = 1440,
        crop_height: Optional[int] = 1080,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.score_threshold = float(score_threshold)
        self.max_per_class = int(max_per_class)
        self.max_detections = max_detections
        self.depth_z_max_mm = depth_z_max_mm
        self.invalid_depth_value = int(invalid_depth_value)
        self.class_id = class_id

        self.crop_x = int(crop_x)
        self.crop_y = int(crop_y)
        self.crop_width = (
            None if crop_width is None else int(crop_width)
        )
        self.crop_height = (
            None if crop_height is None else int(crop_height)
        )

        if not 0 <= self.score_threshold <= 1:
            raise ValueError("score_threshold must be 0..1")

        if self.max_per_class < 1:
            raise ValueError("max_per_class must be >= 1")

        if self.max_detections is not None and self.max_detections < 1:
            raise ValueError("max_detections must be >= 1")

        if self.crop_x < 0 or self.crop_y < 0:
            raise ValueError("crop_x/crop_y must be >= 0")

        if self.crop_width is not None and self.crop_width <= 0:
            raise ValueError("crop_width must be > 0")

        if self.crop_height is not None and self.crop_height <= 0:
            raise ValueError("crop_height must be > 0")

        self.cfg = YAMLConfig(config_path)

        for name in ("PResNet", "PResNet_depth"):
            if name in self.cfg.yaml_cfg:
                self.cfg.yaml_cfg[name]["pretrained"] = False

        self.model = self.cfg.model.to(self.device).eval()
        self.postprocessor = self.cfg.postprocessor.to(self.device).eval()
        self.criterion = self.cfg.criterion.to(self.device).eval()

        self.criterion.set_pose_source(self.model.decoder)

        _load_checkpoint(self.model, model_path)

    def _crop(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
        camera: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        height, width = bgr.shape[:2]

        if self.crop_width is None:
            x0 = self.crop_x
            x1 = width - self.crop_x
        else:
            x0 = self.crop_x
            x1 = x0 + self.crop_width

        if self.crop_height is None:
            y0 = self.crop_y
            y1 = height - self.crop_y
        else:
            y0 = self.crop_y
            y1 = y0 + self.crop_height

        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise ValueError(
                f"Invalid crop {(x0, y0, x1, y1)} for image {width}x{height}"
            )

        cropped_bgr = bgr[y0:y1, x0:x1]
        cropped_depth = depth[y0:y1, x0:x1]

        cropped_camera = camera.copy()
        cropped_camera[0, 2] -= x0
        cropped_camera[1, 2] -= y0

        crop_height = y1 - y0
        crop_width = x1 - x0

        return (
            cropped_bgr,
            cropped_depth,
            cropped_camera,
            crop_height,
            crop_width,
        )

    def _image(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
    ) -> tuple[torch.Tensor, int, int]:
        if (
            bgr is None
            or depth is None
            or bgr.ndim != 3
            or bgr.shape[2] != 3
            or depth.ndim != 2
            or depth.shape != bgr.shape[:2]
        ):
            raise ValueError(
                "RACE requires aligned BGR HxWx3 and depth HxW; "
                f"got {getattr(bgr, 'shape', None)}, "
                f"{getattr(depth, 'shape', None)}"
            )

        height, width = bgr.shape[:2]

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        in_h, in_w = self.cfg.yaml_cfg.get(
            "eval_spatial_size",
            [height, width],
        )

        model_rgb = cv2.resize(
            rgb,
            (int(in_w), int(in_h)),
            interpolation=cv2.INTER_LINEAR,
        )

        rgb_tensor = (
            torch.from_numpy(np.ascontiguousarray(model_rgb))
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )

        channels = (
            self.model.backbone.conv1[0].conv.in_channels
        )

        if channels == 4:
            d = cv2.resize(
                depth,
                (int(in_w), int(in_h)),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)

            d[d == self.invalid_depth_value] = 0

            cap = float(
                self.depth_z_max_mm
                or self.cfg.yaml_cfg.get("val_dataloader", {})
                .get("dataset", {})
                .get("depth_z_max_mm", 2000.0)
            )

            if cap <= 0:
                raise ValueError("depth_z_max_mm must be positive")

            depth_tensor = (
                torch.from_numpy(np.clip(d, 0, cap) / cap)
                .unsqueeze(0)
            )

            image = torch.cat((rgb_tensor, depth_tensor), dim=0)

        elif channels == 3:
            image = rgb_tensor

        else:
            raise RuntimeError(
                f"Unsupported RACE backbone input channels: {channels}"
            )

        return image.unsqueeze(0).to(self.device), height, width

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: Sequence[Sequence[float]],
    ) -> Dict[str, Any]:
        camera = np.asarray(intrinsic, dtype=np.float32)

        if camera.shape != (3, 3):
            raise ValueError(
                f"Expected 3x3 camera matrix, got {camera.shape}"
            )

        original_height, original_width = rgb.shape[:2]

        # 1920x1080 -> 1440x1080 for 1002 geometry.
        (
            rgb_crop,
            depth_crop,
            camera_crop,
            height,
            width,
        ) = self._crop(rgb, depth, camera)

        image, height, width = self._image(
            rgb_crop,
            depth_crop,
        )

        size = torch.tensor(
            [[width, height]],
            dtype=torch.float32,
            device=self.device,
        )

        with torch.inference_mode():
            result = self.postprocessor(
                self.model(image),
                size,
            )[0]

        scores = result["scores"]
        labels = result["labels"]

        keep = torch.nonzero(
            scores >= self.score_threshold,
            as_tuple=False,
        ).squeeze(1)

        if len(keep) == 0:
            return {
                "success": False,
                "detections": [],
                "debug_image": rgb.copy(),
            }

        ordered = keep[
            torch.argsort(scores[keep], descending=True)
        ]

        selected = []
        counts = {}

        for index in ordered.tolist():
            label = int(labels[index])

            if self.class_id is not None and label != self.class_id:
                continue

            if counts.get(label, 0) >= self.max_per_class:
                continue

            selected.append(index)
            counts[label] = counts.get(label, 0) + 1

            if (
                self.max_detections is not None
                and len(selected) >= self.max_detections
            ):
                break

        if not selected:
            return {
                "success": False,
                "detections": [],
                "debug_image": rgb.copy(),
            }

        detections = []
        debug = rgb.copy()

        x_offset = self.crop_x
        y_offset = self.crop_y

        for index in selected:
            box = result["boxes"][index]

            normalized = box.clone()
            normalized[0::2] /= width
            normalized[1::2] /= height

            normalized = torch.stack(
                (
                    (normalized[0] + normalized[2]) / 2,
                    (normalized[1] + normalized[3]) / 2,
                    normalized[2] - normalized[0],
                    normalized[3] - normalized[1],
                )
            ).unsqueeze(0)

            translation_mm = (
                self.criterion._c2t_pred(
                    result["translations"][index].unsqueeze(0),
                    torch.from_numpy(camera_crop).to(self.device),
                    normalized,
                    width,
                    height,
                )[0]
                .detach()
                .cpu()
                .numpy()
            )

            rotation = (
                result["rotations"][index]
                .detach()
                .cpu()
                .numpy()
            )

            quat = _matrix_to_quat_wxyz(rotation)

            score = float(
                result["scores"][index]
                .detach()
                .cpu()
            )

            label = int(result["labels"][index])

            bbox_crop = (
                box.detach()
                .cpu()
                .numpy()
            )

            bbox_xyxy = bbox_crop.copy()
            bbox_xyxy[0::2] += x_offset
            bbox_xyxy[1::2] += y_offset
            bbox_xyxy = bbox_xyxy.tolist()

            x0, y0, x1, y1 = np.rint(
                np.asarray(bbox_xyxy)
            ).astype(int)

            cv2.rectangle(
                debug,
                (x0, y0),
                (x1, y1),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                debug,
                f"RACE label={label} score={score:.3f}",
                (x0, max(18, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

            detections.append(
                {
                    "label": label,
                    "confidence": score,
                    "translation": (
                        translation_mm / 1000.0
                    ).tolist(),
                    "translation_mm": translation_mm.tolist(),
                    "quat": quat.tolist(),
                    "rotation": rotation.tolist(),
                    "bbox_xyxy": bbox_xyxy,
                }
            )

        return {
            "success": len(detections) > 0,
            "detections": detections,
            "num_detections": len(detections),
            "class_counts": {
                str(label): count
                for label, count in counts.items()
            },
            "debug_image": debug,
            "input_debug": {
                "original_size": [original_width, original_height],
                "crop_size": [width, height],
                "crop_xy": [self.crop_x, self.crop_y],
                "camera_crop": camera_crop.tolist(),
                "eval_size": [
                    int(self.cfg.yaml_cfg["eval_spatial_size"][0]),
                    int(self.cfg.yaml_cfg["eval_spatial_size"][1]),
                ],
            },
        }
