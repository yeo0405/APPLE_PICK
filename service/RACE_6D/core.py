"""Reusable in-process RACE-6D pose estimator for ROS."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
import torch


RACE_ROOT = Path(__file__).resolve().parent

if str(RACE_ROOT) not in sys.path:
    sys.path.insert(0, str(RACE_ROOT))

import src.zoo  # noqa: F401,E402
from src.core import YAMLConfig  # noqa: E402


CROP_X = 240
CROP_WIDTH = 1440
CROP_HEIGHT = 1080
MODEL_WIDTH = 640
MODEL_HEIGHT = 480


def _load_checkpoint(
    model: torch.nn.Module,
    path: str,
) -> None:
    state = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        "ema" in state
        and isinstance(state["ema"], dict)
        and "module" in state["ema"]
    ):
        weights = state["ema"]["module"]
    elif "model" in state:
        weights = state["model"]
    else:
        weights = state

    missing, unexpected = model.load_state_dict(
        weights,
        strict=False,
    )

    if missing or unexpected:
        raise RuntimeError(
            "RACE checkpoint/model mismatch: "
            f"missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )


def _matrix_to_quat_wxyz(
    matrix: np.ndarray,
) -> np.ndarray:
    m = np.asarray(
        matrix,
        dtype=np.float64,
    )

    if m.shape != (3, 3):
        raise ValueError(
            f"Expected 3x3 rotation matrix, got {m.shape}"
        )

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
            1.0
            + m[0, 0]
            - m[1, 1]
            - m[2, 2]
        )
        q = [
            (m[2, 1] - m[1, 2]) / s,
            0.25 * s,
            (m[0, 1] + m[1, 0]) / s,
            (m[0, 2] + m[2, 0]) / s,
        ]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(
            1.0
            + m[1, 1]
            - m[0, 0]
            - m[2, 2]
        )
        q = [
            (m[0, 2] - m[2, 0]) / s,
            (m[0, 1] + m[1, 0]) / s,
            0.25 * s,
            (m[1, 2] + m[2, 1]) / s,
        ]
    else:
        s = 2 * np.sqrt(
            1.0
            + m[2, 2]
            - m[0, 0]
            - m[1, 1]
        )
        q = [
            (m[1, 0] - m[0, 1]) / s,
            (m[0, 2] + m[2, 0]) / s,
            (m[1, 2] + m[2, 1]) / s,
            0.25 * s,
        ]

    q = np.asarray(
        q,
        dtype=np.float32,
    )

    norm = np.linalg.norm(q)

    if norm < 1e-8:
        raise RuntimeError(
            "RACE-6D produced an invalid rotation"
        )

    return q / norm


class PoseEstimator:
    """
    RACE-6D pose estimator.

    Input:
        RGB   : 1920x1080 BGR
        Depth : 1920x1080
        K     : original 1920x1080 camera intrinsic

    Internal preprocessing:
        1920x1080
            -> crop 240 px left/right
        1440x1080
            -> resize
        640x480

    The model uses the intrinsic matrix corresponding to
    the final 640x480 image. Returned pose translation
    remains in the original camera coordinate system.
    """

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
    ) -> None:
        self.device = torch.device(
            device
            or (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.score_threshold = float(
            score_threshold
        )
        self.max_per_class = int(
            max_per_class
        )
        self.max_detections = max_detections
        self.depth_z_max_mm = (
            depth_z_max_mm
        )
        self.invalid_depth_value = int(
            invalid_depth_value
        )
        self.class_id = class_id

        if not 0 <= self.score_threshold <= 1:
            raise ValueError(
                "score_threshold must be 0..1"
            )

        if self.max_per_class < 1:
            raise ValueError(
                "max_per_class must be >= 1"
            )

        if (
            self.max_detections is not None
            and self.max_detections < 1
        ):
            raise ValueError(
                "max_detections must be >= 1"
            )

        self.cfg = YAMLConfig(
            config_path
        )

        for name in (
            "PResNet",
            "PResNet_depth",
        ):
            if name in self.cfg.yaml_cfg:
                self.cfg.yaml_cfg[name][
                    "pretrained"
                ] = False

        self.model = (
            self.cfg.model
            .to(self.device)
            .eval()
        )

        self.postprocessor = (
            self.cfg.postprocessor
            .to(self.device)
            .eval()
        )

        self.criterion = (
            self.cfg.criterion
            .to(self.device)
            .eval()
        )

        self.criterion.set_pose_source(
            self.model.decoder
        )

        _load_checkpoint(
            self.model,
            model_path,
        )

    @staticmethod
    def _crop_and_resize(
        image: np.ndarray,
        interpolation: int,
    ) -> np.ndarray:
        height, width = image.shape[:2]

        if (
            width != 1920
            or height != 1080
        ):
            raise ValueError(
                "RACE input must be 1920x1080, "
                f"got {width}x{height}"
            )

        cropped = image[
            :CROP_HEIGHT,
            CROP_X:CROP_X + CROP_WIDTH,
        ]

        if cropped.shape[:2] != (
            CROP_HEIGHT,
            CROP_WIDTH,
        ):
            raise RuntimeError(
                "Unexpected crop size: "
                f"{cropped.shape}"
            )

        return cv2.resize(
            cropped,
            (
                MODEL_WIDTH,
                MODEL_HEIGHT,
            ),
            interpolation=interpolation,
        )

    @staticmethod
    def _update_intrinsic(
        camera: np.ndarray,
    ) -> np.ndarray:
        camera = np.asarray(
            camera,
            dtype=np.float32,
        )

        if camera.shape != (3, 3):
            raise ValueError(
                "Expected 3x3 camera matrix, "
                f"got {camera.shape}"
            )

        K = camera.copy()

        # 1920x1080 -> 1440x1080 crop.
        K[0, 2] -= CROP_X

        # 1440x1080 -> 640x480 resize.
        sx = (
            MODEL_WIDTH
            / CROP_WIDTH
        )
        sy = (
            MODEL_HEIGHT
            / CROP_HEIGHT
        )

        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

        return K

    def _image(
        self,
        bgr: np.ndarray,
        depth: np.ndarray,
    ) -> tuple[
        torch.Tensor,
        int,
        int,
        np.ndarray,
    ]:
        if (
            bgr is None
            or depth is None
            or bgr.ndim != 3
            or bgr.shape[2] != 3
            or depth.ndim != 2
            or depth.shape != bgr.shape[:2]
        ):
            raise ValueError(
                "RACE requires aligned "
                "BGR HxWx3 and depth HxW; "
                f"got "
                f"{getattr(bgr, 'shape', None)}, "
                f"{getattr(depth, 'shape', None)}"
            )

        height, width = bgr.shape[:2]

        if (
            width != 1920
            or height != 1080
        ):
            raise ValueError(
                "RACE input must be 1920x1080, "
                f"got {width}x{height}"
            )

        rgb = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2RGB,
        )

        rgb = self._crop_and_resize(
            rgb,
            cv2.INTER_LINEAR,
        )

        in_h, in_w = (
            self.cfg.yaml_cfg.get(
                "eval_spatial_size",
                [
                    MODEL_HEIGHT,
                    MODEL_WIDTH,
                ],
            )
        )

        in_h = int(in_h)
        in_w = int(in_w)

        if (
            in_w != MODEL_WIDTH
            or in_h != MODEL_HEIGHT
        ):
            rgb = cv2.resize(
                rgb,
                (in_w, in_h),
                interpolation=cv2.INTER_LINEAR,
            )

        rgb_tensor = (
            torch.from_numpy(
                np.ascontiguousarray(
                    rgb
                )
            )
            .permute(2, 0, 1)
            .float()
            .div_(255)
        )

        channels = (
            self.model
            .backbone
            .conv1[0]
            .conv
            .in_channels
        )

        if channels == 4:
            depth = self._crop_and_resize(
                depth,
                cv2.INTER_NEAREST,
            )

            if (
                in_w != MODEL_WIDTH
                or in_h != MODEL_HEIGHT
            ):
                depth = cv2.resize(
                    depth,
                    (in_w, in_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            d = depth.astype(
                np.float32
            )

            d[
                d == self.invalid_depth_value
            ] = 0

            cap = float(
                self.depth_z_max_mm
                or self.cfg.yaml_cfg.get(
                    "val_dataloader",
                    {},
                )
                .get("dataset", {})
                .get(
                    "depth_z_max_mm",
                    2000.0,
                )
            )

            if cap <= 0:
                raise ValueError(
                    "depth_z_max_mm must be positive"
                )

            depth_tensor = (
                torch.from_numpy(
                    np.clip(
                        d,
                        0,
                        cap,
                    ) / cap
                )
                .unsqueeze(0)
            )

            image = torch.cat(
                (
                    rgb_tensor,
                    depth_tensor,
                ),
                dim=0,
            )

        elif channels == 3:
            image = rgb_tensor

        else:
            raise RuntimeError(
                "Unsupported RACE backbone "
                f"input channels: {channels}"
            )

        return (
            image.unsqueeze(0).to(
                self.device
            ),
            in_h,
            in_w,
            rgb,
        )

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: Sequence[
            Sequence[float]
        ],
    ) -> Dict[str, Any]:
        original_camera = np.asarray(
            intrinsic,
            dtype=np.float32,
        )

        if original_camera.shape != (
            3,
            3,
        ):
            raise ValueError(
                "Expected 3x3 camera matrix, "
                f"got {original_camera.shape}"
            )

        camera = self._update_intrinsic(
            original_camera
        )

        (
            image,
            height,
            width,
            debug_image,
        ) = self._image(
            rgb,
            depth,
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
                "debug_image": debug_image,
            }

        ordered = keep[
            torch.argsort(
                scores[keep],
                descending=True,
            )
        ]

        selected = []
        counts = {}

        for index in ordered.tolist():
            label = int(
                labels[index]
            )

            if (
                self.class_id is not None
                and label != self.class_id
            ):
                continue

            if (
                counts.get(label, 0)
                >= self.max_per_class
            ):
                continue

            selected.append(index)

            counts[label] = (
                counts.get(label, 0)
                + 1
            )

            if (
                self.max_detections
                is not None
                and len(selected)
                >= self.max_detections
            ):
                break

        if not selected:
            return {
                "success": False,
                "detections": [],
                "debug_image": debug_image,
            }

        detections = []

        for index in selected:
            box = result["boxes"][index]

            normalized = box.clone()

            normalized[0::2] /= width
            normalized[1::2] /= height

            normalized = torch.stack(
                (
                    (
                        normalized[0]
                        + normalized[2]
                    ) / 2,
                    (
                        normalized[1]
                        + normalized[3]
                    ) / 2,
                    normalized[2]
                    - normalized[0],
                    normalized[3]
                    - normalized[1],
                )
            ).unsqueeze(0)

            translation_mm = (
                self.criterion._c2t_pred(
                    result["translations"][
                        index
                    ].unsqueeze(0),
                    torch.from_numpy(
                        camera
                    ).to(self.device),
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

            quat = _matrix_to_quat_wxyz(
                rotation
            )

            score = float(
                result["scores"][index]
                .detach()
                .cpu()
            )

            label = int(
                result["labels"][index]
            )

            bbox_xyxy = (
                box.detach()
                .cpu()
                .numpy()
                .tolist()
            )

            x0, y0, x1, y1 = (
                np.rint(
                    np.asarray(
                        bbox_xyxy
                    )
                ).astype(int)
            )

            cv2.rectangle(
                debug_image,
                (x0, y0),
                (x1, y1),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                debug_image,
                (
                    f"RACE "
                    f"label={label} "
                    f"score={score:.3f}"
                ),
                (
                    x0,
                    max(
                        18,
                        y0 - 8,
                    ),
                ),
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
                    "translation_mm": (
                        translation_mm
                    ).tolist(),
                    "quat": quat.tolist(),
                    "rotation": (
                        rotation.tolist()
                    ),
                    "bbox_xyxy": bbox_xyxy,
                }
            )

        return {
            "success": (
                len(detections) > 0
            ),
            "detections": detections,
            "num_detections": len(
                detections
            ),
            "class_counts": {
                str(label): count
                for label, count
                in counts.items()
            },
            "debug_image": debug_image,
        }