#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .geometry import calculate_pose


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "model"

SAM2_CONFIG = MODEL_DIR / "sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT = MODEL_DIR / "sam2.1_hiera_small.pt"

DINOV3_ROOT = PROJECT_ROOT / "service" / "dinov3"
DINOV3_WEIGHTS = (
    MODEL_DIR / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
)

MIN_MASK_AREA = 100
AXIS_LENGTH = 0.05

DINO_IMAGE_SIZE = 224
DINO_THRESHOLD = 0.70


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("Invalid zero quaternion")
    q /= norm
    if q[3] < 0:
        q = -q
    return q


def quaternion_inverse(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    return np.array(
        [-q[0], -q[1], -q[2], q[3]],
        dtype=np.float64,
    )


def quaternion_multiply(
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ], dtype=np.float64)


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternion(q)
    x, y, z, w = q

    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)

    sinp = 2.0 * (w * y - z * x)
    pitch = (
        np.copysign(np.pi / 2.0, sinp)
        if abs(sinp) >= 1.0
        else np.arcsin(sinp)
    )

    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)

    return np.degrees(np.array([roll, pitch, yaw]))


class AlignmentDINO:
    def __init__(
        self,
        gt_image: np.ndarray,
        gt_mask: np.ndarray,
        device: str,
    ):
        self.device = torch.device(device)
        self.transform = transforms.Compose([
            transforms.Resize(
                (DINO_IMAGE_SIZE, DINO_IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ])

        self.model = self._load_model()
        self.gt_feature = self._extract_mask_feature(
            gt_image,
            gt_mask,
        )

    def _load_model(self):
        if not DINOV3_ROOT.is_dir():
            raise FileNotFoundError(
                f"DINOv3 root not found: {DINOV3_ROOT}"
            )

        if not DINOV3_WEIGHTS.is_file():
            raise FileNotFoundError(
                f"DINOv3 weights not found: {DINOV3_WEIGHTS}"
            )

        model = torch.hub.load(
            str(DINOV3_ROOT),
            "dinov3_vits16",
            source="local",
            weights=str(DINOV3_WEIGHTS),
        )

        model = model.to(self.device)
        model.eval()
        return model

    @staticmethod
    def _crop_mask(
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            raise RuntimeError("DINO mask is empty")

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max()) + 1
        y2 = int(ys.max()) + 1

        return (
            image[y1:y2, x1:x2],
            mask[y1:y2, x1:x2],
        )

    @torch.inference_mode()
    def _extract_tokens(
        self,
        image: np.ndarray,
    ) -> Tuple[torch.Tensor, int, int]:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        tensor = self.transform(
            Image.fromarray(rgb)
        ).unsqueeze(0).to(self.device)

        if hasattr(
            self.model,
            "get_intermediate_layers",
        ):
            try:
                output = self.model.get_intermediate_layers(
                    tensor,
                    n=1,
                    reshape=False,
                    return_class_token=True,
                )
            except TypeError:
                output = self.model.get_intermediate_layers(
                    tensor,
                    n=1,
                    reshape=False,
                )

            if isinstance(output, (list, tuple)):
                output = output[-1]

            if isinstance(output, tuple):
                tokens = output[0]
            else:
                tokens = output
        else:
            output = self.model.forward_features(tensor)

            if not isinstance(output, dict):
                raise RuntimeError(
                    "DINOv3 forward_features returned "
                    "unsupported output"
                )

            tokens = output.get(
                "x_norm_patchtokens"
            )

            if tokens is None:
                tokens = output.get(
                    "patch_tokens"
                )

            if tokens is None:
                raise RuntimeError(
                    "DINOv3 patch tokens not found"
                )

        if tokens.ndim == 4:
            _, _, gh, gw = tokens.shape
            tokens = tokens.flatten(2).transpose(1, 2)
            return tokens[0], gh, gw

        if tokens.ndim != 3:
            raise RuntimeError(
                f"Unexpected DINOv3 feature shape: "
                f"{tuple(tokens.shape)}"
            )

        tokens = tokens[0]
        count = tokens.shape[0]
        grid = int(round(np.sqrt(count)))

        if grid * grid == count:
            return tokens, grid, grid

        if grid * grid == count - 1:
            return tokens[1:], grid, grid

        raise RuntimeError(
            f"Cannot determine DINO patch grid from "
            f"{count} tokens"
        )

    @torch.inference_mode()
    def _extract_mask_feature(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> torch.Tensor:
        crop, mask_crop = self._crop_mask(
            image,
            mask,
        )

        if crop.size == 0:
            raise RuntimeError(
                "DINO object crop is empty"
            )

        tokens, gh, gw = self._extract_tokens(
            crop
        )

        mask_small = cv2.resize(
            mask_crop,
            (gw, gh),
            interpolation=cv2.INTER_AREA,
        ) > 127

        selected = torch.from_numpy(
            mask_small.reshape(-1)
        ).to(self.device)

        if selected.sum() == 0:
            raise RuntimeError(
                "DINO object mask contains no patches"
            )

        feature = tokens[selected].mean(
            dim=0
        )

        return F.normalize(
            feature,
            dim=0,
        )

    @torch.inference_mode()
    def evaluate(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        feature = self._extract_mask_feature(
            image,
            mask,
        )

        return float(
            torch.dot(
                self.gt_feature,
                feature,
            ).item()
        )


class AlignmentEstimator:
    def __init__(
        self,
        gt_json: Path,
        device: Optional[str] = None,
    ):
        self.gt_json = Path(gt_json)
        self.gt_data = self._load_json(
            self.gt_json
        )
        self.gt_image_path = (
            self._resolve_gt_image()
        )
        self.gt_mask = self._load_gt_mask()
        self.gt_depth = self._load_gt_depth()

        self.gt_image = cv2.imread(
            str(self.gt_image_path),
            cv2.IMREAD_COLOR,
        )

        if self.gt_image is None:
            raise RuntimeError(
                f"Failed to load GT image: "
                f"{self.gt_image_path}"
            )

        self.gt_pose = self._load_gt_pose()
        self.gt_obb = self._load_gt_obb()

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.predictor = self._load_sam2()

        self.dino = AlignmentDINO(
            gt_image=self.gt_image,
            gt_mask=self.gt_mask,
            device=self.device,
        )

        self.gt_height, self.gt_width = (
            self.gt_image.shape[:2]
        )

    @staticmethod
    def _load_json(
        path: Path,
    ) -> Dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return json.load(f)

    def _resolve_gt_image(self) -> Path:
        for ext in (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
        ):
            path = (
                self.gt_json.parent
                / f"GT{ext}"
            )

            if path.is_file():
                return path

        raise FileNotFoundError(
            f"GT image not found in "
            f"{self.gt_json.parent}"
        )

    def _load_gt_mask(self) -> np.ndarray:
        objects = self.gt_data.get(
            "objects",
            [],
        )

        if not objects:
            raise RuntimeError(
                "gt.json contains no objects"
            )

        mask_file = objects[0][
            "mask"
        ]["file"]

        mask_path = (
            self.gt_json.parent
            / mask_file
        )

        mask = cv2.imread(
            str(mask_path),
            cv2.IMREAD_GRAYSCALE,
        )

        if mask is None:
            raise RuntimeError(
                f"Failed to load GT mask: "
                f"{mask_path}"
            )

        return mask

    def _load_gt_depth(self) -> np.ndarray:
        path = (
            self.gt_json.parent
            / "depth.png"
        )

        depth = cv2.imread(
            str(path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            raise RuntimeError(
                f"Failed to load GT depth: "
                f"{path}"
            )

        if depth.ndim != 2:
            raise RuntimeError(
                f"GT depth must be single-channel: "
                f"{depth.shape}"
            )

        return depth

    def _load_gt_pose(
        self,
    ) -> Dict[str, Any]:
        obj = self.gt_data[
            "objects"
        ][0]

        if "xyz" not in obj:
            raise RuntimeError(
                "GT object contains no xyz"
            )

        if "quaternion" not in obj:
            raise RuntimeError(
                "GT object contains no quaternion"
            )

        return {
            "location": np.asarray(
                obj["xyz"],
                dtype=np.float64,
            ),
            "rotation": normalize_quaternion(
                np.asarray(
                    obj["quaternion"],
                    dtype=np.float64,
                )
            ),
        }

    def _load_gt_obb(
        self,
    ) -> Optional[Dict[str, Any]]:
        return self.gt_data[
            "objects"
        ][0].get("obb")

    def _load_sam2(self):
        if not SAM2_CONFIG.is_file():
            raise FileNotFoundError(
                f"SAM2 config not found: "
                f"{SAM2_CONFIG}"
            )

        if not SAM2_CHECKPOINT.is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: "
                f"{SAM2_CHECKPOINT}"
            )

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(
            version_base=None,
            config_dir=str(
                SAM2_CONFIG.parent
            ),
        ):
            cfg = compose(
                config_name=SAM2_CONFIG.stem
            )

        model = instantiate(cfg.model)

        checkpoint = torch.load(
            SAM2_CHECKPOINT,
            map_location="cpu",
        )

        if "model" in checkpoint:
            checkpoint = checkpoint["model"]

        missing, unexpected = (
            model.load_state_dict(
                checkpoint,
                strict=False,
            )
        )

        if missing:
            print(
                f"SAM2 missing keys: "
                f"{len(missing)}"
            )

        if unexpected:
            print(
                f"SAM2 unexpected keys: "
                f"{len(unexpected)}"
            )

        model.to(self.device)
        model.eval()

        return SAM2ImagePredictor(model)

    @staticmethod
    def _fit_obb(
        mask: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        if mask is None:
            return None

        binary = (
            mask > 0
        ).astype(np.uint8)

        ys, xs = np.where(
            binary > 0
        )

        if len(xs) < 3:
            return None

        points = np.column_stack(
            (xs, ys)
        ).astype(np.float32)

        rect = cv2.minAreaRect(
            points
        )

        (cx, cy), (
            width,
            height,
        ), angle = rect

        if width < height:
            width, height = (
                height,
                width,
            )
            angle += 90.0

        while angle >= 90.0:
            angle -= 180.0

        while angle < -90.0:
            angle += 180.0

        return {
            "center": (
                float(cx),
                float(cy),
            ),
            "width": float(width),
            "height": float(height),
            "angle_deg": float(angle),
        }

    def _get_gt_bbox(
        self,
    ) -> np.ndarray:
        if self.gt_obb is not None:
            cx, cy = (
                self.gt_obb["center"]
            )

            width = float(
                self.gt_obb["width"]
            )

            height = float(
                self.gt_obb["height"]
            )

            angle = float(
                self.gt_obb["angle_deg"]
            )

            box = cv2.boxPoints(
                (
                    (
                        float(cx),
                        float(cy),
                    ),
                    (
                        width,
                        height,
                    ),
                    angle,
                )
            )

            return box.astype(
                np.float32
            )

        ys, xs = np.where(
            self.gt_mask > 0
        )

        if len(xs) == 0:
            raise RuntimeError(
                "GT mask contains no pixels"
            )

        return np.array(
            [
                [xs.min(), ys.min()],
                [xs.max(), ys.min()],
                [xs.max(), ys.max()],
                [xs.min(), ys.max()],
            ],
            dtype=np.float32,
        )

    def _predict_mask(
        self,
        image: np.ndarray,
    ) -> Tuple[
        np.ndarray,
        float,
        np.ndarray,
        float,
        bool,
    ]:
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        self.predictor.set_image(
            rgb
        )

        bbox = self._get_gt_bbox()

        box = np.array(
            [
                bbox[:, 0].min(),
                bbox[:, 1].min(),
                bbox[:, 0].max(),
                bbox[:, 1].max(),
            ],
            dtype=np.float32,
        )

        masks, scores, _ = (
            self.predictor.predict(
                box=box,
                multimask_output=True,
            )
        )

        if masks is None or len(masks) == 0:
            raise RuntimeError(
                "SAM2 returned no masks"
            )

        scores = np.asarray(
            scores,
            dtype=np.float64,
        ).reshape(-1)

        candidates = []

        for i, candidate in enumerate(
            masks
        ):
            area = int(
                np.count_nonzero(
                    candidate
                )
            )

            if area >= MIN_MASK_AREA:
                candidates.append(
                    (
                        float(scores[i]),
                        area,
                        candidate,
                    )
                )

        if not candidates:
            raise RuntimeError(
                "SAM2 returned no valid mask"
            )

        candidates.sort(
            key=lambda x: x[0],
            reverse=True,
        )

        sam_score, _, mask = (
            candidates[0]
        )

        mask = (
            mask.astype(np.uint8)
            * 255
        )

        dino_score = self.dino.evaluate(
            image,
            mask,
        )

        accepted = (
            dino_score
            >= DINO_THRESHOLD
        )

        return (
            mask,
            sam_score,
            box,
            dino_score,
            accepted,
        )

    @staticmethod
    def _calculate_pose(
        mask: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> Optional[
        Dict[str, Any]
    ]:
        K = np.asarray(
            K,
            dtype=np.float64,
        )

        if K.shape != (3, 3):
            raise ValueError(
                f"K must have shape "
                f"(3, 3), got {K.shape}"
            )

        pose = calculate_pose(
            mask,
            depth,
            K,
        )

        if pose is None:
            return None

        plane = pose.get(
            "plane"
        )

        if plane is not None:
            pose["debug_masks"] = {
                "obb_mask": plane.get(
                    "obb_mask"
                ),
                "plane_mask": plane.get(
                    "plane_mask"
                ),
            }

        pose["geometry_debug"] = (
            pose.get(
                "orientation_debug"
            )
        )

        return pose

    def _calculate_delta(
        self,
        input_pose: Dict[str, Any],
    ) -> Dict[str, Any]:
        gt_location = np.asarray(
            self.gt_pose["location"],
            dtype=np.float64,
        )

        input_location = np.asarray(
            input_pose["location"],
            dtype=np.float64,
        )

        gt_rotation = (
            normalize_quaternion(
                self.gt_pose["rotation"]
            )
        )

        input_rotation = (
            normalize_quaternion(
                input_pose["rotation"]
            )
        )

        delta_location = (
            gt_location
            - input_location
        )

        delta_rotation = (
            quaternion_multiply(
                gt_rotation,
                quaternion_inverse(
                    input_rotation
                ),
            )
        )

        delta_rotation = (
            normalize_quaternion(
                delta_rotation
            )
        )

        delta_euler = (
            quaternion_to_euler(
                delta_rotation
            )
        )

        delta_angle = (
            2.0
            * np.arccos(
                np.clip(
                    abs(
                        delta_rotation[3]
                    ),
                    -1.0,
                    1.0,
                )
            )
        )

        return {
            "location": delta_location,
            "rotation": delta_rotation,
            "rotation_euler_deg": delta_euler,
            "rotation_angle_deg": float(
                np.degrees(
                    delta_angle
                )
            ),
        }

    @staticmethod
    def _project_point(
        point: np.ndarray,
        K: np.ndarray,
    ) -> Optional[
        Tuple[int, int]
    ]:
        point = np.asarray(
            point,
            dtype=np.float64,
        ).reshape(3)

        if point[2] <= 1e-8:
            return None

        x = (
            K[0, 0]
            * point[0]
            / point[2]
            + K[0, 2]
        )

        y = (
            K[1, 1]
            * point[1]
            / point[2]
            + K[1, 2]
        )

        if not np.isfinite(
            [x, y]
        ).all():
            return None

        return (
            int(round(x)),
            int(round(y)),
        )

    @classmethod
    def _draw_axis(
        cls,
        image: np.ndarray,
        origin_3d: np.ndarray,
        axis_3d: np.ndarray,
        K: np.ndarray,
        color: Tuple[int, int, int],
        label: str,
    ) -> None:
        origin = cls._project_point(
            origin_3d,
            K,
        )

        endpoint_3d = (
            origin_3d
            + axis_3d
            * AXIS_LENGTH
        )

        endpoint = cls._project_point(
            endpoint_3d,
            K,
        )

        if (
            origin is None
            or endpoint is None
        ):
            return

        cv2.arrowedLine(
            image,
            origin,
            endpoint,
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.15,
        )

        cv2.putText(
            image,
            label,
            (
                endpoint[0] + 8,
                endpoint[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
            cv2.LINE_AA,
        )

    @staticmethod
    def _draw_text(
        image: np.ndarray,
        result: Dict[str, Any],
    ) -> None:
        lines = []

        dino_score = result.get(
            "dino_score"
        )

        if dino_score is not None:
            status = (
                "PASS"
                if result.get("dino_accepted")
                else "REJECT"
            )

            lines.append(
                f"DINO: {dino_score:.3f} "
                f"/ {DINO_THRESHOLD:.3f} "
                f"[{status}]"
            )

        sam_score = result.get(
            "sam_score"
        )

        if sam_score is not None:
            lines.append(
                f"SAM score: {sam_score:.3f}"
            )

        input_pose = result.get(
            "input_pose"
        )

        delta = result.get(
            "delta"
        )

        if input_pose is not None:
            location = np.asarray(
                input_pose["location"]
            ).reshape(3)

            rotation = np.asarray(
                input_pose["rotation"]
            ).reshape(4)

            lines.extend(
                [
                    "Input XYZ: "
                    f"{location[0]:.4f}, "
                    f"{location[1]:.4f}, "
                    f"{location[2]:.4f}",
                    "Input Q: "
                    f"{rotation[0]:.4f}, "
                    f"{rotation[1]:.4f}, "
                    f"{rotation[2]:.4f}, "
                    f"{rotation[3]:.4f}",
                ]
            )

        if delta is not None:
            location = np.asarray(
                delta["location"]
            ).reshape(3)

            rotation = np.asarray(
                delta["rotation"]
            ).reshape(4)

            euler = np.asarray(
                delta["rotation_euler_deg"]
            ).reshape(3)

            lines.extend(
                [
                    "Delta XYZ: "
                    f"{location[0]:.4f}, "
                    f"{location[1]:.4f}, "
                    f"{location[2]:.4f}",
                    "Delta Q: "
                    f"{rotation[0]:.4f}, "
                    f"{rotation[1]:.4f}, "
                    f"{rotation[2]:.4f}, "
                    f"{rotation[3]:.4f}",
                    "Delta RPY: "
                    f"{euler[0]:.2f}, "
                    f"{euler[1]:.2f}, "
                    f"{euler[2]:.2f}",
                    "Delta Angle: "
                    f"{delta['rotation_angle_deg']:.2f} deg",
                ]
            )

        x, y = 20, 30

        for line in lines:
            cv2.putText(
                image,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            y += 28

    def _draw_debug(
        self,
        image: np.ndarray,
        result: Dict[str, Any],
        K: np.ndarray,
    ) -> np.ndarray:
        debug = image.copy()

        # GT.json bbox = SAM2 工作范围
        bbox = result.get(
            "bbox"
        )

        if bbox is not None:
            x1, y1, x2, y2 = map(
                int,
                bbox,
            )

            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3,
            )

            cv2.putText(
                debug,
                "SAM ROI (GT bbox)",
                (
                    x1,
                    max(
                        25,
                        y1 - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # DINO 只判断，不修改 SAM2 mask
        mask = result.get(
            "mask"
        )

        if (
            mask is not None
            and result.get(
                "dino_accepted"
            )
        ):
            binary = mask > 0

            if binary.shape == (
                debug.shape[:2]
            ):
                overlay = debug.copy()

                overlay[binary] = (
                    0.5
                    * overlay[binary]
                    + 0.5
                    * np.array(
                        [255, 0, 0],
                        dtype=np.float64,
                    )
                ).astype(
                    np.uint8
                )

                debug = cv2.addWeighted(
                    debug,
                    0.7,
                    overlay,
                    0.3,
                    0.0,
                )

        input_obb = result.get(
            "input_obb"
        )

        if (
            input_obb is not None
            and result.get(
                "dino_accepted"
            )
        ):
            cx, cy = input_obb[
                "center"
            ]

            points = cv2.boxPoints(
                (
                    (
                        float(cx),
                        float(cy),
                    ),
                    (
                        float(
                            input_obb[
                                "width"
                            ]
                        ),
                        float(
                            input_obb[
                                "height"
                            ]
                        ),
                    ),
                    float(
                        input_obb[
                            "angle_deg"
                        ]
                    ),
                )
            )

            points = np.round(
                points
            ).astype(
                np.int32
            )

            cv2.polylines(
                debug,
                [points],
                True,
                (255, 255, 0),
                3,
                cv2.LINE_AA,
            )

            cv2.circle(
                debug,
                (
                    int(cx),
                    int(cy),
                ),
                5,
                (255, 255, 0),
                -1,
            )

        input_pose = result.get(
            "input_pose"
        )

        if (
            input_pose is not None
            and result.get(
                "dino_accepted"
            )
        ):
            location = np.asarray(
                input_pose[
                    "location"
                ],
                dtype=np.float64,
            ).reshape(3)

            orientation_debug = (
                input_pose.get(
                    "orientation_debug"
                )
            )

            if orientation_debug is not None:
                x_axis = np.asarray(
                    orientation_debug[
                        "x_axis"
                    ],
                    dtype=np.float64,
                ).reshape(3)

                y_axis = np.asarray(
                    orientation_debug[
                        "y_axis"
                    ],
                    dtype=np.float64,
                ).reshape(3)

                z_axis = np.asarray(
                    orientation_debug[
                        "z_axis"
                    ],
                    dtype=np.float64,
                ).reshape(3)

                origin = (
                    self._project_point(
                        location,
                        K,
                    )
                )

                if origin is not None:
                    cv2.circle(
                        debug,
                        origin,
                        7,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )

                self._draw_axis(
                    debug,
                    location,
                    x_axis,
                    K,
                    (0, 0, 255),
                    "X",
                )

                self._draw_axis(
                    debug,
                    location,
                    y_axis,
                    K,
                    (0, 255, 0),
                    "Y",
                )

                self._draw_axis(
                    debug,
                    location,
                    z_axis,
                    K,
                    (255, 0, 0),
                    "Z",
                )

        self._draw_text(
            debug,
            result,
        )

        return debug

    def predict(
        self,
        image: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> Dict[str, Any]:
        if image is None:
            raise ValueError(
                "Input image is None"
            )

        if depth is None:
            raise ValueError(
                "Input depth is None"
            )

        if K is None:
            raise ValueError(
                "Camera intrinsic matrix K is None"
            )

        K = np.asarray(
            K,
            dtype=np.float64,
        )

        if K.shape != (3, 3):
            raise ValueError(
                f"K must have shape "
                f"(3, 3), got {K.shape}"
            )

        if image.shape[:2] != depth.shape[:2]:
            raise ValueError(
                "Input RGB/depth resolution mismatch: "
                f"{image.shape[:2]} vs "
                f"{depth.shape[:2]}"
            )

        (
            mask,
            sam_score,
            bbox,
            dino_score,
            dino_accepted,
        ) = self._predict_mask(
            image
        )

        input_obb = (
            self._fit_obb(mask)
            if dino_accepted
            else None
        )

        input_pose = None
        delta = None

        if dino_accepted:
            input_pose = (
                self._calculate_pose(
                    mask,
                    depth,
                    K,
                )
            )

            if (
                input_pose is not None
                and input_pose.get(
                    "rotation"
                ) is not None
                and input_pose.get(
                    "location"
                ) is not None
            ):
                delta = (
                    self._calculate_delta(
                        input_pose
                    )
                )

        result = {
            "present": (
                dino_accepted
                and input_obb is not None
            ),
            "mask": mask,
            "bbox": bbox,
            "sam_score": sam_score,
            "dino_score": dino_score,
            "dino_threshold": DINO_THRESHOLD,
            "dino_accepted": dino_accepted,
            "gt_pose": self.gt_pose,
            "gt_obb": self.gt_obb,
            "input_obb": input_obb,
            "input_pose": input_pose,
            "delta": delta,
        }

        result["debug_image"] = (
            self._draw_debug(
                image,
                result,
                K,
            )
        )

        return result