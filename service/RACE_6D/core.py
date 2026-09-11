"""Reusable in-process RACE-6D pose estimator for ROS."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import cv2
import numpy as np
import torch

RACE_ROOT = Path(__file__).resolve().parent

# src is bundled in this RACE_6D directory.  Add its parent directory so
# import src.zoo works regardless of the directory used to start ros_app.py.
if str(RACE_ROOT) not in sys.path:
    sys.path.insert(0, str(RACE_ROOT))

# RACE-6D repository dependencies; intentionally validated at service startup.
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
        raise RuntimeError(f"RACE checkpoint/model mismatch: missing={len(missing)}, unexpected={len(unexpected)}")


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to ROS-compatible [qw, qx, qy, qz]."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError(f"Expected 3x3 rotation matrix, got {m.shape}")
    trace = float(np.trace(m))
    if trace > 0:
        s = 2 * np.sqrt(trace + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = 2 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = 2 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.asarray(q, dtype=np.float32)
    norm = np.linalg.norm(q)
    if norm < 1e-8:
        raise RuntimeError("RACE-6D produced an invalid rotation")
    return q / norm


class PoseEstimator:
    """RACE-6D model that takes BGR/depth camera arrays and returns a pose."""

    def __init__(self, model_path: str, config_path: str, device: str = "", score_threshold: float = 0.25, max_per_class: int = 1, max_detections: Optional[int] = None, depth_z_max_mm: Optional[float] = None, invalid_depth_value: int = 65535, class_id: Optional[int] = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.score_threshold, self.max_per_class = float(score_threshold), int(max_per_class)
        self.max_detections, self.depth_z_max_mm = max_detections, depth_z_max_mm
        self.invalid_depth_value, self.class_id = int(invalid_depth_value), class_id
        if not 0 <= self.score_threshold <= 1 or self.max_per_class < 1:
            raise ValueError("score_threshold must be 0..1 and max_per_class must be >= 1")
        self.cfg = YAMLConfig(config_path)
        for name in ("PResNet", "PResNet_depth"):
            if name in self.cfg.yaml_cfg:
                self.cfg.yaml_cfg[name]["pretrained"] = False
        self.model = self.cfg.model.to(self.device).eval()
        self.postprocessor = self.cfg.postprocessor.to(self.device).eval()
        self.criterion = self.cfg.criterion.to(self.device).eval()
        self.criterion.set_pose_source(self.model.decoder)
        _load_checkpoint(self.model, model_path)

    def _image(self, bgr: np.ndarray, depth: np.ndarray) -> tuple[torch.Tensor, int, int]:
        if bgr is None or depth is None or bgr.ndim != 3 or bgr.shape[2] != 3 or depth.ndim != 2 or depth.shape != bgr.shape[:2]:
            raise ValueError(f"RACE requires aligned BGR HxWx3 and depth HxW; got {getattr(bgr, 'shape', None)}, {getattr(depth, 'shape', None)}")
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        in_h, in_w = self.cfg.yaml_cfg.get("eval_spatial_size", [height, width])
        model_rgb = cv2.resize(rgb, (int(in_w), int(in_h)), interpolation=cv2.INTER_LINEAR)
        rgb_tensor = torch.from_numpy(np.ascontiguousarray(model_rgb)).permute(2, 0, 1).float().div_(255)
        channels = self.model.backbone.conv1[0].conv.in_channels
        if channels == 4:
            d = cv2.resize(depth, (int(in_w), int(in_h)), interpolation=cv2.INTER_NEAREST).astype(np.float32)
            d[d == self.invalid_depth_value] = 0
            cap = float(self.depth_z_max_mm or self.cfg.yaml_cfg.get("val_dataloader", {}).get("dataset", {}).get("depth_z_max_mm", 2000.0))
            if cap <= 0:
                raise ValueError("depth_z_max_mm must be positive")
            image = torch.cat((rgb_tensor, torch.from_numpy(np.clip(d, 0, cap) / cap).unsqueeze(0)), dim=0)
        elif channels == 3:
            image = rgb_tensor
        else:
            raise RuntimeError(f"Unsupported RACE backbone input channels: {channels}")
        return image.unsqueeze(0).to(self.device), height, width

    def predict(self, rgb: np.ndarray, depth: np.ndarray, intrinsic: Sequence[Sequence[float]]) -> Dict[str, Any]:
        camera = np.asarray(intrinsic, dtype=np.float32)
        if camera.shape != (3, 3):
            raise ValueError(f"Expected 3x3 camera matrix, got {camera.shape}")
        image, height, width = self._image(rgb, depth)
        size = torch.tensor([[width, height]], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            result = self.postprocessor(self.model(image), size)[0]
        keep = torch.nonzero(result["scores"] >= self.score_threshold, as_tuple=False).squeeze(1)
        ordered = keep[torch.argsort(result["scores"][keep], descending=True)] if len(keep) else keep
        selected, counts = [], {}
        for index in ordered.tolist():
            label = int(result["labels"][index])
            if (self.class_id is not None and label != self.class_id) or counts.get(label, 0) >= self.max_per_class:
                continue
            selected.append(index); counts[label] = counts.get(label, 0) + 1
            if self.max_detections is not None and len(selected) >= self.max_detections:
                break
        if not selected:
            return {"success": False, "translation": None, "quat": None, "confidence": None, "debug_image": rgb.copy()}
        index = selected[0]
        box = result["boxes"][index]
        normalized = box.clone(); normalized[0::2] /= width; normalized[1::2] /= height
        normalized = torch.stack(((normalized[0] + normalized[2]) / 2, (normalized[1] + normalized[3]) / 2, normalized[2] - normalized[0], normalized[3] - normalized[1])).unsqueeze(0)
        translation_mm = self.criterion._c2t_pred(result["translations"][index].unsqueeze(0), torch.from_numpy(camera).to(self.device), normalized, width, height)[0].detach().cpu().numpy()
        rotation = result["rotations"][index].detach().cpu().numpy()
        score = float(result["scores"][index].detach().cpu())
        debug = rgb.copy(); x0, y0, x1, y1 = np.rint(box.detach().cpu().numpy()).astype(int)
        cv2.rectangle(debug, (x0, y0), (x1, y1), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(debug, f"RACE label={int(result['labels'][index])} score={score:.3f}", (x0, max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1, cv2.LINE_AA)
        return {"success": True, "label": int(result["labels"][index]), "confidence": score, "translation": (translation_mm / 1000.0).tolist(), "translation_mm": translation_mm.tolist(), "quat": _matrix_to_quat_wxyz(rotation).tolist(), "rotation": rotation.tolist(), "bbox_xyxy": box.detach().cpu().tolist(), "debug_image": debug}
