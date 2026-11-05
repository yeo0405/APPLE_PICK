"""Two-stage RGB pose refinement for RACE-6D.

Stage 1: VertexDetector detects image vertices and edge lines.
Stage 2: PoseRefiner optimizes pose using the detection result.

If either stage fails, keep the original RACE-6D pose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import numpy as np

try:
    from .vertex_detector import VertexDetector
    from .pose_refine import PoseRefiner
except ImportError:
    from vertex_detector import VertexDetector
    from pose_refine import PoseRefiner

logger = logging.getLogger(__name__)


@dataclass
class ManagedResult:
    ok: bool
    R: np.ndarray
    t_mm: np.ndarray
    info: Dict[str, Any]
    method: str = "none"
    vertex: Optional[Any] = None
    pose: Optional[Any] = None


class RefineManager:
    def __init__(self, vertex_detector, pose_refiner):
        self.vertex_detector = vertex_detector
        self.pose_refiner = pose_refiner

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["RefineManager"]:
        if not cfg or not cfg.get("ENABLED", False):
            return None

        vertex_detector = VertexDetector.from_config(cfg)
        pose_refiner = PoseRefiner.from_config(cfg)

        if vertex_detector is None or pose_refiner is None:
            return None

        logger.info(
            "RGB_REFINE: two_stage=True, vertex_detector=True, pose_refiner=True"
        )

        return cls(vertex_detector, pose_refiner)

    @property
    def boxes(self) -> Set[int]:
        if self.vertex_detector is None:
            return set()

        return {int(k) for k in self.vertex_detector.boxes}

    def refine(self, bgr, camera, label, R, t_mm) -> ManagedResult:
        label = int(label)

        R0 = np.asarray(R, dtype=np.float64)
        t0 = np.asarray(t_mm, dtype=np.float64)

        if (
            self.vertex_detector is None
            or self.pose_refiner is None
            or label not in self.boxes
        ):
            return ManagedResult(
                False,
                R0,
                t0,
                {
                    "ok": False,
                    "reason": "refiner_unavailable",
                    "method": "none",
                },
            )

        info: Dict[str, Any] = {
            "ok": False,
            "method": "none",
        }

        vertex_res = None
        pose_res = None

        try:
            # Stage 1: Detect image vertices and edge lines.
            vertex_res = self.vertex_detector.detect(
                bgr=bgr,
                camera=camera,
                label=label,
                R=R0,
                t_mm=t0,
            )

            vertex_info = dict(vertex_res.info)

            info["vertex"] = vertex_info

            if not vertex_res.ok:
                info.update(
                    ok=False,
                    reason="vertex_detection_failed",
                )

                return ManagedResult(
                    False,
                    R0,
                    t0,
                    info,
                    "none",
                    vertex=vertex_res,
                )

            # Stage 2: Optimize pose using detected vertices and lines.
            pose_res = self.pose_refiner.refine(
                bgr=bgr,
                camera=camera,
                label=label,
                R=R0,
                t_mm=t0,
                detection=vertex_res,
            )

            pose_info = dict(pose_res.info)

            info["pose"] = pose_info

            if not pose_res.ok:
                info.update(
                    ok=False,
                    reason="pose_refinement_failed",
                )

                return ManagedResult(
                    False,
                    R0,
                    t0,
                    info,
                    "none",
                    vertex=vertex_res,
                    pose=pose_res,
                )

            info.update(
                ok=True,
                reason="ok",
                method="vertex_pose",
            )

            return ManagedResult(
                True,
                pose_res.R,
                pose_res.t_mm,
                info,
                "vertex_pose",
                vertex_res,
                pose_res,
            )

        except Exception as exc:
            logger.exception("RGB pose refinement failed: %s", exc)

            info.update(
                ok=False,
                reason=f"exception: {exc}",
            )

            return ManagedResult(
                False,
                R0,
                t0,
                info,
                "none",
                vertex_res,
                pose_res,
            )

    def draw(self, debug: np.ndarray, res: ManagedResult) -> None:
        if res.vertex is not None and self.vertex_detector is not None:
            self.vertex_detector.draw(debug, res.vertex)

        if res.pose is not None and self.pose_refiner is not None:
            self.pose_refiner.draw(debug, res.pose)
