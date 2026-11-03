"""Edge-only refiner for RACE-6D pose estimation.

If edge refinement succeeds, use the refined pose.
If it fails for any reason, keep the original RACE-6D pose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import numpy as np

try:
    from .pose_refine import PoseRefiner as EdgeRefiner
except ImportError:
    from pose_refine import PoseRefiner as EdgeRefiner

logger = logging.getLogger(__name__)


@dataclass
class ManagedResult:
    ok: bool
    R: np.ndarray
    t_mm: np.ndarray
    info: Dict[str, Any]
    method: str = "none"
    edge: Optional[Any] = None


class RefineManager:
    def __init__(self, edge):
        self.edge = edge

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["RefineManager"]:
        if not cfg or not cfg.get("ENABLED", False):
            return None

        edge = EdgeRefiner.from_config(cfg)

        if edge is None:
            return None

        logger.info("RGB_REFINE: edge_only=True")

        return cls(edge)

    @property
    def boxes(self) -> Set[int]:
        if self.edge is None:
            return set()

        return {int(k) for k in self.edge.boxes}

    def refine(self, bgr, camera, label, R, t_mm) -> ManagedResult:
        label = int(label)

        R0 = np.asarray(R, dtype=np.float64)
        t0 = np.asarray(t_mm, dtype=np.float64)

        if self.edge is None or label not in self.edge.boxes:
            return ManagedResult(
                False,
                R0,
                t0,
                {
                    "ok": False,
                    "reason": "edge_unavailable",
                    "method": "none",
                },
                "none",
            )

        try:
            edge_res = self.edge.refine(
                bgr,
                camera,
                label,
                R0,
                t0,
            )

            info = dict(edge_res.info)
            info["method"] = "edge"

            if edge_res.ok:
                return ManagedResult(
                    True,
                    edge_res.R,
                    edge_res.t_mm,
                    info,
                    "edge",
                    edge_res,
                )

            return ManagedResult(
                False,
                R0,
                t0,
                info,
                "none",
                edge_res,
            )

        except Exception as exc:
            logger.exception("Edge refinement failed: %s", exc)

            return ManagedResult(
                False,
                R0,
                t0,
                {
                    "ok": False,
                    "reason": f"exception: {exc}",
                    "method": "none",
                },
                "none",
            )

    def draw(self, debug: np.ndarray, res: ManagedResult) -> None:
        if res.edge is not None and self.edge is not None:
            self.edge.draw(debug, res.edge)
