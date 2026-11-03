"""Edge refiner first; if it fails for ANY reason, the texture refiner is tried.

The returned object has the interface core.py already uses:
  .ok, .R, .t_mm, .info, .method     and   RefineManager.draw(debug, res)

info (JSON-safe) keys added here:
  method        "edge" | "texture" | "none"
  fallback      True when the texture refiner was tried because the edge refiner failed
  edge_reason   why the edge refiner failed (only when fallback is True)
  edge          nested info of the failed edge attempt (only when fallback is True)
When the fallback ran, the top-level info is the texture refiner's info.
If both fail, reason = "<edge_reason>+tex:<texture_reason>" and the RACE pose is kept.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

import numpy as np

try:
    from .pose_refine import PoseRefiner as EdgeRefiner
    from .texture_refine import TexturePoseRefiner
except ImportError:
    from pose_refine import PoseRefiner as EdgeRefiner
    from texture_refine import TexturePoseRefiner

logger = logging.getLogger(__name__)


@dataclass
class ManagedResult:
    ok: bool
    R: np.ndarray
    t_mm: np.ndarray
    info: Dict[str, Any]
    method: str = "none"
    edge: Optional[Any] = None  # pose_refine.RefineResult
    texture: Optional[Any] = None  # texture_refine.TextureResult


class RefineManager:
    def __init__(self, edge, texture):
        self.edge = edge
        self.texture = texture

    @classmethod
    def from_config(cls, cfg: Optional[dict]) -> Optional["RefineManager"]:
        if not cfg or not cfg.get("ENABLED", False):
            return None

        edge = EdgeRefiner.from_config(cfg)
        texture = TexturePoseRefiner.from_config(cfg)  # None unless TEXTURE.ENABLED

        if edge is None and texture is None:
            return None

        logger.info(
            "RGB_REFINE: edge=%s texture_fallback=%s",
            edge is not None,
            texture is not None,
        )
        return cls(edge, texture)

    # ------------------------------------------------------------------
    @property
    def boxes(self) -> Set[int]:
        """Labels refinable by at least one method (core.py: `label in boxes`)."""
        out: Set[int] = set()
        if self.edge is not None:
            out |= {int(k) for k in self.edge.boxes}
        if self.texture is not None:
            out |= {int(k) for k in self.texture.boxes}
        return out

    @property
    def last_match_image(self):
        return None if self.texture is None else self.texture.last_match_image

    @last_match_image.setter
    def last_match_image(self, value):
        if self.texture is not None:
            self.texture.last_match_image = value

    # ------------------------------------------------------------------
    def refine(self, bgr, camera, label, R, t_mm) -> ManagedResult:
        label = int(label)
        R0 = np.asarray(R, dtype=np.float64)
        t0 = np.asarray(t_mm, dtype=np.float64)

        # 1) edge-based refine
        edge_res = None
        if self.edge is not None and label in self.edge.boxes:
            edge_res = self.edge.refine(bgr, camera, label, R0, t0)
            if edge_res.ok:
                info = dict(edge_res.info)
                info["method"] = "edge"
                return ManagedResult(
                    True, edge_res.R, edge_res.t_mm, info, "edge", edge_res
                )

        edge_reason = (
            str(edge_res.info.get("reason"))
            if edge_res is not None
            else "edge_unavailable"
        )

        # 2) no texture fallback for this label
        if self.texture is None or label not in self.texture.boxes:
            if edge_res is not None:
                info = dict(edge_res.info)
                info["method"] = "none"
            else:
                info = {"ok": False, "reason": "no_method_for_label", "method": "none"}
            return ManagedResult(False, R0, t0, info, "none", edge_res)

        # 3) texture fallback (any edge failure)
        tex_res = self.texture.refine(bgr, camera, label, R0, t0)

        if tex_res.ok:
            info = dict(tex_res.info)
            info.update(method="texture", fallback=True, edge_reason=edge_reason)
            if edge_res is not None:
                info["edge"] = edge_res.info
            return ManagedResult(
                True, tex_res.R, tex_res.t_mm, info, "texture", edge_res, tex_res
            )

        tex_reason = str(tex_res.info.get("reason"))
        info = dict(tex_res.info)
        info.update(
            ok=False,
            method="none",
            fallback=True,
            edge_reason=edge_reason,
            texture_reason=tex_reason,
            reason=f"{edge_reason}+tex:{tex_reason}",
        )
        if edge_res is not None:
            info["edge"] = edge_res.info
        return ManagedResult(False, R0, t0, info, "none", edge_res, tex_res)

    # ------------------------------------------------------------------
    def draw(self, debug: np.ndarray, res: ManagedResult) -> None:
        if res.edge is not None and self.edge is not None:
            self.edge.draw(debug, res.edge)
        if res.texture is not None and self.texture is not None:
            self.texture.draw(debug, res.texture, draw_pred=res.edge is None)
