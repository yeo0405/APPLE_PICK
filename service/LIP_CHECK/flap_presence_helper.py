from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from RACE_6D.core import PoseEstimator


class FlapPresenceHelper:
    BROWN_BOX_ID = 1
    FLAP_ID = 2

    COLORS = {
        BROWN_BOX_ID: (0, 165, 255),
        FLAP_ID: (0, 255, 0),
    }

    STATUS_COLORS = {
        "PRESENT": (0, 200, 0),
        "ABSENT": (0, 0, 255),
    }

    def __init__(
        self,
        pose_estimator: PoseEstimator,
    ) -> None:
        if pose_estimator is None:
            raise ValueError("pose_estimator must not be None.")

        self.estimator = pose_estimator
        self.debug_image: np.ndarray | None = None

        self.label_names = {
            self.BROWN_BOX_ID: "brown_box",
            self.FLAP_ID: "flap",
        }

        print("[FlapPresenceHelper] Using existing PoseEstimator.")

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: list,
        status: str,
    ) -> np.ndarray:
        debug = image.copy()

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            label = detection.get("label")
            bbox = detection.get("bbox_xyxy")

            if label is None or bbox is None:
                continue

            label = int(label)

            if label not in (self.BROWN_BOX_ID, self.FLAP_ID):
                continue

            bbox = np.asarray(bbox, dtype=np.float32).reshape(-1)

            if bbox.size != 4 or not np.all(np.isfinite(bbox)):
                continue

            x1, y1, x2, y2 = np.round(bbox).astype(int)

            h, w = debug.shape[:2]

            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            x2 = int(np.clip(x2, 0, w - 1))
            y2 = int(np.clip(y2, 0, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            color = self.COLORS[label]
            class_name = self.label_names.get(label, str(label))
            confidence = float(detection.get("confidence", 0.0))

            cv2.rectangle(
                debug,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            text = f"{class_name} {confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                thickness,
            )

            text_y = max(y1, text_h + baseline)

            cv2.rectangle(
                debug,
                (x1, text_y - text_h - baseline),
                (x1 + text_w, text_y),
                color,
                -1,
            )

            cv2.putText(
                debug,
                text,
                (x1, text_y - baseline),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        self._draw_status(debug, status)

        return debug

    def _draw_status(
        self,
        image: np.ndarray,
        status: str,
    ) -> None:
        h, w = image.shape[:2]

        text = status
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3

        (text_w, text_h), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )

        padding = 15

        x2 = w - 20
        y1 = 20
        x1 = x2 - text_w - padding * 2
        y2 = y1 + text_h + baseline + padding * 2

        x1 = max(0, x1)
        y2 = min(h, y2)

        color = self.STATUS_COLORS[status]

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            color,
            -1,
        )

        cv2.putText(
            image,
            text,
            (x1 + padding, y1 + text_h + padding),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    def predict(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        intrinsic: np.ndarray,
    ) -> str:
        """
        Detect Brown Box and Flap using the shared RACE-6D PoseEstimator.

        Returns:
            "PRESENT": Flap detected without Brown Box,
                       or neither object detected.
            "ABSENT": Brown Box detected.

        Debug:
            self.debug_image: Image with detection boxes and final status.
        """

        self.debug_image = None

        if not isinstance(rgb, np.ndarray):
            rgb = np.asarray(rgb)

        if not isinstance(depth, np.ndarray):
            depth = np.asarray(depth)

        intrinsic = np.asarray(intrinsic, dtype=np.float32)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"Invalid RGB shape: {rgb.shape}")

        if depth.ndim != 2:
            raise ValueError(f"Invalid depth shape: {depth.shape}")

        if depth.shape != rgb.shape[:2]:
            raise ValueError(f"RGB/Depth mismatch: {rgb.shape[:2]} vs {depth.shape}")

        if intrinsic.shape != (3, 3):
            raise ValueError(f"Invalid intrinsic matrix shape: {intrinsic.shape}")

        result = self.estimator.predict(
            rgb=rgb,
            depth=depth,
            intrinsic=intrinsic,
        )

        if not isinstance(result, dict):
            raise RuntimeError("Invalid PoseEstimator result.")

        detections = result.get("detections")

        if not isinstance(detections, list):
            raise RuntimeError("Invalid RACE-6D detections.")

        detected_labels = set()

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            label = detection.get("label")

            if label is None:
                continue

            detected_labels.add(int(label))

        brown_box_present = self.BROWN_BOX_ID in detected_labels
        flap_present = self.FLAP_ID in detected_labels

        if brown_box_present and flap_present:
            status = "ABSENT"
        # elif brown_box_present:
        #     status = "ABSENT"
        # elif flap_present:
        #     status = "PRESENT"
        else:
            status = "PRESENT"

        self.debug_image = self._draw_detections(
            rgb,
            detections,
            status,
        )

        print(
            f"[FlapPresenceHelper] "
            f"Brown Box: {'PRESENT' if brown_box_present else 'ABSENT'} | "
            f"Flap: {'PRESENT' if flap_present else 'ABSENT'} | "
            f"Result: {status}"
        )

        return status
