#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from typing import Any, Dict, Optional, Tuple

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from vision_msgs.msg import Detection3DArray

TRIGGER_SERVICES = [
    "/pose6dof/predict",
]

POSE_TOPIC = "/pose6dof/result"

LABEL_NAMES = {
    0: "iphone14_plus_blue",
    1: "brown_box",
    2: "flap",
    3: "iphone14_pro_max",
    4: "iphone14_plus_yellow",
}

DEFAULT_RATE_HZ = 1.0


class ContinuousTrigger(Node):
    """Continuously trigger Pose6D prediction and print Detection3DArray results."""

    def __init__(
        self,
        rate_hz: float,
        label_names: Dict[int, str],
    ) -> None:
        super().__init__("continuous_trigger")

        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be greater than 0.")

        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz
        self.label_names = dict(sorted(label_names.items()))
        self.name_to_label = {name: label for label, name in self.label_names.items()}

        self.trigger_clients = {}

        for service_name in TRIGGER_SERVICES:
            self.trigger_clients[service_name] = self.create_client(
                Trigger,
                service_name,
            )

        self.pose_subscription = self.create_subscription(
            Detection3DArray,
            POSE_TOPIC,
            self.pose_callback,
            10,
        )

        self.get_logger().info(f"Trigger services: {TRIGGER_SERVICES}")
        self.get_logger().info(f"Pose topic: {POSE_TOPIC} [Detection3DArray]")
        self.get_logger().info(f"Labels: {self.label_names}")
        self.get_logger().info(f"Rate: {self.rate_hz:.3f} Hz")
        self.get_logger().info(f"Period: {self.period:.3f} s")

        self.wait_for_services()

        self.timer = self.create_timer(
            self.period,
            self.publish_triggers,
        )

        self.get_logger().info("Continuous trigger started.")
        self.get_logger().info("Press Ctrl+C to stop.")

    def wait_for_services(self) -> None:
        """Wait until all configured Trigger services are available."""

        for service_name, client in self.trigger_clients.items():
            self.get_logger().info(f"Waiting for {service_name} ...")

            while rclpy.ok():
                if client.wait_for_service(timeout_sec=1.0):
                    break

                self.get_logger().warning(f"{service_name} not available...")

            if rclpy.ok():
                self.get_logger().info(f"Connected to {service_name}")

    def publish_triggers(self) -> None:
        """Send one Trigger request to every configured service."""

        for service_name, client in self.trigger_clients.items():
            if not client.service_is_ready():
                self.get_logger().warning(f"Service unavailable: {service_name}")
                continue

            client.call_async(Trigger.Request())

            self.get_logger().info(f"Triggered: {service_name}")

    @staticmethod
    def _valid_pose(pose) -> bool:
        values = [
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ]

        return all(math.isfinite(float(value)) for value in values)

    @staticmethod
    def _format_pose(pose) -> str:
        return (
            f"XYZ=("
            f"{pose.position.x:.6f}, "
            f"{pose.position.y:.6f}, "
            f"{pose.position.z:.6f}) "
            f"Q=("
            f"{pose.orientation.x:.6f}, "
            f"{pose.orientation.y:.6f}, "
            f"{pose.orientation.z:.6f}, "
            f"{pose.orientation.w:.6f})"
        )

    def _resolve_label(self, class_id: str) -> Optional[int]:
        """Resolve a detection class ID to the configured numeric label."""

        class_id = str(class_id).strip()

        if class_id in self.name_to_label:
            return self.name_to_label[class_id]

        try:
            label = int(class_id)
        except (TypeError, ValueError):
            return None

        if label in self.label_names:
            return label

        return None

    def pose_callback(self, msg: Detection3DArray) -> None:
        """Process Detection3DArray results and match them by class ID."""

        self.get_logger().info("========================================")
        self.get_logger().info(
            f"Received {len(msg.detections)} detections " f"from {POSE_TOPIC}"
        )

        detected: Dict[int, Tuple[float, Any]] = {}

        for detection in msg.detections:
            if not detection.results:
                self.get_logger().warning(
                    f"Detection id={detection.id!r} has no pose results."
                )
                continue

            matched_label = None
            matched_result = None

            for result in detection.results:
                class_id = result.hypothesis.class_id
                label = self._resolve_label(class_id)

                if label is not None:
                    matched_label = label
                    matched_result = result
                    break

            if matched_label is None:
                self.get_logger().warning(
                    f"Unknown detection: id={detection.id!r}, "
                    f"class_ids={[r.hypothesis.class_id for r in detection.results]}"
                )
                continue

            pose = matched_result.pose.pose
            score = float(matched_result.hypothesis.score)

            previous = detected.get(matched_label)

            if previous is None or score > previous[0]:
                detected[matched_label] = (score, pose)

        expected_count = len(self.label_names)

        if len(detected) != expected_count:
            self.get_logger().warning(
                f"Expected {expected_count} configured objects, "
                f"but received {len(detected)} unique recognized objects."
            )

        for label, name in self.label_names.items():
            if label not in detected:
                self.get_logger().warning(f"[{label}] name={name}: NOT DETECTED")
                continue

            score, pose = detected[label]

            if not self._valid_pose(pose):
                self.get_logger().warning(f"[{label}] name={name}: INVALID POSE")
                continue

            self.get_logger().info(
                f"[{label}] name={name} "
                f"score={score:.4f}: "
                f"{self._format_pose(pose)}"
            )

        self.get_logger().info("========================================")


def parse_labels(value: str) -> Dict[int, str]:
    """Parse labels in the form '0:iphone,1:box'."""

    result: Dict[int, str] = {}

    if not value.strip():
        raise ValueError("LABEL_NAMES cannot be empty.")

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        if ":" not in item:
            raise ValueError(
                f"Invalid label format: '{item}'. " "Expected 'label:name'."
            )

        label_text, name = item.split(":", 1)

        label = int(label_text.strip())
        name = name.strip()

        if not name:
            raise ValueError(f"Empty name for label {label}.")

        if label in result:
            raise ValueError(f"Duplicate label: {label}.")

        result[label] = name

    if not result:
        raise ValueError("LABEL_NAMES cannot be empty.")

    names = list(result.values())

    if len(names) != len(set(names)):
        raise ValueError("LABEL_NAMES contains duplicate object names.")

    return dict(sorted(result.items()))


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously trigger ROS 2 Pose6D prediction "
            "and print Detection3DArray results."
        )
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE_HZ,
        help=f"Trigger rate in Hz. Default: {DEFAULT_RATE_HZ}",
    )

    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(f"{label}:{name}" for label, name in LABEL_NAMES.items()),
        help="Object label mapping. Example: 0:iphone,1:box",
    )

    parsed_args = parser.parse_args(args=args)

    try:
        label_names = parse_labels(parsed_args.labels)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
        return

    rclpy.init(args=args)

    node = None

    try:
        node = ContinuousTrigger(
            rate_hz=parsed_args.rate,
            label_names=label_names,
        )
        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nStopping...", flush=True)

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
