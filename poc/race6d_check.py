#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
from typing import Any, Dict

import rclpy
from geometry_msgs.msg import PoseArray
from rclpy.node import Node
from std_srvs.srv import Trigger


TRIGGER_SERVICES = [
    "/pose6dof/predict",
]

POSE_TOPIC = "/pose6dof/result"

LABEL_NAMES = {
    0: "iphone14_plus_blue",
    1: "brown_box",
    2: "flap",
    3: "iphone14_pro_max",
    4: "iphone14_plus_yellow"
}

DEFAULT_RATE_HZ = 1.0


class ContinuousTrigger(Node):
    """Continuously trigger Pose6D prediction and print PoseArray results."""

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

        self.trigger_clients = {}

        for service_name in TRIGGER_SERVICES:
            self.trigger_clients[service_name] = self.create_client(
                Trigger,
                service_name,
            )

        self.pose_subscription = self.create_subscription(
            PoseArray,
            POSE_TOPIC,
            self.pose_callback,
            10,
        )

        self.get_logger().info(
            f"Trigger services: {TRIGGER_SERVICES}"
        )
        self.get_logger().info(
            f"Pose topic: {POSE_TOPIC}"
        )
        self.get_logger().info(
            f"Labels: {self.label_names}"
        )
        self.get_logger().info(
            f"Rate: {self.rate_hz:.3f} Hz"
        )
        self.get_logger().info(
            f"Period: {self.period:.3f} s"
        )

        self.wait_for_services()

        self.timer = self.create_timer(
            self.period,
            self.publish_triggers,
        )

        self.get_logger().info(
            "Continuous trigger started."
        )
        self.get_logger().info(
            "Press Ctrl+C to stop."
        )

    def wait_for_services(self) -> None:
        """Wait until all configured Trigger services are available."""

        for service_name, client in self.trigger_clients.items():
            self.get_logger().info(
                f"Waiting for {service_name} ..."
            )

            while rclpy.ok():
                if client.wait_for_service(timeout_sec=1.0):
                    break

                self.get_logger().warning(
                    f"{service_name} not available..."
                )

            if rclpy.ok():
                self.get_logger().info(
                    f"Connected to {service_name}"
                )

    def publish_triggers(self) -> None:
        """Send one Trigger request to every configured service."""

        for service_name, client in self.trigger_clients.items():
            if not client.service_is_ready():
                self.get_logger().warning(
                    f"Service unavailable: {service_name}"
                )
                continue

            request = Trigger.Request()
            client.call_async(request)

            self.get_logger().info(
                f"Triggered: {service_name}"
            )

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

    def pose_callback(self, msg: PoseArray) -> None:
        """Process the combined PoseArray result."""

        self.get_logger().info(
            "========================================"
        )

        self.get_logger().info(
            f"Received {len(msg.poses)} poses "
            f"from {POSE_TOPIC}"
        )

        expected_count = len(self.label_names)

        if len(msg.poses) != expected_count:
            self.get_logger().warning(
                f"Expected {expected_count} poses, "
                f"but received {len(msg.poses)}."
            )

        for index, label in enumerate(self.label_names):
            name = self.label_names[label]

            if index >= len(msg.poses):
                self.get_logger().warning(
                    f"[{index}] label={label} "
                    f"name={name}: MISSING"
                )
                continue

            pose = msg.poses[index]

            if not self._valid_pose(pose):
                self.get_logger().warning(
                    f"[{index}] label={label} "
                    f"name={name}: NOT DETECTED"
                )
                continue

            self.get_logger().info(
                f"[{index}] label={label} "
                f"name={name}: "
                f"{self._format_pose(pose)}"
            )

        self.get_logger().info(
            "========================================"
        )


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
                f"Invalid label format: '{item}'. "
                "Expected 'label:name'."
            )

        label_text, name = item.split(":", 1)

        label = int(label_text.strip())
        name = name.strip()

        if not name:
            raise ValueError(
                f"Empty name for label {label}."
            )

        if label in result:
            raise ValueError(
                f"Duplicate label: {label}."
            )

        result[label] = name

    if not result:
        raise ValueError("LABEL_NAMES cannot be empty.")

    names = list(result.values())

    if len(names) != len(set(names)):
        raise ValueError(
            "LABEL_NAMES contains duplicate object names."
        )

    return dict(sorted(result.items()))


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously trigger ROS 2 Pose6D prediction "
            "and print PoseArray results."
        )
    )

    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_RATE_HZ,
        help=(
            "Trigger rate in Hz. "
            f"Default: {DEFAULT_RATE_HZ}"
        ),
    )

    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(
            f"{label}:{name}"
            for label, name in LABEL_NAMES.items()
        ),
        help=(
            "Pose label mapping. "
            "Example: 0:iphone,1:box"
        ),
    )

    parsed_args = parser.parse_args()

    try:
        label_names = parse_labels(
            parsed_args.labels
        )
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
        print(
            "\nStopping...",
            flush=True,
        )

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()