#!/usr/bin/env python3

from __future__ import annotations

import argparse
from typing import Any

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


# ============================================================
# Global Configuration
# ============================================================

TRIGGER_SERVICES = [
    "/pose6dof/predict",
]

DEFAULT_RATE_HZ = 1.0


class ContinuousTrigger(Node):
    """Continuously send Trigger requests."""

    def __init__(self, rate_hz: float) -> None:
        super().__init__("continuous_trigger")

        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be greater than 0.")

        self.rate_hz = rate_hz
        self.period = 1.0 / rate_hz

        self.trigger_clients = {}

        for service_name in TRIGGER_SERVICES:
            self.trigger_clients[service_name] = self.create_client(
                Trigger,
                service_name,
            )

        self.get_logger().info(
            f"Trigger services: {TRIGGER_SERVICES}"
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

                if client.wait_for_service(
                    timeout_sec=1.0
                ):
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


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously publish ROS 2 Trigger requests."
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

    parsed_args = parser.parse_args()

    rclpy.init(args=args)

    node = None

    try:
        node = ContinuousTrigger(
            rate_hz=parsed_args.rate,
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