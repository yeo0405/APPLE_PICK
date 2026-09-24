#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_srvs.srv import Trigger

TRIGGER_SERVICE = "/alignment/predict"
RESULT_TOPIC = "/alignment/result"


class ContinuousAlignment(Node):
    def __init__(self) -> None:
        super().__init__("continuous_alignment_trigger")

        self.trigger_client = self.create_client(
            Trigger,
            TRIGGER_SERVICE,
        )
        self.result_subscriber = self.create_subscription(
            Pose,
            RESULT_TOPIC,
            self.result_callback,
            10,
        )

        self.request_start_time: float | None = None
        self.wait_for_service()

        self.get_logger().info(
            "Continuous Alignment started."
        )
        self.get_logger().info(
            "Press Ctrl+C to stop."
        )

        self.send_trigger()

    def wait_for_service(self) -> None:
        self.get_logger().info(
            f"Waiting for {TRIGGER_SERVICE} ..."
        )

        while rclpy.ok():
            if self.trigger_client.wait_for_service(
                timeout_sec=1.0
            ):
                break

            self.get_logger().warning(
                f"{TRIGGER_SERVICE} not available..."
            )

        if rclpy.ok():
            self.get_logger().info(
                f"Connected to {TRIGGER_SERVICE}"
            )

    def send_trigger(self) -> None:
        if not self.trigger_client.service_is_ready():
            self.get_logger().warning(
                f"Service unavailable: {TRIGGER_SERVICE}"
            )
            return

        self.request_start_time = time.perf_counter()

        future = self.trigger_client.call_async(
            Trigger.Request()
        )
        future.add_done_callback(
            self.trigger_response_callback
        )

        self.get_logger().info(
            "Triggered Alignment."
        )

    def trigger_response_callback(self, future) -> None:
        elapsed = (
            time.perf_counter() - self.request_start_time
            if self.request_start_time is not None
            else 0.0
        )

        try:
            response = future.result()

            self.get_logger().info(
                f"[SERVICE] success={response.success} "
                f"message={response.message} "
                f"time={elapsed * 1000.0:.1f} ms"
            )

        except Exception as error:
            self.get_logger().error(
                f"Trigger request failed: {error} "
                f"time={elapsed * 1000.0:.1f} ms"
            )

        finally:
            self.request_start_time = None

            if rclpy.ok():
                self.send_trigger()

    def result_callback(self, msg: Pose) -> None:
        self.get_logger().info(
            "[RESULT] "
            f"Delta XYZ=("
            f"{msg.position.x:.6f}, "
            f"{msg.position.y:.6f}, "
            f"{msg.position.z:.6f}) "
            f"Delta Q=("
            f"{msg.orientation.x:.6f}, "
            f"{msg.orientation.y:.6f}, "
            f"{msg.orientation.z:.6f}, "
            f"{msg.orientation.w:.6f})"
        )


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously trigger Alignment after "
            "each service response."
        )
    )
    parser.parse_args(args)

    rclpy.init(args=args)
    node = None

    try:
        node = ContinuousAlignment()
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