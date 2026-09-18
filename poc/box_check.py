#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

TRIGGER_SERVICE = "/box_check/predict"
RESULT_TOPIC = "/box_check/result"


class ContinuousTrigger(Node):
    def __init__(self) -> None:
        super().__init__("continuous_box_check_trigger")

        self.trigger_client = self.create_client(Trigger, TRIGGER_SERVICE)
        self.result_subscriber = self.create_subscription(
            String, RESULT_TOPIC, self.result_callback, 10
        )

        self.request_start_time: float | None = None
        self.wait_for_service()

        self.get_logger().info("Continuous SAM2 Box Check started.")
        self.get_logger().info("Press Ctrl+C to stop.")

        self.send_trigger()

    def wait_for_service(self) -> None:
        self.get_logger().info(f"Waiting for {TRIGGER_SERVICE} ...")

        while rclpy.ok():
            if self.trigger_client.wait_for_service(timeout_sec=1.0):
                break
            self.get_logger().warning(f"{TRIGGER_SERVICE} not available...")

        if rclpy.ok():
            self.get_logger().info(f"Connected to {TRIGGER_SERVICE}")

    def send_trigger(self) -> None:
        if not self.trigger_client.service_is_ready():
            self.get_logger().warning(f"Service unavailable: {TRIGGER_SERVICE}")
            return

        self.request_start_time = time.perf_counter()
        future = self.trigger_client.call_async(Trigger.Request())
        future.add_done_callback(self.trigger_response_callback)

        self.get_logger().info("Triggered Box Check.")

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

    def result_callback(self, msg: String) -> None:
        self.get_logger().info(f"[RESULT] {msg.data}")


def main(args: Any = None) -> None:
    parser = argparse.ArgumentParser(
        description="Continuously trigger Box Check after each response."
    )
    parser.parse_args(args)

    rclpy.init(args=args)
    node = None

    try:
        node = ContinuousTrigger()
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