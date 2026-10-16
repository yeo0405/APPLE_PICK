#!/usr/bin/env python3

from __future__ import annotations

import argparse
import time
from typing import Any

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from robot_localization.srv import SetPose

TRIGGER_SERVICE = "/alignment/predict"
RESULT_TOPIC = "/alignment/result"

TARGET_XYZ = [
    -0.1437396978855786,
    -0.06308379248382938,
    0.25,
]

TARGET_QUATERNION = [
    -0.1852490322102135,
    0.01065864156467927,
    0.015970591181817106,
    0.9825040099879889,
]


class ContinuousAlignment(Node):
    def __init__(self) -> None:
        super().__init__("continuous_alignment_trigger")

        self.trigger_client = self.create_client(
            SetPose,
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

        self.get_logger().info("Continuous Alignment started.")
        self.get_logger().info("Press Ctrl+C to stop.")
        self.get_logger().info(
            f"Target XYZ={TARGET_XYZ}, " f"Quaternion XYZW={TARGET_QUATERNION}"
        )

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

        request = SetPose.Request()
        stamped = request.pose

        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = "Head_camera_link"

        pose = stamped.pose.pose
        pose.position.x = TARGET_XYZ[0]
        pose.position.y = TARGET_XYZ[1]
        pose.position.z = TARGET_XYZ[2]

        pose.orientation.x = TARGET_QUATERNION[0]
        pose.orientation.y = TARGET_QUATERNION[1]
        pose.orientation.z = TARGET_QUATERNION[2]
        pose.orientation.w = TARGET_QUATERNION[3]

        self.request_start_time = time.perf_counter()

        future = self.trigger_client.call_async(request)
        future.add_done_callback(self.trigger_response_callback)

        self.get_logger().info("Triggered Alignment.")

    def trigger_response_callback(self, future) -> None:
        elapsed = (
            time.perf_counter() - self.request_start_time
            if self.request_start_time is not None
            else 0.0
        )

        try:
            future.result()
            self.get_logger().info(
                f"[SERVICE] Response received " f"time={elapsed * 1000.0:.1f} ms"
            )

        except Exception as error:
            self.get_logger().error(
                f"Alignment request failed: {error} " f"time={elapsed * 1000.0:.1f} ms"
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
        description="Continuously trigger Alignment with a fixed target pose."
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
