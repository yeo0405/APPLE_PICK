#!/usr/bin/env python3
"""ROS 2 endpoint for one RACE-6D RGB-D pose prediction."""
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from RACE_6D.core import PoseEstimator
from tomo_camera_tcp import CameraTCPClient


class PoseNode(Node):
    """Keep one RACE model and camera alive; process one request at a time."""

    def __init__(self, cfg: Dict[str, Any], args: Any) -> None:
        super().__init__("pose6dof_ros_node")
        self.cfg, self.args, self.bridge, self.processing = cfg, args, CvBridge(), False
        self.rgb: Optional[np.ndarray] = None
        camera_cfg, race_cfg = cfg["CAMERA"], cfg["RACE_6D"]
        project_root = Path(__file__).resolve().parents[1]
        model_config = Path(race_cfg["MODEL_CONFIG"])
        model_path = Path(race_cfg["MODEL_PATH"])
        if not model_config.is_absolute():
            model_config = project_root / model_config
        if not model_path.is_absolute():
            model_path = project_root / model_path
        self.pose_estimator = PoseEstimator(
            model_path=str(model_path), config_path=str(model_config), device=args.device,
            score_threshold=float(race_cfg.get("SCORE_THRESHOLD", .25)),
            max_per_class=int(race_cfg.get("MAX_PER_CLASS", 1)),
            max_detections=race_cfg.get("MAX_DETECTIONS"), depth_z_max_mm=race_cfg.get("DEPTH_Z_MAX_MM"),
            invalid_depth_value=int(race_cfg.get("INVALID_DEPTH_VALUE", 65535)), class_id=race_cfg.get("CLASS_ID"),
        )
        self.camera = CameraTCPClient(camera_cfg["IP"], camera_cfg["PORT"])
        if not self.camera.is_running() and not self.camera.start():
            raise RuntimeError("Camera start failed")
        self.camera_matrix = self.camera.get_color_intri_matrix()
        self.test_srv = self.create_service(Trigger, "/pose6dof/test", self.trigger_callback)
        self.pose_pub = self.create_publisher(Pose, "/pose6dof/result", 10)
        self.debug_pub = self.create_publisher(Image, "/pose6dof/debug_image", 10)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info("RACE-6D ready: /pose6dof/test -> /pose6dof/result")

    @staticmethod
    def _failed_pose() -> Pose:
        pose = Pose()
        pose.position.x = pose.position.y = pose.position.z = float("nan")
        pose.orientation.x = pose.orientation.y = pose.orientation.z = pose.orientation.w = float("nan")
        return pose

    def _debug(self, image: np.ndarray) -> None:
        msg = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.debug_pub.publish(msg)

    def _publish(self, result: Dict[str, Any]) -> None:
        if not result.get("success"):
            self.pose_pub.publish(self._failed_pose())
            if self.rgb is not None:
                self._debug(self.rgb)
            return
        t, q = result["translation"], result["quat"]  # metres; [qw,qx,qy,qz]
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = map(float, t)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = float(q[1]), float(q[2]), float(q[3]), float(q[0])
        self.pose_pub.publish(pose)
        self._debug(result["debug_image"])
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id, tf.child_frame_id = "Head_camera_link", "target"
        tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z = map(float, t)
        tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        self.tf_broadcaster.sendTransform(tf)

    def trigger_callback(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.processing:
            response.success, response.message = False, "busy"
            return response
        self.processing = True
        try:
            frame = self.camera.get_frame()
            if frame.is_empty:
                self._publish({"success": False})
                response.success, response.message = False, "camera failed"
                return response
            self.rgb, depth = self.camera.get_color_frame().data, self.camera.get_depth_frame().data
            result = self.pose_estimator.predict(self.rgb, depth, self.camera_matrix)
            self._publish(result)
            response.success = bool(result["success"])
            response.message = "Pose estimated." if response.success else "No RACE-6D detection above threshold."
        except Exception as error:
            self.get_logger().error(f"Pose trigger exception: {error}")
            self._publish({"success": False})
            response.success, response.message = False, str(error)
        finally:
            self.rgb = None
            self.processing = False
        return response

    def shutdown(self) -> None:
        if self.camera is not None:
            self.camera.stop()
