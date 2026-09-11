#!/usr/bin/env python3

"""ROS 2 endpoint for RACE-6D RGB-D pose prediction."""

from pathlib import Path
from typing import Any, Dict, Optional
import time

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
    """Keep one RACE-6D model and camera alive."""

    def __init__(self, cfg: Dict[str, Any], args: Any) -> None:
        super().__init__("pose6dof_ros_node")

        self.cfg = cfg
        self.args = args
        self.bridge = CvBridge()
        self.processing = False

        self.rgb: Optional[np.ndarray] = None
        self.depth: Optional[np.ndarray] = None
        self.camera_matrix: Optional[np.ndarray] = None

        camera_cfg = cfg["CAMERA"]
        race_cfg = cfg["RACE_6D"]

        # ------------------------------------------------------------
        # Label → object name mapping
        # ------------------------------------------------------------

        label_names = race_cfg.get("LABEL_NAMES")

        if not isinstance(label_names, dict) or not label_names:
            raise ValueError(
                "RACE_6D.LABEL_NAMES must be a non-empty mapping, "
                "for example: {0: iphone, 1: box}"
            )

        self.label_names = {
            int(label): str(name)
            for label, name in label_names.items()
        }

        names = list(self.label_names.values())

        if any(not name for name in names):
            raise ValueError(
                "RACE_6D.LABEL_NAMES contains an empty object name."
            )

        if len(names) != len(set(names)):
            raise ValueError(
                "RACE_6D.LABEL_NAMES contains duplicate object names."
            )

        self.get_logger().info(
            f"Configured labels: {self.label_names}"
        )

        # ------------------------------------------------------------
        # Resolve RACE-6D paths
        # ------------------------------------------------------------

        project_root = Path(__file__).resolve().parents[1]

        model_config = Path(race_cfg["MODEL_CONFIG"])
        model_path = Path(race_cfg["MODEL_PATH"])

        if not model_config.is_absolute():
            model_config = project_root / model_config

        if not model_path.is_absolute():
            model_path = project_root / model_path

        self.get_logger().info("==============================")
        self.get_logger().info("RACE-6D ROS Node")
        self.get_logger().info(f"Model config : {model_config}")
        self.get_logger().info(f"Model path   : {model_path}")
        self.get_logger().info(f"Device       : {args.device}")
        self.get_logger().info(f"Labels       : {self.label_names}")
        self.get_logger().info("==============================")

        # ------------------------------------------------------------
        # RACE-6D
        # ------------------------------------------------------------

        self.pose_estimator = PoseEstimator(
            model_path=str(model_path),
            config_path=str(model_config),
            device=args.device,
            score_threshold=float(
                race_cfg.get("SCORE_THRESHOLD", 0.25)
            ),
            max_per_class=int(
                race_cfg.get("MAX_PER_CLASS", 1)
            ),
            max_detections=race_cfg.get("MAX_DETECTIONS"),
            depth_z_max_mm=race_cfg.get("DEPTH_Z_MAX_MM"),
            invalid_depth_value=int(
                race_cfg.get("INVALID_DEPTH_VALUE", 65535)
            ),
            class_id=None,
        )

        self.get_logger().info(
            "RACE-6D PoseEstimator initialized."
        )

        # ------------------------------------------------------------
        # Camera
        # ------------------------------------------------------------

        self.camera = CameraTCPClient(
            camera_cfg["IP"],
            camera_cfg["PORT"],
        )

        self.get_logger().info(
            f"Camera: {camera_cfg['IP']}:{camera_cfg['PORT']}"
        )

        if not self.camera.is_running():
            if not self.camera.start():
                raise RuntimeError("Camera start failed.")

        self.camera_matrix = self._load_camera_matrix()

        self.get_logger().info(
            "Camera intrinsic matrix loaded successfully:"
        )
        self.get_logger().info(f"\n{self.camera_matrix}")

        # ------------------------------------------------------------
        # ROS service
        # ------------------------------------------------------------

        self.predict_srv = self.create_service(
            Trigger,
            "/pose6dof/predict",
            self.trigger_callback,
        )

        # ------------------------------------------------------------
        # Pose publishers
        # ------------------------------------------------------------

        self.pose_publishers = {}

        for label, name in self.label_names.items():
            topic = f"/pose6dof/{name}"
            self.pose_publishers[label] = self.create_publisher(
                Pose,
                topic,
                10,
            )
            self.get_logger().info(
                f"Pose topic: label={label} -> {topic}"
            )

        self.debug_pub = self.create_publisher(
            Image,
            "/pose6dof/debug_image",
            10,
        )

        # ------------------------------------------------------------
        # TF
        # ------------------------------------------------------------

        self.tf_broadcaster = TransformBroadcaster(self)
        self.parent_frame = "Head_camera_link"

        self.get_logger().info("==============================")
        self.get_logger().info("RACE-6D ROS node started.")
        self.get_logger().info("Service : /pose6dof/predict")
        self.get_logger().info("Debug   : /pose6dof/debug_image")

        for label, name in self.label_names.items():
            self.get_logger().info(
                f"label={label}: "
                f"/pose6dof/{name}, TF child={name}"
            )

        self.get_logger().info("==============================")

    def _load_camera_matrix(
        self,
        max_retries: int = 10,
        retry_delay: float = 0.5,
    ) -> np.ndarray:
        """Wait for and load camera color intrinsic matrix."""

        self.get_logger().info(
            "Waiting for camera color intrinsic matrix..."
        )

        last_value = None

        for attempt in range(1, max_retries + 1):
            try:
                matrix = self.camera.get_color_intri_matrix()
                last_value = matrix

                if matrix is not None:
                    matrix = np.asarray(matrix, dtype=np.float32)

                    self.get_logger().info(
                        f"Camera intrinsic attempt "
                        f"{attempt}/{max_retries}: "
                        f"shape={matrix.shape}"
                    )

                    if matrix.shape == (3, 3):
                        return matrix

                    self.get_logger().warning(
                        f"Invalid intrinsic shape: {matrix.shape}"
                    )
                else:
                    self.get_logger().warning(
                        f"Intrinsic not available "
                        f"(attempt {attempt}/{max_retries})."
                    )

            except Exception as error:
                self.get_logger().warning(
                    f"Failed to get intrinsic "
                    f"(attempt {attempt}/{max_retries}): "
                    f"{type(error).__name__}: {error}"
                )

            if attempt < max_retries:
                time.sleep(retry_delay)

        raise RuntimeError(
            "Failed to obtain valid 3x3 camera intrinsic "
            f"after {max_retries} attempts. "
            f"Last value: {last_value}"
        )

    @staticmethod
    def _failed_pose() -> Pose:
        """Create invalid pose message."""

        pose = Pose()

        pose.position.x = float("nan")
        pose.position.y = float("nan")
        pose.position.z = float("nan")

        pose.orientation.x = float("nan")
        pose.orientation.y = float("nan")
        pose.orientation.z = float("nan")
        pose.orientation.w = float("nan")

        return pose

    def _debug(self, image: Optional[np.ndarray]) -> None:
        """Publish debug image."""

        if image is None:
            return

        msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"

        self.debug_pub.publish(msg)

    def _publish_detection(
        self,
        detection: Dict[str, Any],
    ) -> None:
        """Publish one detection according to its configured label."""

        label = detection.get("label")

        if label is None:
            self.get_logger().warning(
                "Detection has no label."
            )
            return

        label = int(label)

        if label not in self.label_names:
            self.get_logger().warning(
                f"Detected label={label}, "
                "but it is not configured in LABEL_NAMES. "
                f"Known labels={list(self.label_names.keys())}"
            )
            return

        name = self.label_names[label]
        publisher = self.pose_publishers[label]

        t = detection.get("translation")
        q = detection.get("quat")

        if t is None or q is None:
            self.get_logger().warning(
                f"Detection label={label} ({name}) "
                "has no translation or quaternion."
            )
            return

        pose = Pose()

        pose.position.x = float(t[0])
        pose.position.y = float(t[1])
        pose.position.z = float(t[2])

        # RACE-6D: [qw, qx, qy, qz]
        # ROS:     [qx, qy, qz, qw]

        pose.orientation.x = float(q[1])
        pose.orientation.y = float(q[2])
        pose.orientation.z = float(q[3])
        pose.orientation.w = float(q[0])

        publisher.publish(pose)

        # ------------------------------------------------------------
        # TF
        # ------------------------------------------------------------

        tf = TransformStamped()

        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.parent_frame
        tf.child_frame_id = name

        tf.transform.translation.x = float(t[0])
        tf.transform.translation.y = float(t[1])
        tf.transform.translation.z = float(t[2])

        tf.transform.rotation.x = pose.orientation.x
        tf.transform.rotation.y = pose.orientation.y
        tf.transform.rotation.z = pose.orientation.z
        tf.transform.rotation.w = pose.orientation.w

        self.tf_broadcaster.sendTransform(tf)

        confidence = detection.get("confidence")

        if confidence is not None:
            self.get_logger().info(
                f"Published {name} "
                f"(label={label}) "
                f"confidence={float(confidence):.6f}"
            )
        else:
            self.get_logger().info(
                f"Published {name} (label={label})"
            )

    def _get_camera_frame(self):
        """Get one RGB-D frame from camera."""

        frame = self.camera.get_frame()

        if frame.is_empty:
            self.get_logger().warning(
                "Camera returned empty frame."
            )
            return None, None

        rgb_frame = self.camera.get_color_frame()
        depth_frame = self.camera.get_depth_frame()

        if rgb_frame is None or depth_frame is None:
            self.get_logger().error(
                "Camera returned empty RGB or depth frame."
            )
            return None, None

        rgb = rgb_frame.data
        depth = depth_frame.data

        if rgb is None or depth is None:
            self.get_logger().error(
                "Camera returned None RGB/depth data."
            )
            return None, None

        return rgb, depth

    def trigger_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:

        del request

        if self.processing:
            response.success = False
            response.message = "busy"
            return response

        self.processing = True

        try:
            self.get_logger().info(
                "========================================"
            )
            self.get_logger().info(
                "Pose prediction triggered."
            )

            rgb, depth = self._get_camera_frame()

            if rgb is None or depth is None:
                self._publish_failed_all()
                response.success = False
                response.message = "camera failed"
                return response

            self.rgb = rgb
            self.depth = depth

            self.get_logger().info(
                f"RGB   : shape={rgb.shape}, dtype={rgb.dtype}"
            )
            self.get_logger().info(
                f"Depth : shape={depth.shape}, dtype={depth.dtype}"
            )

            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise RuntimeError(
                    f"Invalid RGB shape: {rgb.shape}"
                )

            if depth.ndim != 2:
                raise RuntimeError(
                    f"Invalid depth shape: {depth.shape}"
                )

            if depth.shape != rgb.shape[:2]:
                raise RuntimeError(
                    "RGB/depth resolution mismatch: "
                    f"RGB={rgb.shape[:2]}, "
                    f"Depth={depth.shape}"
                )

            if self.camera_matrix is None:
                raise RuntimeError(
                    "Camera intrinsic matrix is None."
                )

            camera_matrix = np.asarray(
                self.camera_matrix,
                dtype=np.float32,
            )

            if camera_matrix.shape != (3, 3):
                raise RuntimeError(
                    "Invalid camera intrinsic matrix shape: "
                    f"{camera_matrix.shape}"
                )

            # --------------------------------------------------------
            # RACE-6D prediction
            # --------------------------------------------------------

            self.get_logger().info(
                "Running RACE-6D prediction..."
            )

            result = self.pose_estimator.predict(
                rgb=rgb,
                depth=depth,
                intrinsic=camera_matrix,
            )

            detections = result.get("detections")

            if not isinstance(detections, list):
                self.get_logger().error(
                    "RACE-6D returned invalid "
                    "'detections' result."
                )

                self._publish_failed_all()

                response.success = False
                response.message = "Invalid RACE-6D result."
                return response

            # --------------------------------------------------------
            # Debug all detections
            # --------------------------------------------------------

            self.get_logger().info(
                f"RACE-6D returned {len(detections)} detections."
            )

            detected_labels = []

            for i, detection in enumerate(detections):
                if not isinstance(detection, dict):
                    self.get_logger().warning(
                        f"Detection #{i} is not a dict."
                    )
                    continue

                label = detection.get("label")
                confidence = detection.get("confidence")

                if label is None:
                    self.get_logger().warning(
                        f"Detection #{i} has no label."
                    )
                    continue

                label = int(label)
                detected_labels.append(label)

                name = self.label_names.get(
                    label,
                    "UNCONFIGURED",
                )

                if confidence is not None:
                    self.get_logger().info(
                        f"  Detection #{i}: "
                        f"label={label}, "
                        f"name={name}, "
                        f"confidence={float(confidence):.6f}"
                    )
                else:
                    self.get_logger().info(
                        f"  Detection #{i}: "
                        f"label={label}, "
                        f"name={name}"
                    )

            self.get_logger().info(
                f"Detected labels: {detected_labels}"
            )

            # --------------------------------------------------------
            # Publish every configured detection
            # --------------------------------------------------------

            published_labels = set()

            for detection in detections:
                if not isinstance(detection, dict):
                    continue

                label = detection.get("label")

                if label is None:
                    continue

                label = int(label)

                if label not in self.label_names:
                    continue

                self._publish_detection(detection)
                published_labels.add(label)

            # --------------------------------------------------------
            # Publish invalid pose for configured labels that
            # were not detected.
            # --------------------------------------------------------

            for label, name in self.label_names.items():
                if label in published_labels:
                    continue

                self.pose_publishers[label].publish(
                    self._failed_pose()
                )

                self.get_logger().warning(
                    f"No detection for label={label} "
                    f"({name}). Published invalid pose."
                )

            self._debug(
                result.get("debug_image")
                if result.get("debug_image") is not None
                else rgb
            )

            response.success = len(published_labels) > 0

            if response.success:
                published_names = [
                    self.label_names[label]
                    for label in sorted(published_labels)
                ]

                response.message = (
                    "Pose estimated for: "
                    + ", ".join(published_names)
                )
            else:
                response.message = (
                    "No configured RACE-6D object detected."
                )

            self.get_logger().info(
                "========================================"
            )

        except Exception as error:
            self.get_logger().error(
                f"Pose prediction exception: "
                f"{type(error).__name__}: {error}"
            )

            self._publish_failed_all()

            response.success = False
            response.message = str(error)

        finally:
            self.rgb = None
            self.depth = None
            self.processing = False

        return response

    def _publish_failed_all(self) -> None:
        """Publish invalid poses for all configured objects."""

        failed = self._failed_pose()

        for label, publisher in self.pose_publishers.items():
            publisher.publish(failed)

            self.get_logger().warning(
                f"Published invalid pose for "
                f"label={label} ({self.label_names[label]})."
            )

    def shutdown(self) -> None:
        """Stop camera."""

        self.get_logger().info(
            "Shutting down PoseNode..."
        )

        try:
            if self.camera is not None:
                self.camera.stop()
        except Exception as error:
            self.get_logger().warning(
                f"Camera stop error: {error}"
            )