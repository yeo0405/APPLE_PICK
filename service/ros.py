#!/usr/bin/env python3

"""ROS 2 endpoint for RACE-6D RGB-D pose prediction, DINOv3 box check and Alignment."""

from pathlib import Path
from typing import Any, Dict, Optional
import time

import numpy as np
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from RACE_6D.core import PoseEstimator
from BOX_CHECK.ear_esitimator import DINOv3Estimator
from ALIGNMENT.alignment import AlignmentEstimator
from tomo_camera_tcp import CameraTCPClient


class PoseNode(Node):
    """Keep RACE-6D, DINOv3, Alignment and camera alive."""

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

        # Keep original RACE-6D project root unchanged.
        project_root = Path(__file__).resolve().parents[1]

        model_config = Path(race_cfg["MODEL_CONFIG"])
        model_path = Path(race_cfg["MODEL_PATH"])

        if not model_config.is_absolute():
            model_config = project_root / model_config

        if not model_path.is_absolute():
            model_path = project_root / model_path

        self.get_logger().info("==============================")
        self.get_logger().info("RACE-6D + DINOv3 BOX CHECK ROS Node")
        self.get_logger().info(f"Model config : {model_config}")
        self.get_logger().info(f"Model path   : {model_path}")
        self.get_logger().info(f"Device       : {args.device}")
        self.get_logger().info(f"Labels       : {self.label_names}")
        self.get_logger().info("==============================")

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
        # DINOv3 Box Check
        # ------------------------------------------------------------

        box_cfg = cfg.get("BOX_CHECK", {})

        self.box_check_enabled = bool(
            box_cfg.get("ENABLED", True)
        )

        self.box_check_estimator: Optional[DINOv3Estimator] = None

        if self.box_check_enabled:
            self.get_logger().info(
                "Initializing DINOv3 Box Check..."
            )

            self.box_check_estimator = DINOv3Estimator(
                sim_threshold=float(
                    box_cfg.get("SIM_THRESHOLD", 0.8)
                ),
                coverage_threshold=float(
                    box_cfg.get("COVERAGE_THRESHOLD", 0.2)
                ),
            )

            self.get_logger().info(
                "DINOv3 Box Check initialized."
            )
        else:
            self.get_logger().info(
                "DINOv3 Box Check disabled."
            )

        # ------------------------------------------------------------
        # Alignment
        # ------------------------------------------------------------

        alignment_cfg = cfg.get("ALIGNMENT", {})
        self.alignment_enabled = bool(
            alignment_cfg.get("ENABLED", True)
        )

        self.alignment_estimator: Optional[AlignmentEstimator] = None

        if self.alignment_enabled:
            alignment_root = Path(__file__).resolve().parent

            gt_json = Path(
                alignment_cfg.get(
                    "GT_JSON",
                    "ALIGNMENT/GT/gt.json",
                )
            )

            if not gt_json.is_absolute():
                gt_json = alignment_root / gt_json

            self.get_logger().info(
                "Initializing Alignment..."
            )
            self.get_logger().info(
                f"Alignment GT: {gt_json}"
            )

            if not gt_json.is_file():
                raise FileNotFoundError(
                    f"Alignment GT JSON not found: {gt_json}"
                )

            self.alignment_estimator = AlignmentEstimator(
                gt_json=gt_json,
                device=args.device,
            )

            self.get_logger().info(
                "Alignment initialized."
            )
        else:
            self.get_logger().info(
                "Alignment disabled."
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
        self.get_logger().info(
            f"\n{self.camera_matrix}"
        )

        # ------------------------------------------------------------
        # Services
        # ------------------------------------------------------------

        self.predict_srv = self.create_service(
            Trigger,
            "/pose6dof/predict",
            self.trigger_callback,
        )

        self.box_check_srv = self.create_service(
            Trigger,
            "/box_check/predict",
            self.box_check_callback,
        )

        self.alignment_srv = self.create_service(
            Trigger,
            "/alignment/predict",
            self.alignment_callback,
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

        # ------------------------------------------------------------
        # Debug publishers
        # ------------------------------------------------------------

        self.debug_pub = self.create_publisher(
            Image,
            "/pose6dof/debug_image",
            10,
        )

        self.box_check_result_pub = self.create_publisher(
            String,
            "/box_check/result",
            10,
        )

        self.box_check_debug_pub = self.create_publisher(
            Image,
            "/box_check/debug_image",
            10,
        )

        self.alignment_result_pub = self.create_publisher(
            Pose,
            "/alignment/result",
            10,
        )

        self.alignment_debug_pub = self.create_publisher(
            Image,
            "/alignment/debug_image",
            10,
        )

        # ------------------------------------------------------------
        # TF
        # ------------------------------------------------------------

        self.tf_broadcaster = TransformBroadcaster(self)
        self.parent_frame = "Head_camera_link"
        self.ripcord_frame = "ripcord"

        self.get_logger().info("==============================")
        self.get_logger().info("RACE-6D ROS node started.")
        self.get_logger().info("Service : /pose6dof/predict")
        self.get_logger().info("Debug   : /pose6dof/debug_image")
        self.get_logger().info("Service : /box_check/predict")
        self.get_logger().info("Result  : /box_check/result")
        self.get_logger().info("Debug   : /box_check/debug_image")
        self.get_logger().info("Service : /alignment/predict")
        self.get_logger().info("Result  : /alignment/result")
        self.get_logger().info("Debug   : /alignment/debug_image")
        self.get_logger().info(
            f"Alignment TF: {self.parent_frame} -> {self.ripcord_frame}"
        )

        for label, name in self.label_names.items():
            self.get_logger().info(
                f"label={label}: "
                f"/pose6dof/{name}, TF child={name}"
            )

        self.get_logger().info("==============================")

    def _load_camera_matrix(
        self,
        max_retries: int = 10,
        retry_delay: float = 2.0,
    ) -> np.ndarray:

        self.get_logger().info(
            "Waiting for camera color intrinsic matrix..."
        )

        last_value = None

        for attempt in range(1, max_retries + 1):
            try:
                matrix = self.camera.get_color_intri_matrix()
                last_value = matrix

                if matrix is not None:
                    matrix = np.asarray(
                        matrix,
                        dtype=np.float32,
                    )

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
        if image is None:
            return

        msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        msg.header.frame_id = "camera"

        self.debug_pub.publish(msg)

    def _publish_box_debug(
        self,
        image: Optional[np.ndarray],
    ) -> None:

        if image is None:
            return

        msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        msg.header.frame_id = "camera"

        self.box_check_debug_pub.publish(msg)

    def _publish_alignment_debug(
        self,
        image: Optional[np.ndarray],
    ) -> None:

        if image is None:
            return

        msg = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )

        msg.header.stamp = (
            self.get_clock().now().to_msg()
        )
        msg.header.frame_id = "camera"

        self.alignment_debug_pub.publish(msg)

    def _publish_detection(
        self,
        detection: Dict[str, Any],
    ) -> None:

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
                "but it is not configured in LABEL_NAMES."
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

        pose.orientation.x = float(q[1])
        pose.orientation.y = float(q[2])
        pose.orientation.z = float(q[3])
        pose.orientation.w = float(q[0])

        publisher.publish(pose)

        tf = TransformStamped()

        tf.header.stamp = (
            self.get_clock().now().to_msg()
        )
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

    def _get_camera_rgb(self):
        frame = self.camera.get_frame()

        if frame.is_empty:
            self.get_logger().warning(
                "Camera returned empty frame."
            )
            return None

        rgb_frame = self.camera.get_color_frame()

        if rgb_frame is None or rgb_frame.data is None:
            self.get_logger().error(
                "Camera returned empty RGB frame."
            )
            return None

        rgb = rgb_frame.data

        if not isinstance(rgb, np.ndarray):
            rgb = np.asarray(rgb)

        return rgb

    # ============================================================
    # DINOv3 BOX CHECK
    # ============================================================

    def box_check_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:

        del request

        if not self.box_check_enabled:
            response.success = False
            response.message = "Box Check disabled."
            return response

        if self.processing:
            response.success = False
            response.message = "busy"
            return response

        if self.box_check_estimator is None:
            response.success = False
            response.message = "DINOv3 estimator unavailable."
            return response

        self.processing = True

        try:
            self.get_logger().info(
                "========================================"
            )
            self.get_logger().info(
                "Box Check prediction triggered."
            )

            rgb = self._get_camera_rgb()

            if rgb is None:
                response.success = False
                response.message = "camera failed"
                return response

            if rgb.ndim != 3 or rgb.shape[2] != 3:
                raise RuntimeError(
                    f"Invalid RGB shape: {rgb.shape}"
                )

            self.get_logger().info(
                f"RGB: shape={rgb.shape}, dtype={rgb.dtype}"
            )

            self.get_logger().info(
                "Running DINOv3 Box Check..."
            )

            result = self.box_check_estimator.predict(rgb)

            left = bool(result.get("left", False))
            right = bool(result.get("right", False))

            message = (
                f"left:{str(left).lower()} "
                f"right:{str(right).lower()}"
            )

            result_msg = String()
            result_msg.data = message

            self.box_check_result_pub.publish(
                result_msg
            )

            debug_image = result.get("debug_image")

            if debug_image is None:
                debug_image = rgb

            self._publish_box_debug(
                debug_image
            )

            response.success = True
            response.message = message

            self.get_logger().info(
                f"Box Check result: {message}"
            )

            self.get_logger().info(
                "========================================"
            )

        except Exception as error:
            self.get_logger().error(
                f"Box Check exception: "
                f"{type(error).__name__}: {error}"
            )

            response.success = False
            response.message = (
                f"left:false right:false error:{error}"
            )

        finally:
            self.processing = False

        return response

    # ============================================================
    # ALIGNMENT
    # ============================================================

    def alignment_callback(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:

        del request

        if not self.alignment_enabled:
            response.success = False
            response.message = "Alignment disabled."
            return response

        if self.processing:
            response.success = False
            response.message = "busy"
            return response

        if self.alignment_estimator is None:
            response.success = False
            response.message = "Alignment estimator unavailable."
            return response

        self.processing = True

        try:
            self.get_logger().info(
                "========================================"
            )
            self.get_logger().info(
                "Alignment prediction triggered."
            )

            rgb, depth = self._get_camera_frame()

            if rgb is None or depth is None:
                response.success = False
                response.message = "camera failed"
                return response

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
                dtype=np.float64,
            )

            if camera_matrix.shape != (3, 3):
                raise RuntimeError(
                    "Invalid camera intrinsic matrix shape: "
                    f"{camera_matrix.shape}"
                )

            self.get_logger().info(
                f"RGB   : shape={rgb.shape}, dtype={rgb.dtype}"
            )
            self.get_logger().info(
                f"Depth : shape={depth.shape}, dtype={depth.dtype}"
            )

            self.get_logger().info(
                "Running Alignment..."
            )

            result = self.alignment_estimator.predict(
                image=rgb,
                depth=depth,
                K=camera_matrix,
            )

            input_pose = result.get("input_pose")
            delta = result.get("delta")

            debug_image = result.get("debug_image")

            if debug_image is None:
                debug_image = rgb

            self._publish_alignment_debug(debug_image)

            if input_pose is None:
                response.success = False
                response.message = "Alignment failed: no input pose."
                return response

            if delta is None:
                response.success = False
                response.message = "Alignment failed: no delta."
                return response

            location = np.asarray(
                input_pose.get("location"),
                dtype=np.float64,
            )

            rotation = np.asarray(
                input_pose.get("rotation"),
                dtype=np.float64,
            )

            delta_location = np.asarray(
                delta.get("location"),
                dtype=np.float64,
            )

            delta_rotation = np.asarray(
                delta.get("rotation"),
                dtype=np.float64,
            )

            if (
                location.shape != (3,)
                or rotation.shape != (4,)
                or delta_location.shape != (3,)
                or delta_rotation.shape != (4,)
            ):
                raise RuntimeError(
                    "Invalid Alignment pose/delta shape."
                )

            if not (
                np.all(np.isfinite(location))
                and np.all(np.isfinite(rotation))
                and np.all(np.isfinite(delta_location))
                and np.all(np.isfinite(delta_rotation))
            ):
                raise RuntimeError(
                    "Alignment returned non-finite pose/delta."
                )

            pose_msg = Pose()

            pose_msg.position.x = float(delta_location[0])
            pose_msg.position.y = float(delta_location[1])
            pose_msg.position.z = float(delta_location[2])

            # Alignment quaternion format: [qx, qy, qz, qw].
            pose_msg.orientation.x = float(delta_rotation[0])
            pose_msg.orientation.y = float(delta_rotation[1])
            pose_msg.orientation.z = float(delta_rotation[2])
            pose_msg.orientation.w = float(delta_rotation[3])

            self.alignment_result_pub.publish(
                pose_msg
            )

            tf = TransformStamped()

            tf.header.stamp = (
                self.get_clock().now().to_msg()
            )
            tf.header.frame_id = self.parent_frame
            tf.child_frame_id = self.ripcord_frame

            tf.transform.translation.x = float(location[0])
            tf.transform.translation.y = float(location[1])
            tf.transform.translation.z = float(location[2])

            # Input pose quaternion format: [qx, qy, qz, qw].
            tf.transform.rotation.x = float(rotation[0])
            tf.transform.rotation.y = float(rotation[1])
            tf.transform.rotation.z = float(rotation[2])
            tf.transform.rotation.w = float(rotation[3])

            self.tf_broadcaster.sendTransform(tf)

            angle_deg = delta.get("angle_deg")

            self.get_logger().info(
                "Alignment input pose: "
                f"XYZ=({location[0]:.6f}, "
                f"{location[1]:.6f}, "
                f"{location[2]:.6f}) "
                f"Q=({rotation[0]:.6f}, "
                f"{rotation[1]:.6f}, "
                f"{rotation[2]:.6f}, "
                f"{rotation[3]:.6f})"
            )

            self.get_logger().info(
                "Alignment delta: "
                f"XYZ=({delta_location[0]:.6f}, "
                f"{delta_location[1]:.6f}, "
                f"{delta_location[2]:.6f}) "
                f"Q=({delta_rotation[0]:.6f}, "
                f"{delta_rotation[1]:.6f}, "
                f"{delta_rotation[2]:.6f}, "
                f"{delta_rotation[3]:.6f})"
            )

            if angle_deg is not None:
                self.get_logger().info(
                    f"Alignment delta angle: "
                    f"{float(angle_deg):.6f} deg"
                )

            response.success = True
            response.message = "Alignment prediction successful."

            self.get_logger().info(
                "========================================"
            )

        except Exception as error:
            self.get_logger().error(
                f"Alignment exception: "
                f"{type(error).__name__}: {error}"
            )

            response.success = False
            response.message = str(error)

        finally:
            self.processing = False

        return response

    # ============================================================
    # RACE-6D
    # ============================================================

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
                response.message = (
                    "Invalid RACE-6D result."
                )

                return response

            self.get_logger().info(
                f"RACE-6D returned "
                f"{len(detections)} detections."
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

            response.success = (
                len(published_labels) > 0
            )

            if response.success:
                published_names = [
                    self.label_names[label]
                    for label in sorted(
                        published_labels
                    )
                ]

                response.message = (
                    "Pose estimated for: "
                    + ", ".join(published_names)
                )
            else:
                response.message = (
                    "No configured RACE-6D "
                    "object detected."
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
        failed = self._failed_pose()

        for label, publisher in self.pose_publishers.items():
            publisher.publish(failed)

            self.get_logger().warning(
                f"Published invalid pose for "
                f"label={label} "
                f"({self.label_names[label]})."
            )

    def shutdown(self) -> None:
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


def main():
    import argparse
    import yaml
    import rclpy

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML configuration file.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="RACE-6D/DINOv3/Alignment device.",
    )

    args = parser.parse_args()

    config_path = Path(args.config).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as f:
        cfg = yaml.safe_load(f)

    rclpy.init()

    node = None

    try:
        node = PoseNode(cfg, args)
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        print(
            f"[ERROR] ROS node failed: "
            f"{type(error).__name__}: {error}"
        )

    finally:
        if node is not None:
            node.shutdown()
            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()