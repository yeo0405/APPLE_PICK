#!/usr/bin/env python3
from __future__ import annotations

import json

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from diagnostic_msgs.msg import DiagnosticStatus, KeyValue
from geometry_msgs.msg import Pose, TransformStamped
from rclpy.node import Node
from robot_localization.srv import SetPose
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

# ======================== Mock results: edit here ========================
IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)

POSE_RESULTS = [
    {
        "id": "mock_object_0",
        "position": (0.10, 0.00, 0.50),
        "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
        "score": 0.99,
    },
    {
        "id": "mock_object_1",
        "position": (-0.10, 0.02, 0.60),
        "orientation_xyzw": (0.0, 0.0, 0.7071068, 0.7071068),
        "score": 0.99,
    },
]
BOX_CHECK_RESULTS = {"left_ear": "present", "right_ear": "absent"}
FLAP_CHECK_RESULTS = {"left_ear": "present", "right_ear": "absent"}
ALIGNMENT_RESULT = {
    "position": (0.01, -0.02, 0.03),
    "orientation_xyzw": (0.0, 0.0, 0.0, 1.0),
}
RIPCORD_RESULTS = {"0": "PRESENT", "1": "PRESENT", "2": "ABSENT", "3": "ABSENT"}
RIPCORD_CHECK_RESULTS = {"ripcord": "PRESENT"}
# ========================================================================


class MockPoseNode(Node):
    def __init__(self) -> None:
        super().__init__("pose6dof_mock_ros_node")
        self.bridge = CvBridge()
        self.processing = False
        self.parent_frame = "Head_camera_link"
        self.ripcord_frame = "ripcord"
        self.tf_broadcaster = TransformBroadcaster(self)

        self.pose_pub = self.create_publisher(Detection3DArray, "/pose6dof/result", 10)
        self.debug_pub = self.create_publisher(Image, "/pose6dof/debug_image", 10)
        self.box_check_result_pub = self.create_publisher(
            DiagnosticStatus, "/box_check/result", 10
        )
        self.box_check_debug_pub = self.create_publisher(
            Image, "/box_check/debug_image", 10
        )
        self.flap_check_result_pub = self.create_publisher(
            DiagnosticStatus, "/flap_check/result", 10
        )
        self.flap_check_debug_pub = self.create_publisher(
            Image, "/flap_check/debug_image", 10
        )
        self.alignment_result_pub = self.create_publisher(Pose, "/alignment/result", 10)
        self.alignment_debug_pub = self.create_publisher(
            Image, "/alignment/debug_image", 10
        )
        self.ripcord_result_pub = self.create_publisher(
            DiagnosticStatus, "/ripcord/result", 10
        )
        self.ripcord_debug_pub = self.create_publisher(
            Image, "/ripcord/debug_image", 10
        )
        self.ripcord_check_result_pub = self.create_publisher(
            DiagnosticStatus, "/ripcord_check/result", 10
        )
        self.ripcord_check_debug_pub = self.create_publisher(
            Image, "/ripcord_check/debug_image", 10
        )

        self.mock_services = [
            self.create_service(Trigger, "/pose6dof/predict", self.pose_callback),
            self.create_service(Trigger, "/box_check/predict", self.box_check_callback),
            self.create_service(
                Trigger, "/flap_check/predict", self.flap_check_callback
            ),
            # alignment: robot_localization/srv/SetPose (request carries a pose)
            self.create_service(SetPose, "/alignment/predict", self.alignment_callback),
            self.create_service(Trigger, "/ripcord/predict", self.ripcord_callback),
            self.create_service(
                Trigger, "/ripcord_check/predict", self.ripcord_check_callback
            ),
        ]
        self.get_logger().info(
            "Mock ROS node started; no models or cameras are initialized."
        )
        for name in (
            "/pose6dof/predict",
            "/box_check/predict",
            "/flap_check/predict",
            "/alignment/predict (robot_localization/srv/SetPose)",
            "/ripcord/predict",
            "/ripcord_check/predict",
        ):
            self.get_logger().info(f"Service: {name}")
        self.get_logger().info(
            "Mock results and debug images will publish when triggered."
        )

    def _debug_image(self, title: str, lines: list[str]) -> np.ndarray:
        image = np.full((480, 640, 3), 32, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (620, 460), (100, 100, 100), 2)
        cv2.putText(
            image,
            title,
            (40, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        for i, line in enumerate(lines):
            cv2.putText(
                image,
                line,
                (40, 120 + i * 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (230, 230, 230),
                2,
                cv2.LINE_AA,
            )
        return image

    def _publish_image(self, publisher, title: str, lines: list[str]) -> None:
        msg = self.bridge.cv2_to_imgmsg(
            self._debug_image(title, lines), encoding="bgr8"
        )
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        publisher.publish(msg)

    @staticmethod
    def _status(name: str, values: dict[str, str]) -> DiagnosticStatus:
        msg = DiagnosticStatus()
        msg.name = name
        msg.level = DiagnosticStatus.OK
        msg.message = "success"
        msg.values = [KeyValue(key=key, value=value) for key, value in values.items()]
        return msg

    @staticmethod
    def _pose(
        position: tuple[float, float, float],
        orientation_xyzw: tuple[float, float, float, float] = IDENTITY_QUAT,
    ) -> Pose:
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = position
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = orientation_xyzw
        return pose

    def _publish_mock_tf(self, child_frame: str, pose: Pose) -> None:
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = self.parent_frame
        tf.child_frame_id = child_frame
        tf.transform.translation.x = pose.position.x
        tf.transform.translation.y = pose.position.y
        tf.transform.translation.z = pose.position.z
        tf.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform(tf)

    def _begin(self, response: Trigger.Response) -> bool:
        if self.processing:
            response.success = False
            response.message = "busy"
            return False
        self.processing = True
        return True

    def pose_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        del request
        if not self._begin(response):
            return response
        try:
            array = Detection3DArray()
            array.header.stamp = self.get_clock().now().to_msg()
            array.header.frame_id = self.parent_frame
            debug_lines = []
            for item in POSE_RESULTS:
                orientation = item.get("orientation_xyzw", IDENTITY_QUAT)
                pose = self._pose(item["position"], orientation)
                detection = Detection3D()
                detection.header = array.header
                detection.id = item["id"]
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = item["id"]
                hypothesis.hypothesis.score = float(item["score"])
                hypothesis.pose.pose = pose
                detection.results.append(hypothesis)
                array.detections.append(detection)
                self._publish_mock_tf(item["id"], pose)
                debug_lines.append(f'{item["id"]}: {tuple(item["position"])}')
                debug_lines.append(f"  q: {tuple(round(v, 3) for v in orientation)}")
            self.pose_pub.publish(array)
            self._publish_image(
                self.debug_pub, "MOCK POSE6DOF", debug_lines or ["No detections"]
            )
            response.success = True
            response.message = "Mock pose result published."
        except Exception as error:
            response.success = False
            response.message = str(error)
        finally:
            self.processing = False
        return response

    def _ear_callback(
        self,
        request,
        response,
        name: str,
        values: dict[str, str],
        result_pub,
        debug_pub,
    ) -> Trigger.Response:
        del request
        if not self._begin(response):
            return response
        try:
            result_pub.publish(self._status(name, values))
            self._publish_image(
                debug_pub,
                f"MOCK {name.upper()}",
                [f"{key}: {value}" for key, value in values.items()],
            )
            response.success = True
            response.message = json.dumps(values, separators=(",", ":"))
        except Exception as error:
            response.success = False
            response.message = str(error)
        finally:
            self.processing = False
        return response

    def box_check_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        return self._ear_callback(
            request,
            response,
            "box_check",
            BOX_CHECK_RESULTS,
            self.box_check_result_pub,
            self.box_check_debug_pub,
        )

    def flap_check_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        return self._ear_callback(
            request,
            response,
            "flap_check",
            FLAP_CHECK_RESULTS,
            self.flap_check_result_pub,
            self.flap_check_debug_pub,
        )

    def alignment_callback(
        self, request: SetPose.Request, response: SetPose.Response
    ) -> SetPose.Response:
        # SetPose has an empty response, so busy/errors can only be logged.
        if self.processing:
            self.get_logger().warning("alignment request ignored: busy")
            return response
        self.processing = True
        try:
            # Pose sent by the client: Stamped -> WithCovariance -> Pose
            req_stamped = request.pose
            req_pose = req_stamped.pose.pose
            req_frame = req_stamped.header.frame_id or "(empty)"
            rp, rq = req_pose.position, req_pose.orientation
            self.get_logger().info(
                f"alignment request pose in '{req_frame}': "
                f"pos=({rp.x:.3f}, {rp.y:.3f}, {rp.z:.3f}) "
                f"quat=({rq.x:.3f}, {rq.y:.3f}, {rq.z:.3f}, {rq.w:.3f})"
            )

            # Published result still comes from ALIGNMENT_RESULT (mock)
            pose = self._pose(
                ALIGNMENT_RESULT["position"], ALIGNMENT_RESULT["orientation_xyzw"]
            )
            self.alignment_result_pub.publish(pose)
            self._publish_mock_tf(self.ripcord_frame, pose)
            self._publish_image(
                self.alignment_debug_pub,
                "MOCK ALIGNMENT",
                [
                    f'position: {ALIGNMENT_RESULT["position"]}',
                    f'orientation xyzw: {ALIGNMENT_RESULT["orientation_xyzw"]}',
                    f"req frame: {req_frame}",
                    f"req pos: ({rp.x:.3f}, {rp.y:.3f}, {rp.z:.3f})",
                    f"req q: ({rq.x:.3f}, {rq.y:.3f}, {rq.z:.3f}, {rq.w:.3f})",
                ],
            )
        except Exception as error:
            self.get_logger().error(f"alignment callback failed: {error}")
        finally:
            self.processing = False
        return response

    def ripcord_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        return self._ear_callback(
            request,
            response,
            "ripcord",
            RIPCORD_RESULTS,
            self.ripcord_result_pub,
            self.ripcord_debug_pub,
        )

    def ripcord_check_callback(
        self, request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        return self._ear_callback(
            request,
            response,
            "ripcord_check",
            RIPCORD_CHECK_RESULTS,
            self.ripcord_check_result_pub,
            self.ripcord_check_debug_pub,
        )


def main() -> None:
    rclpy.init()
    node = MockPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
