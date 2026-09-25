#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from RACE_6D.core import PoseEstimator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

INPUT_WIDTH = 1920
INPUT_HEIGHT = 1080
CROP_X = 240
CROP_WIDTH = 1440
MODEL_WIDTH = 640
MODEL_HEIGHT = 480
AXIS_LENGTH_MM = 50.0

DEFAULT_K = np.array(
    [
        [1070.7897900136718, 0.0, 962.3093872070312],
        [0.0, 1070.5269785615235, 512.0372314453125],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="RACE-6D RGB-D video inference."
    )
    parser.add_argument("rgb", help="Path to RGB MP4.")
    parser.add_argument("depth", help="Path to depth MKV.")
    return parser.parse_args()


def load_config():
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {CONFIG_PATH}")

    return config


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_race6d_config():
    config = load_config()
    race_config = config.get("RACE_6D")

    if not isinstance(race_config, dict):
        raise ValueError(f"RACE_6D section is missing in {CONFIG_PATH}")

    label_names = race_config.get("LABEL_NAMES", {})
    if not isinstance(label_names, dict):
        raise ValueError("RACE_6D.LABEL_NAMES must be a mapping.")

    return {
        "model_path": resolve_project_path(race_config["MODEL_PATH"]),
        "model_config": resolve_project_path(race_config["MODEL_CONFIG"]),
        "score_threshold": float(race_config.get("SCORE_THRESHOLD", 0.25)),
        "max_per_class": int(race_config.get("MAX_PER_CLASS", 1)),
        "max_detections": race_config.get("MAX_DETECTIONS"),
        "class_id": race_config.get("CLASS_ID"),
        "depth_z_max_mm": float(race_config.get("DEPTH_Z_MAX_MM", 2000.0)),
        "invalid_depth_value": int(race_config.get("INVALID_DEPTH_VALUE", 65535)),
        "label_names": {int(k): str(v) for k, v in label_names.items()},
    }


def get_model_intrinsic(K: np.ndarray) -> np.ndarray:
    model_K = np.asarray(K, dtype=np.float32).copy()

    model_K[0, 2] -= CROP_X

    sx = MODEL_WIDTH / CROP_WIDTH
    sy = MODEL_HEIGHT / INPUT_HEIGHT

    model_K[0, 0] *= sx
    model_K[0, 2] *= sx
    model_K[1, 1] *= sy
    model_K[1, 2] *= sy

    return model_K


def preprocess_debug_image(image: np.ndarray) -> np.ndarray:
    cropped = image[:INPUT_HEIGHT, CROP_X:CROP_X + CROP_WIDTH]
    resized = cv2.resize(
        cropped,
        (MODEL_WIDTH, MODEL_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
    return resized


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)

    if q.size != 4:
        raise ValueError(f"Invalid quaternion: {q}")

    qw, qx, qy, qz = q
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        raise ValueError("Quaternion norm is zero.")

    qw, qx, qy, qz = q / norm

    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=np.float64,
    )


def project_point(point: np.ndarray, K: np.ndarray):
    x, y, z = point

    if not np.isfinite(z) or z <= 1e-6:
        return None

    u = K[0, 0] * x / z + K[0, 2]
    v = K[1, 1] * y / z + K[1, 2]

    if not np.isfinite(u) or not np.isfinite(v):
        return None

    return int(round(u)), int(round(v))


def draw_axis(
    image: np.ndarray,
    translation_mm,
    quaternion,
    K: np.ndarray,
    axis_length_mm: float,
) -> bool:
    t = np.asarray(translation_mm, dtype=np.float64).reshape(-1)

    if t.size != 3 or not np.all(np.isfinite(t)):
        return False

    try:
        R = quaternion_to_rotation_matrix(quaternion)
    except Exception:
        return False

    points = np.array(
        [
            t,
            t + R[:, 0] * axis_length_mm,
            t + R[:, 1] * axis_length_mm,
            t + R[:, 2] * axis_length_mm,
        ],
        dtype=np.float64,
    )

    projected = [project_point(point, K) for point in points]

    if projected[0] is None:
        return False

    origin = projected[0]
    h, w = image.shape[:2]

    def valid(point):
        if point is None:
            return False
        x, y = point
        return -w <= x <= 2 * w and -h <= y <= 2 * h

    cv2.circle(
        image,
        origin,
        5,
        (255, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    for name, endpoint, color in (
        ("X", projected[1], (0, 0, 255)),
        ("Y", projected[2], (0, 255, 0)),
        ("Z", projected[3], (255, 0, 0)),
    ):
        if not valid(endpoint):
            continue

        cv2.arrowedLine(
            image,
            origin,
            endpoint,
            color,
            3,
            cv2.LINE_AA,
            tipLength=0.15,
        )

        cv2.putText(
            image,
            name,
            (endpoint[0] + 8, endpoint[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return True


def draw_detection(
    image: np.ndarray,
    detection: dict,
    label_names: dict[int, str],
    K: np.ndarray,
    axis_length_mm: float,
) -> bool:
    translation_mm = detection.get("translation_mm")
    quaternion = detection.get("quat")

    if translation_mm is None or quaternion is None:
        return False

    label = int(detection.get("label", -1))
    name = label_names.get(label, f"class_{label}")

    if not draw_axis(
        image,
        translation_mm,
        quaternion,
        K,
        axis_length_mm,
    ):
        return False

    t = np.asarray(translation_mm, dtype=np.float64).reshape(-1)
    confidence = detection.get("confidence")

    if confidence is None:
        text = f"{name} T=({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f})mm"
    else:
        text = (
            f"{name} {float(confidence):.2f} "
            f"T=({t[0]:.0f}, {t[1]:.0f}, {t[2]:.0f})mm"
        )

    y = 35 + label * 30
    y = min(y, image.shape[0] - 15)

    cv2.putText(
        image,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return True


def open_depth_pipe(path: Path):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray16le",
        "pipe:1",
    ]

    return subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1024 * 1024,
    )


def read_depth_frame(process, width: int, height: int):
    frame_bytes = width * height * 2
    raw = process.stdout.read(frame_bytes)

    if len(raw) != frame_bytes:
        return None

    return np.frombuffer(
        raw,
        dtype="<u2",
    ).reshape(height, width)


def main():
    args = parse_args()

    rgb_path = Path(args.rgb).resolve()
    depth_path = Path(args.depth).resolve()

    if not rgb_path.is_file():
        raise FileNotFoundError(rgb_path)

    if not depth_path.is_file():
        raise FileNotFoundError(depth_path)

    race_config = load_race6d_config()
    model_path = race_config["model_path"]
    model_config = race_config["model_config"]
    label_names = race_config["label_names"]

    if not model_path.is_file():
        raise FileNotFoundError(f"RACE-6D model not found: {model_path}")

    if not model_config.is_file():
        raise FileNotFoundError(f"RACE-6D config not found: {model_config}")

    output_path = Path.cwd() / f"{rgb_path.stem}_race6d.mp4"
    camera_matrix = DEFAULT_K.copy()
    model_intrinsic = get_model_intrinsic(camera_matrix)

    cap = cv2.VideoCapture(str(rgb_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RGB video: {rgb_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width != INPUT_WIDTH or height != INPUT_HEIGHT:
        cap.release()
        raise RuntimeError(
            f"RGB video must be {INPUT_WIDTH}x{INPUT_HEIGHT}, "
            f"got {width}x{height}"
        )

    print("=" * 70)
    print("RACE-6D RGB-D VIDEO INFERENCE")
    print("=" * 70)
    print(f"Project   : {PROJECT_ROOT}")
    print(f"Config    : {CONFIG_PATH}")
    print(f"RGB       : {rgb_path}")
    print(f"Depth     : {depth_path}")
    print(f"Output    : {output_path}")
    print(f"Model     : {model_path}")
    print(f"Model cfg : {model_config}")
    print(f"Input     : {width} x {height}")
    print(f"Output    : {MODEL_WIDTH} x {MODEL_HEIGHT}")
    print(f"FPS       : {fps:.3f}")
    print(f"Frames    : {frame_count}")
    print(f"Score     : {race_config['score_threshold']}")
    print(f"Max/class : {race_config['max_per_class']}")
    print(f"Axis      : {AXIS_LENGTH_MM:.1f} mm")
    print(f"Original K:\n{camera_matrix}")
    print(f"Model K:\n{model_intrinsic}")
    print("=" * 70)

    print("[INFO] Initializing RACE-6D...")

    estimator = PoseEstimator(
        model_path=str(model_path),
        config_path=str(model_config),
        device="cuda" if torch.cuda.is_available() else "cpu",
        score_threshold=race_config["score_threshold"],
        max_per_class=race_config["max_per_class"],
        max_detections=race_config["max_detections"],
        depth_z_max_mm=race_config["depth_z_max_mm"],
        invalid_depth_value=race_config["invalid_depth_value"],
        class_id=race_config["class_id"],
    )

    print("[INFO] RACE-6D ready.")

    depth_process = open_depth_pipe(depth_path)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (MODEL_WIDTH, MODEL_HEIGHT),
    )

    if not writer.isOpened():
        depth_process.kill()
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_path}")

    frame_index = 0
    success_frames = 0
    total_detections = 0

    try:
        while True:
            ok, rgb = cap.read()

            if not ok:
                break

            depth = read_depth_frame(
                depth_process,
                width,
                height,
            )

            if depth is None:
                print(f"\n[INFO] Depth ended at frame {frame_index}.")
                break

            if depth.shape != (height, width):
                raise RuntimeError(
                    f"RGB/depth resolution mismatch: "
                    f"RGB={rgb.shape[:2]}, Depth={depth.shape}"
                )

            try:
                result = estimator.predict(
                    rgb=rgb,
                    depth=depth,
                    intrinsic=camera_matrix,
                )

                detections = result.get("detections", [])
                if not isinstance(detections, list):
                    detections = []

                debug = result.get("debug_image")

                if debug is None:
                    debug = preprocess_debug_image(rgb)
                else:
                    debug = debug.copy()

                if debug.shape[:2] != (MODEL_HEIGHT, MODEL_WIDTH):
                    debug = cv2.resize(
                        debug,
                        (MODEL_WIDTH, MODEL_HEIGHT),
                        interpolation=cv2.INTER_LINEAR,
                    )

                if debug.ndim == 3 and debug.shape[2] == 3:
                    debug = cv2.cvtColor(
                        debug,
                        cv2.COLOR_RGB2BGR,
                    )

                valid_detections = 0

                for detection in detections:
                    if not isinstance(detection, dict):
                        continue

                    if draw_detection(
                        debug,
                        detection,
                        label_names,
                        model_intrinsic,
                        AXIS_LENGTH_MM,
                    ):
                        valid_detections += 1

                if valid_detections:
                    success_frames += 1

                total_detections += valid_detections

                debug_height, debug_width = debug.shape[:2]

                cv2.putText(
                    debug,
                    f"Frame: {frame_index}",
                    (20, debug_height - 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    debug,
                    f"Detections: {valid_detections}",
                    (20, debug_height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            except Exception as error:
                print(
                    f"\n[ERROR] Frame {frame_index}: "
                    f"{type(error).__name__}: {error}"
                )

                debug = preprocess_debug_image(rgb)

                cv2.putText(
                    debug,
                    "RACE-6D ERROR",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            if debug.shape[1] != MODEL_WIDTH or debug.shape[0] != MODEL_HEIGHT:
                debug = cv2.resize(
                    debug,
                    (MODEL_WIDTH, MODEL_HEIGHT),
                    interpolation=cv2.INTER_LINEAR,
                )

            writer.write(debug)
            frame_index += 1

            if frame_index % 10 == 0:
                print(
                    f"\rFrame: {frame_index} | "
                    f"Pose frames: {success_frames} | "
                    f"Detections: {total_detections}",
                    end="",
                    flush=True,
                )

    finally:
        cap.release()
        writer.release()

        if depth_process.poll() is None:
            depth_process.terminate()

        try:
            depth_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            depth_process.kill()

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Frames      : {frame_index}")
    print(f"Pose frames : {success_frames}")
    print(f"Detections  : {total_detections}")
    print(f"Output      : {output_path}")
    print(f"Output size : {MODEL_WIDTH} x {MODEL_HEIGHT}")
    print("=" * 70)


if __name__ == "__main__":
    main()