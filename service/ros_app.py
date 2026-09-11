#!/usr/bin/env python3
"""Pose6DoF ROS AI Service."""

import argparse
import time
import traceback
from typing import Any, Dict

from ros import PoseNode
from tomo_service import (
    BaseService,
    BaseServiceConfig,
    RosService,
    RosServiceConfig,
)
from tomo_system.file_utils import read_yaml_dir


# ==========================================================
# Arguments
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pose6DoF ROS AI Service")
    parser.add_argument("--port", type=int, default=5002, help="BaseService control port")
    parser.add_argument("--model-version", "--checkpoint", dest="model_version", default=None, help="Optional RACE-6D checkpoint override; default is RACE_6D.MODEL_PATH in config")
    parser.add_argument("--config-dir", type=str, required=True, help="Directory containing camera/service YAML configuration")
    parser.add_argument("--task-name", type=str, default="pose6dof")
    parser.add_argument("--device", type=str, default="")
    return parser.parse_args()


# ==========================================================
# Service
# ==========================================================

class Pose6DoFService(BaseService):
    """Pose6DoF ROS AI Service Controller."""

    def __init__(
        self,
        config: Dict[str, Any],
        args: argparse.Namespace,
    ) -> None:
        self.cfg = config
        self.args = args

        self.port = args.port
        self.model_version = args.model_version or self.cfg["RACE_6D"]["MODEL_PATH"]
        self.device = args.device
        self.service_name = args.task_name or "pose6dof"
        self.service_id = "pose6dof-001"

        # ROS backend
        self.ros_backend = RosService()

        # BaseService config
        base_config = BaseServiceConfig(
            title="Pose6DoF ROS Service",
            version="1.0.0",
            service_name=self.service_name,
            host="127.0.0.1",
            port=self.port,
            model_version=self.model_version,
            num_workers=1,
        )

        super().__init__(
            backend=self.ros_backend,
            config=base_config,
        )

        print("[SERVICE] Pose6DoFService created", flush=True)

    # ======================================================
    # Node Factory
    # ======================================================

    def create_node(self) -> PoseNode:
        print("[SERVICE] Creating PoseNode...", flush=True)
        node = PoseNode(cfg=self.cfg, args=self.args)
        print("[SERVICE] PoseNode created", flush=True)
        return node

    # ======================================================
    # ROS Ready Callback
    # ======================================================

    def on_ready(self) -> None:
        print("[CALLBACK] ROS backend is ready", flush=True)
        print("[CALLBACK] Pose6DoF ROS node is ready", flush=True)

    # ======================================================
    # Initialize
    # ======================================================

    def initialize_service(self) -> None:
        print("[SERVICE] Initializing ROS backend...", flush=True)

        ros_config = RosServiceConfig(
            service_name=self.service_name,
            service_id=self.service_id,
            port=str(self.port),
            node_factory=self.create_node,
            on_ready=self.on_ready,
        )

        self.initialize(ros_config)
        print("[SERVICE] ROS backend initialized", flush=True)

    # ======================================================
    # Start
    # ======================================================

    def start_service(self) -> bool:
        try:
            print("[SERVICE] Starting service...", flush=True)

            # Initialize backend FIRST
            self.initialize_service()

            # BaseService.start() starts:
            # 1. control server
            # 2. RosService
            print("[SERVICE] Calling BaseService.start()...", flush=True)
            self.start()
            print("[SERVICE] BaseService.start() returned", flush=True)

            # Wait ROS ready
            print("[SERVICE] Waiting for ROS ready...", flush=True)
            ready = self.wait_until_ready(timeout_sec=30.0)

            if not ready:
                print("[ERROR] ROS backend failed to become ready", flush=True)
                print(f"[ERROR] status={self.get_status()}", flush=True)
                self.stop()
                return False

            print("[SERVICE] ==============================", flush=True)
            print("[SERVICE] Service initialized and running", flush=True)
            print(f"[SERVICE] Control server: 127.0.0.1:{self.port}", flush=True)
            print("[SERVICE] ROS services:", flush=True)
            print("  /pose6dof/test", flush=True)
            print("[SERVICE] ==============================", flush=True)

            return True

        except Exception as e:
            print(f"[ERROR] Failed to start service: {e}", flush=True)
            traceback.print_exc()

            try:
                if self.running:
                    self.stop()
            except Exception:
                pass

            return False

    # ======================================================
    # Run
    # ======================================================

    def run(self) -> bool:
        try:
            if not self.start_service():
                return False

            print("[SERVICE] Main loop started", flush=True)

            while self.running:
                time.sleep(0.1)

            return True

        except KeyboardInterrupt:
            print("[SERVICE] Shutdown requested", flush=True)
            return True

        except Exception as e:
            print(f"[SERVICE] Runtime error: {e}", flush=True)
            traceback.print_exc()
            return False

        finally:
            self.shutdown()

    # ======================================================
    # Shutdown
    # ======================================================

    def shutdown(self) -> None:
        print("[SERVICE] Shutting down...", flush=True)

        try:
            if self.running:
                self.stop()
        except Exception as e:
            print(f"[SERVICE] Stop error: {e}", flush=True)

        print("[SERVICE] Service stopped", flush=True)


# ==========================================================
# Main
# ==========================================================

def main() -> int:
    args = parse_args()
    cfg = read_yaml_dir(args.config_dir)

    print("======================================", flush=True)
    print("Starting Pose6DoF AI Service", flush=True)
    print(f"Config directory : {args.config_dir}", flush=True)
    print(f"RACE checkpoint  : {args.model_version or cfg["RACE_6D"]["MODEL_PATH"]}", flush=True)
    print(f"Port             : {args.port}", flush=True)
    print(f"Task             : {args.task_name}", flush=True)
    print("======================================", flush=True)

    service = Pose6DoFService(
        config=cfg,
        args=args,
    )

    success = service.run()
    return 0 if success else 1


# ==========================================================
# Entry
# ==========================================================

if __name__ == "__main__":
    raise SystemExit(main())
