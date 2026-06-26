#!/usr/bin/env python3
"""
Fault Detection on ROS2 Rosbag
==============================

Performs fault detection on frames extracted from recorded MCAP bag.
Uses ros2 bag play + rclpy subscription (proven approach).

Based on the working mcap_video_recorder.py pattern.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
import cv2
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
from ultralytics import YOLO
import subprocess
import time
import os

# ============================================================================
# CONFIGURATION - MODIFY THIS SECTION FOR YOUR ROSBAG2 AND SETUP
# ============================================================================
ROSBAG_FOLDER = r"/media/viraj/9ab9ab55-5238-4bd4-ab76-54c20d1c2d0f/home/combat-ind-nuc/Viraj/rosbag2_2026_06_25-01_47_36"
IMAGE_TOPIC = "/ace_camera_rail_left/pylon_ros2_camera_node_ace_rail_left/image/compressed"

MODEL_PATH = r"/home/viraj/Drishti_code/Drishti_Detection/Models/fault_detectionv4.2.pt"
OUTPUT_DIR = r"/home/viraj/Drishti_code/Drishti_detection_output/direct_mcap"
CONFIDENCE_THRESHOLD = 0.5
DETERMINISM_LEVEL = 1
CATEGORIZE = True


def setup_determinism(level: int):
    """Apply determinism settings."""
    if level == 0:
        return

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if level >= 2:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if level >= 3:
        os.environ["CUDNN_DETERMINISTIC"] = "1"
        os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"


class FaultDetectionNode(Node):
    """Node that detects faults in camera frames from rosbag."""

    def __init__(self, image_topic: str, model_path: str, output_dir: str):
        super().__init__("fault_detection_node")

        self.image_topic = image_topic
        self.output_dir = Path(output_dir)
        self.bridge = CvBridge()

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.faults_dir = self.run_dir / "faults"
        self.faults_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f"📂 Output directory: {self.run_dir}")

        # Load model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        if model_path.endswith(".pt"):
            self.model.to(self.device)

        self.get_logger().info(f"✓ Model loaded on {self.device}")

        # Determine if topic is compressed
        self.is_compressed = "compressed" in image_topic

        # Set permissive QoS for bag playback
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to image topic
        if self.is_compressed:
            self.sub = self.create_subscription(
                CompressedImage,
                self.image_topic,
                self.image_callback,
                qos_profile
            )
        else:
            self.sub = self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                qos_profile
            )

        self.get_logger().info(f"✓ Subscribed to {self.image_topic}\n")

        # Stats
        self.frame_count = 0
        self.detection_count = 0
        self.frames_with_detections = 0
        self.inference_times = []
        self.class_dirs = {}

    def image_callback(self, msg):
        """Callback to detect faults in image messages."""
        try:
            if self.is_compressed:
                frame = cv2.imdecode(
                    np.frombuffer(msg.data, np.uint8),
                    cv2.IMREAD_COLOR
                )
            else:
                frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if frame is None:
                return

            self.frame_count += 1

            # Run detection
            inf_start = time.time()
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, device=self.device)
            inf_time = (time.time() - inf_start) * 1000
            self.inference_times.append(inf_time)

            # Process detections
            if len(results[0].boxes) > 0:
                num_detections = len(results[0].boxes)
                self.detection_count += num_detections
                self.frames_with_detections += 1

                # Save detected frame
                confidence = float(results[0].boxes.conf.max())
                filename = f"frame_{self.frame_count:06d}_{num_detections}_{confidence:.2f}.jpg"
                faults_path = self.faults_dir / filename
                cv2.imwrite(str(faults_path), frame)

                # Categorize by class if enabled
                if CATEGORIZE:
                    processed_frame = results[0].plot()
                    boxes = results[0].boxes
                    classes = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else []

                    if len(classes) > 0:
                        class_idx = int(classes[0])
                        class_name = results[0].names.get(class_idx, f"class_{class_idx}")

                        if class_name not in self.class_dirs:
                            class_dir = self.run_dir / class_name
                            class_dir.mkdir(parents=True, exist_ok=True)
                            self.class_dirs[class_name] = class_dir

                        class_path = self.class_dirs[class_name] / filename
                        cv2.imwrite(str(class_path), processed_frame)

                if self.frame_count % 100 == 0 or self.frame_count <= 5:
                    self.get_logger().info(
                        f"Frame {self.frame_count:6d} | {num_detections:2d} det | {inf_time:7.2f}ms"
                    )

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def finalize(self):
        """Save metrics and display results."""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("DETECTION COMPLETE")
        self.get_logger().info("="*70)

        self.get_logger().info(f"\n📊 Results:")
        self.get_logger().info(f"  Frames processed: {self.frame_count}")
        self.get_logger().info(f"  Detections found: {self.detection_count}")
        self.get_logger().info(f"  Frames with detections: {self.frames_with_detections}")

        if self.inference_times:
            avg_inf = np.mean(self.inference_times)
            self.get_logger().info(f"\n⏱️  Inference:")
            self.get_logger().info(f"  Average: {avg_inf:.2f} ms/frame")

        # Save metrics
        metrics_file = self.run_dir / "metrics.json"
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "frames_processed": self.frame_count,
            "detections": self.detection_count,
            "frames_with_detections": self.frames_with_detections,
            "avg_inference_ms": float(np.mean(self.inference_times)) if self.inference_times else 0,
            "output_dir": str(self.run_dir),
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.get_logger().info(f"\n✓ Metrics saved: {metrics_file}")
        self.get_logger().info("="*70 + "\n")


def main():
    """Main function."""

    rosbag_folder = Path(ROSBAG_FOLDER)

    if not rosbag_folder.exists():
        print(f"❌ Rosbag folder not found: {ROSBAG_FOLDER}")
        return

    print("\n" + "="*70)
    print("FAULT DETECTION - ROS2 ROSBAG")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"  Rosbag folder: {rosbag_folder.name}")
    print(f"  Image topic: {IMAGE_TOPIC}")
    print(f"  Model: {Path(MODEL_PATH).name}")
    print(f"  Output dir: {OUTPUT_DIR}\n")

    setup_determinism(DETERMINISM_LEVEL)

    # Initialize ROS2
    rclpy.init()
    node = FaultDetectionNode(IMAGE_TOPIC, MODEL_PATH, OUTPUT_DIR)

    print("🎥 Starting bag playback and detection...\n")

    # Play bag in subprocess
    bag_process = subprocess.Popen(
        ["ros2", "bag", "play", str(ROSBAG_FOLDER), "--topics", IMAGE_TOPIC],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    # Spin node to process messages
    print("Processing frames (this may take several minutes)...\n")
    try:
        time.sleep(1)  # Give bag player time to start

        # Keep spinning until bag process actually finishes
        while bag_process.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        # Catch any remaining messages after bag finishes
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        if bag_process.poll() is None:
            bag_process.terminate()
            bag_process.wait()

        node.finalize()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
