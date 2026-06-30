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
import yaml
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
ROSBAG_FOLDER = r"/media/viraj/e84db5a4-80aa-41f1-a173-de4e78ab820d1/home/cri-pc-0/june-july-dhristi-bags/day1_20260630_024350"
IMAGE_TOPIC = "/ace_camera_rail_left/pylon_ros2_camera_node_ace_rail_left/image/compressed"

MODEL_PATH = r"/home/viraj/Drishti_code/Drishti_Detection/Models/fault_detectionv4.2.pt"
OUTPUT_DIR = r"/media/viraj/e84db5a4-80aa-41f1-a173-de4e78ab820d1/home/cri-pc-0/june-july-dhristi-bags/day1_20260630_024350/left_data"
CONFIDENCE_THRESHOLD = 0.35
DETERMINISM_LEVEL = 1
CATEGORIZE = True


def get_expected_frame_count(rosbag_folder: str, image_topic: str) -> int:
    """Read rosbag metadata to get expected frame count for the image topic."""
    metadata_path = Path(rosbag_folder) / "metadata.yaml"
    if not metadata_path.exists():
        print(f"⚠️  metadata.yaml not found: {metadata_path}")
        return None

    try:
        with open(metadata_path, 'r') as f:
            metadata = yaml.safe_load(f)

        topics = metadata.get('rosbag2_bagfile_information', {}).get('topics_with_message_count', [])
        for topic_info in topics:
            topic_name = topic_info.get('topic_metadata', {}).get('name')
            message_count = topic_info.get('message_count')
            if topic_name == image_topic:
                return message_count

        print(f"❌ Topic {image_topic} not found in metadata.yaml")
        return None
    except Exception as e:
        print(f"❌ Error reading metadata.yaml: {e}")
        return None


def check_multiple_publishers(image_topic: str) -> int:
    """Check how many publishers exist on the image topic."""
    try:
        result = subprocess.run(
            ["ros2", "topic", "info", image_topic],
            capture_output=True,
            text=True,
            timeout=5
        )
        # Count "Publisher" lines in output
        publisher_count = result.stdout.count("Publisher")
        return publisher_count
    except Exception as e:
        print(f"⚠️  Could not check publishers: {e}")
        return None


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
        self.timestamps = []
        self.class_dirs = {}
        self.expected_frames = None
        self.current_timestamp_sec = 0
        self.current_timestamp_nsec = 0

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

            # Store timestamp for FPS calculation
            timestamp_sec = msg.header.stamp.sec
            timestamp_nsec = msg.header.stamp.nanosec
            timestamp_float = timestamp_sec + timestamp_nsec / 1e9
            self.timestamps.append(timestamp_float)

            # Store for filename generation
            self.current_timestamp_sec = timestamp_sec
            self.current_timestamp_nsec = timestamp_nsec

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

                # Generate filename from timestamp
                dt = datetime.fromtimestamp(self.current_timestamp_sec)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H-%M-%S")
                microsec = self.current_timestamp_nsec // 1000  # nanosec to microsec
                filename = f"frame_{date_str}_{time_str}.{microsec:06d}_{num_detections}.jpg"
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

    def finalize(self, total_elapsed_time, device):
        """Save metrics and display results."""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("DETECTION COMPLETE")
        self.get_logger().info("="*70)

        self.get_logger().info(f"\n📊 Results:")
        self.get_logger().info(f"  Frames processed: {self.frame_count}")
        self.get_logger().info(f"  Detections found: {self.detection_count}")
        self.get_logger().info(f"  Frames with detections: {self.frames_with_detections}")

        # Validate frame count matches expected
        if self.expected_frames:
            if self.frame_count != self.expected_frames:
                self.get_logger().error(f"\n❌ FRAME COUNT MISMATCH!")
                self.get_logger().error(f"   Expected: {self.expected_frames}")
                self.get_logger().error(f"   Actual: {self.frame_count}")
                self.get_logger().error(f"   Difference: {self.frame_count - self.expected_frames}")
                self.get_logger().error(f"   Possible cause: Multiple publishers on {self.image_topic}")
                self.get_logger().error(f"   Check if another bag/node is publishing on the same topic!")
            else:
                self.get_logger().info(f"\n✅ FRAME COUNT VALIDATED: {self.frame_count} frames (matches metadata)")

        # Calculate actual FPS from timestamps
        calculated_fps = 30.0
        if len(self.timestamps) > 1:
            time_diffs = np.diff(np.array(self.timestamps))
            mean_diff_sec = np.mean(time_diffs)
            if mean_diff_sec > 0:
                calculated_fps = 1.0 / mean_diff_sec

        if self.inference_times:
            avg_inf = np.mean(self.inference_times)
            self.get_logger().info(f"\n⏱️  Inference:")
            self.get_logger().info(f"  Average: {avg_inf:.2f} ms/frame")
            self.get_logger().info(f"  Total: {sum(self.inference_times)/1000:.2f}s")
            self.get_logger().info(f"\n📊 FPS:")
            self.get_logger().info(f"  Calculated from timestamps: {calculated_fps:.2f} fps")

        # Save comprehensive metrics
        metrics_file = self.run_dir / "metrics.json"

        total_inference_seconds = sum(self.inference_times) / 1000 if self.inference_times else 0
        processing_overhead = total_elapsed_time - total_inference_seconds
        overhead_percent = (processing_overhead / total_elapsed_time * 100) if total_elapsed_time > 0 else 0

        metrics = {
            "timestamp": datetime.now().isoformat(),
            "rosbag_folder": ROSBAG_FOLDER,
            "image_topic": IMAGE_TOPIC,
            "detection": {
                "frames_processed": self.frame_count,
                "detections_found": self.detection_count,
                "frames_with_detections": self.frames_with_detections,
            },
            "inference": {
                "avg_ms": float(np.mean(self.inference_times)) if self.inference_times else 0,
                "min_ms": float(np.min(self.inference_times)) if self.inference_times else 0,
                "max_ms": float(np.max(self.inference_times)) if self.inference_times else 0,
                "total_seconds": total_inference_seconds,
            },
            "fps": {
                "calculated_from_timestamps": calculated_fps,
            },
            "execution": {
                "total_elapsed_seconds": total_elapsed_time,
                "processing_overhead_seconds": processing_overhead,
                "processing_overhead_percent": round(overhead_percent, 2),
                "device": device,
            },
            "model": {
                "path": MODEL_PATH,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
            },
            "system": {
                "platform": __import__("platform").platform(),
                "python_version": __import__("sys").version.split()[0],
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
            },
            "determinism": {
                "level": DETERMINISM_LEVEL,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
            "output_directory": str(self.run_dir),
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        self.get_logger().info(f"\n✓ Metrics saved: {metrics_file}")
        self.get_logger().info("="*70 + "\n")


def main():
    """Main function."""

    total_start_time = time.time()

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

    # Get expected frame count from metadata
    expected_frames = get_expected_frame_count(ROSBAG_FOLDER, IMAGE_TOPIC)
    if expected_frames:
        print(f"📊 Expected frames from rosbag metadata: {expected_frames}\n")
    else:
        print("⚠️  Could not read expected frame count from metadata\n")

    # Check for multiple publishers
    print("🔍 Checking for multiple publishers on image topic...")
    time.sleep(0.5)
    pub_count = check_multiple_publishers(IMAGE_TOPIC)
    if pub_count and pub_count > 1:
        print(f"❌ ERROR: {pub_count} publishers detected on {IMAGE_TOPIC}")
        print("   This will cause frame mixing! Please ensure no other bags/nodes are publishing.")
        return

    setup_determinism(DETERMINISM_LEVEL)

    # Initialize ROS2
    rclpy.init()
    node = FaultDetectionNode(IMAGE_TOPIC, MODEL_PATH, OUTPUT_DIR)
    node.expected_frames = expected_frames

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
        total_elapsed_time = time.time() - total_start_time
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if bag_process.poll() is None:
            bag_process.terminate()
            bag_process.wait()

        node.finalize(total_elapsed_time, device)
        rclpy.shutdown()


if __name__ == "__main__":
    main()
