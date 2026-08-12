#!/usr/bin/env python3
"""
YOLO Segmentation on ROS2 Rosbag
=================================

Performs semantic/instance segmentation on frames extracted from MCAP bag.
Uses ros2 bag play + rclpy subscription (proven approach).

Saves segmented frames with mask overlays and comprehensive metrics.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Float32
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
ROSBAG_FOLDER = r"/media/viraj/e84db5a4-80aa-41f1-a173-de4e78ab820d1/home/cri-pc-0/june-july-dhristi-bags/odisha-day-2/day2_p1_pt193_dhristi_plinth_cameras_20260812_155158"
IMAGE_TOPIC = "/ace_camera_plinth_right/pylon_ros2_camera_node_ace_plinth_right/image/compressed"
ENCODER_TOPIC = "/right_encoder_mm"  # Set to encoder topic or None to skip

MODEL_PATH = r"/home/viraj/Drishti_code/Drishti_Detection/Models/ERC_Odissa.pt"  # YOLO segmentation model
OUTPUT_DIR = r"/media/viraj/e84db5a4-80aa-41f1-a173-de4e78ab820d1/home/cri-pc-0/june-july-dhristi-bags/odisha-day-2/day2_p1_pt193_dhristi_plinth_cameras_20260812_155158/ERC_rightside"
CONFIDENCE_THRESHOLD = 0.5
DETERMINISM_LEVEL = 1


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


class SegmentationNode(Node):
    """Node that performs segmentation on camera frames from rosbag."""

    def __init__(self, image_topic: str, model_path: str, output_dir: str):
        super().__init__("segmentation_node")

        self.image_topic = image_topic
        self.output_dir = Path(output_dir)
        self.bridge = CvBridge()

        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # Per-class output directories and all_classes cumulative folder
        self.class_dirs = {}  # {class_id: Path}
        self.all_classes_dir = self.run_dir / "all_classes"
        self.all_classes_dir.mkdir(parents=True, exist_ok=True)

        self.get_logger().info(f"📂 Output directory: {self.run_dir}")

        # Load model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)
        if model_path.endswith(".pt"):
            self.model.to(self.device)

        self.get_logger().info(f"✓ Model loaded on {self.device}")

        # Class color mapping (unique color per class)
        self.class_colors = self._generate_class_colors()

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

        # Subscribe to encoder topic if provided
        self.encoder_topic = ENCODER_TOPIC
        self.latest_encoder_value = None
        self.encoder_sub = None
        if self.encoder_topic:
            try:
                self.encoder_sub = self.create_subscription(
                    Float32,
                    self.encoder_topic,
                    self.encoder_callback,
                    qos_profile
                )
                self.get_logger().info(f"✓ Subscribed to encoder topic: {self.encoder_topic}\n")
            except Exception as e:
                self.get_logger().warning(f"⚠️  Could not subscribe to encoder topic: {e}\n")

        # Stats
        self.frame_count = 0
        self.segmented_frames = 0
        self.inference_times = []
        self.timestamps = []
        self.expected_frames = None
        self.current_timestamp_sec = 0
        self.current_timestamp_nanosec = 0
        self.segmentation_data = []

    def _generate_class_colors(self) -> dict:
        """Generate distinct colors for each class (up to 10 classes)."""
        colors = {
            0: (0, 255, 0),      # Green
            1: (0, 0, 255),      # Red
            2: (255, 0, 0),      # Blue
            3: (255, 255, 0),    # Cyan
            4: (255, 0, 255),    # Magenta
            5: (0, 255, 255),    # Yellow
            6: (128, 0, 255),    # Purple
            7: (255, 128, 0),    # Orange
            8: (128, 255, 0),    # Lime
            9: (0, 128, 255),    # Sky Blue
        }
        return colors

    def get_or_create_class_dir(self, class_id: int) -> Path:
        """Get or create directory for a specific class."""
        if class_id not in self.class_dirs:
            class_dir = self.run_dir / f"class_{class_id}"
            class_dir.mkdir(parents=True, exist_ok=True)
            self.class_dirs[class_id] = class_dir
        return self.class_dirs[class_id]

    def encoder_callback(self, msg):
        """Callback to store latest encoder value."""
        try:
            self.latest_encoder_value = float(msg.data)
            if self.frame_count % 1000 == 0:
                self.get_logger().info(f"🔧 Encoder update: {self.latest_encoder_value}")
        except Exception as e:
            self.get_logger().error(f"Error processing encoder data: {e}")

    def image_callback(self, msg):
        """Callback to perform segmentation on image messages."""
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
            timestamp_nanosec = msg.header.stamp.nanosec
            timestamp_float = timestamp_sec + timestamp_nanosec / 1e9
            self.timestamps.append(timestamp_float)

            # Store for filename generation
            self.current_timestamp_sec = timestamp_sec
            self.current_timestamp_nanosec = timestamp_nanosec

            # Run segmentation
            inf_start = time.time()
            results = self.model(frame, conf=CONFIDENCE_THRESHOLD, device=self.device)
            inf_time = (time.time() - inf_start) * 1000
            self.inference_times.append(inf_time)

            # Process segmentation results
            if results[0].masks is not None and len(results[0].masks) > 0:
                self.segmented_frames += 1

                # Generate filename from timestamp
                dt = datetime.fromtimestamp(self.current_timestamp_sec)
                date_str = dt.strftime("%Y-%m-%d")
                time_str = dt.strftime("%H-%M-%S")
                microsec = self.current_timestamp_nanosec // 1000
                filename = f"frame_{date_str}_{time_str}.{microsec:06d}_{len(results[0].masks)}.jpg"

                frame_h, frame_w = frame.shape[:2]
                detected_classes = set()

                # Debug: Check results structure
                if self.frame_count <= 5:
                    self.get_logger().info(f"🔍 DEBUG Frame {self.frame_count}: masks.data shape = {results[0].masks.data.shape}")
                    if hasattr(results[0], 'cls') and results[0].cls is not None:
                        self.get_logger().info(f"   cls attribute exists, shape = {results[0].cls.shape}")
                    else:
                        self.get_logger().info(f"   cls attribute does NOT exist")

                # Save original frame to all_classes folder (no mask overlay)
                try:
                    all_classes_path = self.all_classes_dir / filename
                    cv2.imwrite(str(all_classes_path), frame)
                except Exception as e:
                    self.get_logger().error(f"❌ Failed to save frame to all_classes: {e}")

                # Process each mask and save to its class folder
                for i, mask in enumerate(results[0].masks.data):
                    try:
                        # Try to extract class ID (may not exist for pure segmentation models)
                        if hasattr(results[0], 'cls') and results[0].cls is not None:
                            class_id = int(results[0].cls[i].cpu().numpy())
                        else:
                            # Fallback: use a single default class if model doesn't output classes
                            class_id = 0
                            if i == 0:  # Log only once per frame
                                self.get_logger().warning("⚠️  Model does not output class information, using class 0 for all masks")
                    except Exception as e:
                        self.get_logger().warning(f"⚠️  Could not extract class ID for mask {i}: {e}, using class 0")
                        class_id = 0

                    detected_classes.add(class_id)

                    try:
                        # Resize mask to original frame size
                        mask_np = mask.cpu().numpy().astype(np.float32)
                        mask_resized = cv2.resize(mask_np, (frame_w, frame_h), interpolation=cv2.INTER_LINEAR)
                        mask_binary = (mask_resized > 0.5).astype(np.uint8)

                        # Create overlay with class-specific color
                        color = self.class_colors.get(class_id, (128, 128, 128))
                        overlay = np.zeros_like(frame, dtype=np.uint8)
                        overlay[mask_binary == 1] = color

                        # Blend overlay with original frame
                        frame_with_mask = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

                        # Save to class-specific folder
                        class_dir = self.get_or_create_class_dir(class_id)
                        class_path = class_dir / filename
                        success = cv2.imwrite(str(class_path), frame_with_mask)
                        if self.frame_count <= 5 and i == 0:  # Debug log on first frame
                            self.get_logger().info(f"   Saving class {class_id} to: {class_path}")
                        if not success:
                            self.get_logger().error(f"❌ Failed to write image to {class_path}")
                    except Exception as e:
                        self.get_logger().error(f"❌ Error processing mask {i} for class {class_id}: {e}")

                # Record segmentation data with detected classes
                seg_entry = {
                    "frame_number": self.frame_count,
                    "frame_timestamp_sec": self.current_timestamp_sec,
                    "frame_timestamp_nanosec": self.current_timestamp_nanosec,
                    "frame_timestamp_ist": dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    "frame_filename": filename,
                    "num_masks": len(results[0].masks),
                    "detected_classes": sorted(list(detected_classes)),
                    "encoder_value": self.latest_encoder_value,
                    "inference_time_ms": float(inf_time)
                }
                self.segmentation_data.append(seg_entry)

                if self.frame_count % 100 == 0 or self.frame_count <= 5:
                    self.get_logger().info(
                        f"Frame {self.frame_count:6d} | {len(results[0].masks):2d} masks (classes: {sorted(list(detected_classes))}) | {inf_time:7.2f}ms"
                    )

        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")

    def finalize(self, total_elapsed_time, device):
        """Save metrics and display results."""
        self.get_logger().info("\n" + "="*70)
        self.get_logger().info("SEGMENTATION COMPLETE")
        self.get_logger().info("="*70)

        self.get_logger().info(f"\n📊 Results:")
        self.get_logger().info(f"  Frames processed: {self.frame_count}")
        self.get_logger().info(f"  Frames with segmentation: {self.segmented_frames}")
        self.get_logger().info(f"\n📁 Output structure:")
        self.get_logger().info(f"  ├─ all_classes/ ({len(list(self.all_classes_dir.glob('*.jpg')))} frames, no masks)")
        for class_id in sorted(self.class_dirs.keys()):
            class_files = list(self.class_dirs[class_id].glob('*.jpg'))
            color = self.class_colors.get(class_id, (128, 128, 128))
            self.get_logger().info(f"  ├─ class_{class_id}/ ({len(class_files)} frames, color BGR{color})")
        self.get_logger().info(f"  ├─ segmentation_with_encoder.json")
        self.get_logger().info(f"  └─ metrics.json")

        # Validate frame count matches expected
        if self.expected_frames:
            if self.frame_count != self.expected_frames:
                self.get_logger().error(f"\n❌ FRAME COUNT MISMATCH!")
                self.get_logger().error(f"   Expected: {self.expected_frames}")
                self.get_logger().error(f"   Actual: {self.frame_count}")
                self.get_logger().error(f"   Difference: {self.frame_count - self.expected_frames}")
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
            "encoder_topic": ENCODER_TOPIC,
            "segmentation": {
                "frames_processed": self.frame_count,
                "frames_with_masks": self.segmented_frames,
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
                "type": "segmentation",
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

        # Save segmentation data with encoder values
        if self.segmentation_data:
            seg_json_file = self.run_dir / "segmentation_with_encoder.json"
            seg_output = {
                "timestamp": datetime.now().isoformat(),
                "rosbag_folder": ROSBAG_FOLDER,
                "image_topic": IMAGE_TOPIC,
                "encoder_topic": ENCODER_TOPIC,
                "total_segmented_frames": len(self.segmentation_data),
                "class_color_mapping": {
                    str(cls_id): {"bgr": color, "description": f"Class {cls_id}"}
                    for cls_id, color in self.class_colors.items()
                },
                "output_structure": {
                    "all_classes": "Original frames with no mask overlay",
                    "class_N": "Frames with segmentation masks for class N (color-coded)"
                },
                "segmentation_data": self.segmentation_data
            }

            with open(seg_json_file, 'w') as f:
                json.dump(seg_output, f, indent=2)

            self.get_logger().info(f"✓ Segmentation data saved: {seg_json_file}")

        self.get_logger().info("="*70 + "\n")


def main():
    """Main function."""

    total_start_time = time.time()

    rosbag_folder = Path(ROSBAG_FOLDER)

    if not rosbag_folder.exists():
        print(f"❌ Rosbag folder not found: {ROSBAG_FOLDER}")
        return

    print("\n" + "="*70)
    print("SEGMENTATION - ROS2 ROSBAG")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"  Rosbag folder: {rosbag_folder.name}")
    print(f"  Image topic: {IMAGE_TOPIC}")
    if ENCODER_TOPIC:
        print(f"  Encoder topic: {ENCODER_TOPIC}")
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
    node = SegmentationNode(IMAGE_TOPIC, MODEL_PATH, OUTPUT_DIR)
    node.expected_frames = expected_frames

    print("🎥 Starting bag playback and segmentation...\n")

    # Build topic list for bag playback
    topics_to_play = [IMAGE_TOPIC]
    if ENCODER_TOPIC:
        topics_to_play.append(ENCODER_TOPIC)

    # Play bag in subprocess
    bag_cmd = ["ros2", "bag", "play", str(ROSBAG_FOLDER), "--topics"] + topics_to_play
    bag_process = subprocess.Popen(
        bag_cmd,
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
