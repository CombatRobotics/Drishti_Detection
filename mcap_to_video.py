#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from pathlib import Path
import subprocess
import time

ROSBAG_FOLDER = "/media/viraj/9ab9ab55-5238-4bd4-ab76-54c20d1c2d0f/home/combat-ind-nuc/Viraj/rosbag2_2026_06_25-01_47_36"
IMAGE_TOPIC = "/ace_camera_rail_left/pylon_ros2_camera_node_ace_rail_left/image/compressed"

class FrameCapture(Node):
    def __init__(self, topic, output_path):
        super().__init__("frame_capture")
        self.output_path = output_path
        self.writer = None
        self.frame_count = 0

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.sub = self.create_subscription(CompressedImage, topic, self.callback, qos)

    def callback(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        # Create writer on first frame
        if self.writer is None:
            h, w = frame.shape[:2]
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                30.0,
                (w, h)
            )
            if not self.writer.isOpened():
                print("❌ Failed to create video writer")
                return
            print(f"✓ Writing to: {self.output_path}")

        # Write frame immediately (no RAM buildup!)
        self.writer.write(frame)
        self.frame_count += 1

        if self.frame_count % 1000 == 0:
            print(f"  {self.frame_count} frames written...")

# Determine output path
output_path = Path(ROSBAG_FOLDER) / "output.avi"

print("\n" + "="*70)
print("MCAP TO VIDEO CONVERTER")
print("="*70)
print(f"Rosbag folder: {Path(ROSBAG_FOLDER).name}")
print(f"Image topic: {IMAGE_TOPIC}")
print(f"Output: {output_path}\n")

rclpy.init()
node = FrameCapture(IMAGE_TOPIC, output_path)

print("Starting bag playback and frame extraction...\n")
proc = subprocess.Popen(
    ["ros2", "bag", "play", ROSBAG_FOLDER, "--topics", IMAGE_TOPIC],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL
)

time.sleep(1)

# Keep spinning until bag finishes
while proc.poll() is None:
    rclpy.spin_once(node, timeout_sec=0.1)

# Catch any remaining messages
for _ in range(10):
    rclpy.spin_once(node, timeout_sec=0.1)

# Cleanup
if node.writer:
    node.writer.release()

print(f"\n" + "="*70)
print(f"✓ COMPLETE")
print("="*70)
print(f"Frames written: {node.frame_count}")
print(f"Output: {output_path}\n")

rclpy.shutdown()
