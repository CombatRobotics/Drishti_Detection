#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import subprocess
import time

ROSBAG_FOLDER = "/media/viraj/9ab9ab55-5238-4bd4-ab76-54c20d1c2d0f/home/combat-ind-nuc/Viraj/rosbag2_2026_06_25-01_47_36"
IMAGE_TOPIC = "/ace_camera_rail_right/pylon_ros2_camera_node_ace_rail_right/image/compressed"

class FrameCapture(Node):
    def __init__(self, topic, output_path=None, timestamps_only=False):
        super().__init__("frame_capture")
        self.output_path = output_path
        self.timestamps_only = timestamps_only
        self.writer = None
        self.frame_count = 0
        self.timestamps = []

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.sub = self.create_subscription(CompressedImage, topic, self.callback, qos)

    def callback(self, msg):
        frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        self.frame_count += 1

        # Store timestamp
        timestamp_sec = msg.header.stamp.sec
        timestamp_nsec = msg.header.stamp.nanosec
        timestamp_float = timestamp_sec + timestamp_nsec / 1e9

        self.timestamps.append({
            "frame_number": self.frame_count,
            "timestamp_sec": timestamp_sec,
            "timestamp_nanosec": timestamp_nsec,
            "timestamp_float": timestamp_float,
        })

        # If timestamps_only, skip writing video
        if self.timestamps_only:
            if self.frame_count % 1000 == 0:
                print(f"  {self.frame_count} frames processed...")
            return

        # Create writer on first frame
        if self.writer is None:
            h, w = frame.shape[:2]

            # Calculate actual FPS from timestamps (not hardcoded)
            if len(self.timestamps) > 1:
                time_diff = self.timestamps[-1]['timestamp_float'] - self.timestamps[-2]['timestamp_float']
                calculated_fps = 1.0 / time_diff if time_diff > 0 else 30.0
            else:
                calculated_fps = 30.0

            self.writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*"MJPG"),
                calculated_fps,
                (w, h)
            )
            if not self.writer.isOpened():
                print("❌ Failed to create video writer")
                return
            print(f"✓ Writing to: {self.output_path} @ {calculated_fps:.2f} fps")

        # Write frame immediately (no RAM buildup!)
        self.writer.write(frame)

        if self.frame_count % 1000 == 0:
            print(f"  {self.frame_count} frames written...")

def main():
    parser = argparse.ArgumentParser(description="Extract frames from MCAP bag")
    parser.add_argument("--timestamps", action="store_true",
                       help="Extract ONLY timestamps (no video file)")
    args = parser.parse_args()

    # Determine output path
    output_video = Path(ROSBAG_FOLDER) / "output.avi"
    output_timestamps = Path(ROSBAG_FOLDER) / "frame_timestamps.json"

    print("\n" + "="*70)
    if args.timestamps:
        print("EXTRACTING FRAME TIMESTAMPS ONLY")
    else:
        print("MCAP TO VIDEO CONVERTER")
    print("="*70)
    print(f"Rosbag folder: {Path(ROSBAG_FOLDER).name}")
    print(f"Image topic: {IMAGE_TOPIC}")
    if args.timestamps:
        print(f"Output: {output_timestamps}\n")
    else:
        print(f"Output: {output_video}\n")

    rclpy.init()
    node = FrameCapture(IMAGE_TOPIC, output_video if not args.timestamps else None, timestamps_only=args.timestamps)

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

    # Save timestamps to JSON
    if args.timestamps:
        with open(output_timestamps, 'w') as f:
            json.dump(node.timestamps, f, indent=2)
        print(f"\n" + "="*70)
        print(f"✓ COMPLETE")
        print("="*70)
        print(f"Frames processed: {node.frame_count}")
        print(f"Timestamps saved: {output_timestamps}\n")
    else:
        print(f"\n" + "="*70)
        print(f"✓ COMPLETE")
        print("="*70)
        print(f"Frames written: {node.frame_count}")
        print(f"Output: {output_video}\n")

    rclpy.shutdown()

if __name__ == "__main__":
    main()
