#!/usr/bin/env python3
"""
Encoder Data Extractor from ROS2 Rosbag
========================================

Extracts all encoder messages from a rosbag and saves to CSV.
Creates a lookup reference for matching detection frames to encoder values.

Usage:
    python encoder_extractor.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Float32, Int32
from sensor_msgs.msg import JointState
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# ============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# ============================================================================
ROSBAG_FOLDER = r"/media/viraj/e84db5a4-80aa-41f1-a173-de4e78ab820d1/home/cri-pc-0/june-july-dhristi-bags/day_4/day4_dhristi_track_measurement_20260707_022848"
ENCODER_TOPIC = "/right_wheel_encoder"  # Encoder topic to extract
OUTPUT_DIR = ROSBAG_FOLDER  # Save CSV in rosbag folder or specify custom path


class EncoderExtractor(Node):
    """Node that extracts all encoder data from rosbag."""

    def __init__(self, encoder_topic: str, output_path: str):
        super().__init__("encoder_extractor")

        self.encoder_topic = encoder_topic
        self.output_path = output_path
        self.encoder_data = []
        self.last_encoder_value = None  # Track last unique distance/encoder value
        self.last_message_time = time.time()  # Track when last message arrived

        # Set permissive QoS for bag playback
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to encoder topic (handles JointState, Float32, Int32)
        try:
            self.sub = self.create_subscription(
                JointState,
                self.encoder_topic,
                self.encoder_callback,
                qos_profile
            )
            self.get_logger().info(f"✓ Subscribed to encoder topic: {self.encoder_topic}")
        except Exception as e:
            self.get_logger().error(f"❌ Could not subscribe to encoder topic: {e}")
            raise

        self.frame_count = 0

    def encoder_callback(self, msg):
        """Callback to extract encoder data."""
        try:
            timestamp_sec = msg.header.stamp.sec
            timestamp_nsec = msg.header.stamp.nanosec
            ist_time = datetime.fromtimestamp(timestamp_sec).strftime("%Y-%m-%d %H:%M:%S.%f")

            # Extract encoder value (handle JointState, Float32, Int32)
            if isinstance(msg, JointState):
                encoder_value = msg.position[0] if msg.position else 0.0
            else:
                encoder_value = msg.data

            encoder_value = float(encoder_value)

            # Only write if encoder value is different from last unique value
            if encoder_value == self.last_encoder_value:
                return

            self.last_encoder_value = encoder_value
            self.last_message_time = time.time()  # Update when we received a message

            self.encoder_data.append({
                "seconds": timestamp_sec,
                "nanoseconds": timestamp_nsec,
                "ist_time": ist_time,
                "encoder_value": encoder_value
            })

            self.frame_count += 1

            if self.frame_count % 1000 == 0:
                self.get_logger().info(f"  {self.frame_count} unique encoder values extracted...")

        except Exception as e:
            self.get_logger().error(f"Error processing encoder data: {e}")

    def save_to_csv(self):
        """Save all encoder data to CSV."""
        try:
            with open(self.output_path, 'w', newline='') as csvfile:
                fieldnames = ['seconds', 'nanoseconds', 'ist_time', 'encoder_value']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in self.encoder_data:
                    writer.writerow(row)

            self.get_logger().info(f"✓ CSV saved: {self.output_path}")
            self.get_logger().info(f"  Total encoder messages: {len(self.encoder_data)}")

        except Exception as e:
            self.get_logger().error(f"❌ Error saving CSV: {e}")
            raise


def main():
    """Main function."""

    total_start_time = time.time()

    # Prepare output path
    output_path = Path(OUTPUT_DIR) / f"encoder_data_{ENCODER_TOPIC.replace('/', '_').lstrip('_')}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("ENCODER DATA EXTRACTOR - MANUAL BAG PLAYBACK")
    print("="*70)
    print(f"\n📋 Configuration:")
    print(f"  Encoder topic: {ENCODER_TOPIC}")
    print(f"  Output CSV: {output_path}\n")

    print("📌 Instructions:")
    print("  1. In another terminal, run:")
    print(f"     ros2 bag play <rosbag_folder> --topics {ENCODER_TOPIC}")
    print("  2. This script will listen and record encoder values")
    print("  3. Press Ctrl+C to stop listening and save CSV\n")

    # Initialize ROS2
    rclpy.init()
    node = EncoderExtractor(ENCODER_TOPIC, str(output_path))

    print("🎧 Listening for encoder messages...\n")
    try:
        # Spin and wait for messages from manually started bag playback
        while True:
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    finally:
        total_elapsed_time = time.time() - total_start_time

        # Save CSV
        node.save_to_csv()

        # Print summary
        print("\n" + "="*70)
        print("✓ EXTRACTION COMPLETE")
        print("="*70)
        print(f"Unique encoder values: {node.frame_count}")
        print(f"Total rows in CSV: {len(node.encoder_data)}")
        print(f"Extraction time: {total_elapsed_time:.2f}s")
        print(f"Output CSV: {output_path}")
        print(f"\n⚠️  Verify count matches 'ros2 bag info' for {ENCODER_TOPIC}\n")

        rclpy.shutdown()


if __name__ == "__main__":
    main()
