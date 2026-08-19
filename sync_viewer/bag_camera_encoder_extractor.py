#!/usr/bin/env python3
"""
Multi-Camera + Encoder Bag Extractor
=====================================

Extracts raw JPEG frames and encoder samples from ROS2 rosbag2 sessions.
Handles multi-folder sessions (plinth_cameras, railhead_cameras, track_measurement)
recorded in parallel, auto-discovers topic-to-folder mapping, and extracts
a user-selected subset of cameras with synchronized encoder data.

Usage:
    python3 bag_camera_encoder_extractor.py --session /path/to/Day-4
    python3 bag_camera_encoder_extractor.py --session /path/to/Day-4 --cameras rail_left,rail_right
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
import cv2
import numpy as np
import json
import yaml
import csv
import argparse
import subprocess
import time
import os
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from functools import partial


def load_camera_config(config_path: str) -> Tuple[List[Dict], Dict]:
    """Load camera and encoder topic configuration from JSON."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        cameras = config.get('cameras', [])
        encoders = config.get('encoders', {})
        return cameras, encoders
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config: {e}")


def discover_session(session_dir: str, config_cameras: List[Dict], config_encoders: Dict) -> Dict:
    """
    Scan session_dir for rosbag2 sub-folders (metadata.yaml) and build topic->bag mapping.
    Returns:
        {
            "bags": [
                {"path": Path, "name": str, "topics": {topic: msg_count}, ...},
                ...
            ],
            "topic_to_bags": {"/topic": [(path, msg_count), ...]},
            "encoder_source_bag": Path or None,
            "camera_status": {key: (topic, [bag_names_containing_it])},
            "requested_cameras": [keys that can actually be extracted]
        }
    """
    session_path = Path(session_dir)
    if not session_path.is_dir():
        raise ValueError(f"Session directory not found: {session_dir}")

    bags = []
    topic_to_bags = {}

    # Scan immediate subdirectories for metadata.yaml
    for subfolder in sorted(session_path.iterdir()):
        if not subfolder.is_dir():
            continue
        metadata_path = subfolder / 'metadata.yaml'
        if not metadata_path.exists():
            continue

        # Parse metadata.yaml
        try:
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            topics_with_counts = metadata.get('rosbag2_bagfile_information', {}).get('topics_with_message_count', [])
            topics = {}
            for tc in topics_with_counts:
                topic_name = tc.get('topic_metadata', {}).get('name')
                msg_count = tc.get('message_count')
                if topic_name:
                    topics[topic_name] = msg_count
                    if topic_name not in topic_to_bags:
                        topic_to_bags[topic_name] = []
                    topic_to_bags[topic_name].append((subfolder, msg_count))

            bags.append({
                'path': subfolder,
                'name': subfolder.name,
                'topics': topics
            })
        except Exception as e:
            print(f"⚠️  Error parsing {metadata_path}: {e}")
            continue

    if not bags:
        raise ValueError(f"No rosbag2 folders (metadata.yaml) found in {session_dir}")

    # Check which requested cameras exist
    config_camera_topics = {cam['key']: cam['topic'] for cam in config_cameras}
    camera_status = {}
    requested_cameras = []

    for key, topic in config_camera_topics.items():
        if topic in topic_to_bags:
            bag_names = [Path(bag_path).name for bag_path, _ in topic_to_bags[topic]]
            camera_status[key] = (topic, bag_names)
            requested_cameras.append(key)
        else:
            camera_status[key] = (topic, [])  # topic not found in any bag

    # Pick encoder source bag: prefer a bag that also has a selected camera
    encoder_topics = [config_encoders.get('left'), config_encoders.get('right')]
    encoder_source_bag = None
    encoder_source_name = None

    for encoder_topic in encoder_topics:
        if not encoder_topic or encoder_topic not in topic_to_bags:
            continue
        # Pick the first bag containing this encoder topic
        encoder_source_bag = topic_to_bags[encoder_topic][0][0]
        encoder_source_name = encoder_source_bag.name
        break

    return {
        'bags': bags,
        'topic_to_bags': topic_to_bags,
        'encoder_source_bag': encoder_source_bag,
        'encoder_source_name': encoder_source_name,
        'camera_status': camera_status,
        'requested_cameras': requested_cameras,
        'encoder_topics': [t for t in encoder_topics if t]
    }


class MultiCameraEncoderCapture(Node):
    """ROS2 Node to capture multiple camera topics and encoder data."""

    def __init__(self, selected_cameras: List[Dict], encoder_topics: List[str], output_dir: Path):
        super().__init__(
            'bag_camera_encoder_extractor',
            parameter_overrides=[Parameter('use_sim_time', Parameter.Type.BOOL, True)]
        )

        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / 'frames'
        self.index_dir = self.output_dir / 'index'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        # Per-camera index: camera_key -> list of (timestamp_ns, filename)
        self.camera_index = {cam['key']: [] for cam in selected_cameras}
        self.encoder_raw = {
            'left': [],
            'right': []
        }

        # QoS profile matching existing scripts
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscribe to each selected camera topic
        for cam in selected_cameras:
            key = cam['key']
            topic = cam['topic']
            self.create_subscription(
                CompressedImage,
                topic,
                partial(self._camera_cb, key),
                qos
            )
            self.get_logger().info(f"✓ Subscribed to {key}: {topic}")

            # Create frame subdirectory for this camera
            (self.frames_dir / key).mkdir(exist_ok=True)

        # Subscribe to encoder topics
        self.encoder_topic_left = encoder_topics[0] if len(encoder_topics) > 0 else None
        self.encoder_topic_right = encoder_topics[1] if len(encoder_topics) > 1 else None

        if self.encoder_topic_left:
            self.create_subscription(Float32, self.encoder_topic_left, partial(self._encoder_cb, 'left'), qos)
            self.get_logger().info(f"✓ Subscribed to left encoder: {self.encoder_topic_left}")

        if self.encoder_topic_right:
            self.create_subscription(Float32, self.encoder_topic_right, partial(self._encoder_cb, 'right'), qos)
            self.get_logger().info(f"✓ Subscribed to right encoder: {self.encoder_topic_right}")

        self.frame_count = {key: 0 for key in self.camera_index.keys()}
        self.encoder_count = {'left': 0, 'right': 0}

    def _camera_cb(self, key: str, msg: CompressedImage):
        """Callback for camera messages. Write raw JPEG bytes directly."""
        try:
            # Convert timestamp to nanoseconds
            ts_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

            # Write raw JPEG bytes directly (no re-encode)
            filename = f"{ts_ns}.jpg"
            frame_path = self.frames_dir / key / filename
            with open(frame_path, 'wb') as f:
                f.write(msg.data)

            # Record index entry
            self.camera_index[key].append((ts_ns, filename))
            self.frame_count[key] += 1

            if self.frame_count[key] % 100 == 0 or self.frame_count[key] <= 5:
                self.get_logger().info(f"  {key}: {self.frame_count[key]:6d} frames")
        except Exception as e:
            self.get_logger().error(f"Error processing frame from {key}: {e}")

    def _encoder_cb(self, side: str, msg: Float32):
        """Callback for encoder messages. Use sim-time (from --clock)."""
        try:
            # Get timestamp from ROS clock (will be sim-time if --clock is running)
            ts_ns = self.get_clock().now().nanoseconds
            value_mm = float(msg.data)

            self.encoder_raw[side].append((ts_ns, value_mm))
            self.encoder_count[side] += 1

            if self.encoder_count[side] % 1000 == 0:
                self.get_logger().info(f"  {side} encoder: {self.encoder_count[side]} samples (latest: {value_mm:.1f} mm)")
        except Exception as e:
            self.get_logger().error(f"Error processing encoder {side}: {e}")

    def finalize(self) -> Dict:
        """
        Finalize extraction: write CSVs and manifest.
        Returns manifest dict for summary printing.
        """
        self.get_logger().info("\n" + "=" * 70)
        self.get_logger().info("FINALIZING EXTRACTION")
        self.get_logger().info("=" * 70)

        manifest = {
            'schema_version': 1,
            'extraction_timestamp_iso': datetime.now(timezone.utc).isoformat(),
            'output_dir': str(self.output_dir),
            'cameras': [],
            'encoders': []
        }

        # Write per-camera indices
        for key in sorted(self.camera_index.keys()):
            index = self.camera_index[key]
            if not index:
                self.get_logger().warning(f"  ⚠️  No frames for {key}")
                continue

            # Sort by timestamp
            index.sort(key=lambda x: x[0])

            # Write CSV
            csv_path = self.index_dir / f"{key}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_ns', 'filename'])
                for ts_ns, filename in index:
                    writer.writerow([ts_ns, filename])

            # Record in manifest
            first_ts = index[0][0]
            last_ts = index[-1][0]
            manifest['cameras'].append({
                'key': key,
                'frame_count': len(index),
                'first_ts_ns': first_ts,
                'last_ts_ns': last_ts
            })
            self.get_logger().info(f"✓ {key}: {len(index)} frames, {csv_path.name}")

        # Write encoder indices
        for side in ['left', 'right']:
            encoder_list = self.encoder_raw[side]
            if not encoder_list:
                self.get_logger().warning(f"  ⚠️  No samples for {side} encoder")
                continue

            # Sort by timestamp
            encoder_list.sort(key=lambda x: x[0])

            # Write CSV
            csv_path = self.index_dir / f"encoder_{side}.csv"
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp_ns', 'value_mm'])
                for ts_ns, value_mm in encoder_list:
                    writer.writerow([ts_ns, value_mm])

            first_ts = encoder_list[0][0]
            last_ts = encoder_list[-1][0]
            manifest['encoders'].append({
                'side': side,
                'sample_count': len(encoder_list),
                'first_ts_ns': first_ts,
                'last_ts_ns': last_ts
            })
            self.get_logger().info(f"✓ encoder_{side}: {len(encoder_list)} samples, {csv_path.name}")

        # Write manifest
        manifest_path = self.output_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        self.get_logger().info(f"✓ Manifest written to {manifest_path.name}")

        return manifest


def main():
    try:
        parser = argparse.ArgumentParser(
            description='Extract frames and encoder data from rosbag2 multi-folder sessions.'
        )
        parser.add_argument('--session', required=True, help='Path to session directory (e.g., Day-4)')
        parser.add_argument('--output-dir', default=None, help='Output directory (default: <session>/sync_extract1)')
        parser.add_argument('--config', default='./camera_topics_config.json', help='Camera config file')
        parser.add_argument('--cameras', default=None, help='Comma-separated camera keys (default: all)')
        parser.add_argument('--clock-rate', type=int, default=200, help='Clock publish rate for --clock (Hz)')
        parser.add_argument('--rate', type=float, default=1.0, help='Bag playback rate')

        args = parser.parse_args()

        # Load configuration
        config_cameras, config_encoders = load_camera_config(args.config)

        # Discover session
        session_info = discover_session(args.session, config_cameras, config_encoders)

        print("\n" + "=" * 70)
        print("BAG DISCOVERY")
        print("=" * 70)
        print(f"Session: {args.session}")
        print(f"Bags found: {len(session_info['bags'])}")
        for bag in session_info['bags']:
            print(f"  - {bag['name']}: {len(bag['topics'])} topics")

        print(f"\nCamera status:")
        for key, (topic, bag_names) in session_info['camera_status'].items():
            if bag_names:
                print(f"  ✓ {key:20s} {topic}")
            else:
                print(f"  ✗ {key:20s} NOT FOUND")

        print(f"\nEncoder source bag: {session_info['encoder_source_name']}")

        # Parse requested cameras
        if args.cameras:
            requested_keys = args.cameras.split(',')
            selected_cameras = [cam for cam in config_cameras if cam['key'] in requested_keys]
        else:
            selected_cameras = [cam for cam in config_cameras if cam['key'] in session_info['requested_cameras']]

        if not selected_cameras:
            print("❌ No valid cameras selected or found.")
            return 1

        print(f"\nSelected cameras for extraction: {[cam['key'] for cam in selected_cameras]}")

        # Set output directory
        if args.output_dir is None:
            output_dir = Path(args.session) / 'sync_extract'
        else:
            output_dir = Path(args.output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        # Identify which bag folders we need to play
        encoder_topic_list = [config_encoders.get('left'), config_encoders.get('right')]
        encoder_topic_list = [t for t in encoder_topic_list if t]

        bags_to_play = {}  # bag_path_str -> list of topics to play from it
        for cam in selected_cameras:
            # Find which bag(s) contain this camera
            topic = cam['topic']
            if topic in session_info['topic_to_bags']:
                for bag_path, _ in session_info['topic_to_bags'][topic]:
                    bag_path_str = str(bag_path)
                    if bag_path_str not in bags_to_play:
                        bags_to_play[bag_path_str] = []
                    bags_to_play[bag_path_str].append(topic)

        # Add encoder source bag if not already included
        if session_info['encoder_source_bag']:
            encoder_src_path_str = str(session_info['encoder_source_bag'])
            if encoder_src_path_str not in bags_to_play:
                bags_to_play[encoder_src_path_str] = []
            # Add encoder topics to encoder source bag
            for encoder_topic in encoder_topic_list:
                if encoder_topic not in bags_to_play[encoder_src_path_str]:
                    bags_to_play[encoder_src_path_str].append(encoder_topic)

        print(f"\n" + "=" * 70)
        print("STARTING EXTRACTION")
        print("=" * 70)
        print(f"Output directory: {output_dir}")
        print(f"Clock rate: {args.clock_rate} Hz")
        print(f"Playback rate: {args.rate}x")
        print(f"\nBags to play: {len(bags_to_play)}")
        for bag_path_str, topics in sorted(bags_to_play.items()):
            bag_name = Path(bag_path_str).name
            print(f"  - {bag_name}: {len(topics)} topics")

        # Launch ros2 bag play processes
        processes = []
        for bag_path_str, topics in bags_to_play.items():
            cmd = ['ros2', 'bag', 'play', bag_path_str, '--topics'] + topics + ['-r', str(args.rate)]

            # Only the encoder source bag gets --clock
            encoder_src_str = str(session_info['encoder_source_bag']) if session_info['encoder_source_bag'] else None
            if bag_path_str == encoder_src_str:
                cmd.extend(['--clock', str(args.clock_rate)])

            bag_name = Path(bag_path_str).name
            print(f"\nLaunching: {bag_name}")
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(proc)

        time.sleep(1)  # Give processes time to start

        # Initialize ROS and capture node
        rclpy.init()
        node = MultiCameraEncoderCapture(selected_cameras, encoder_topic_list, output_dir)

        print("\nSpinning... (waiting for bag playback to complete)")

        # Spin until all processes exit
        try:
            while any(proc.poll() is None for proc in processes):
                rclpy.spin_once(node, timeout_sec=0.1)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")

        # Final drain
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)

        # Finalize
        manifest = node.finalize()

        rclpy.shutdown()

        # Summary
        print("\n" + "=" * 70)
        print("EXTRACTION COMPLETE")
        print("=" * 70)
        print(f"Output directory: {output_dir}\n")
        for cam_info in manifest['cameras']:
            print(f"{cam_info['key']:20s}: {cam_info['frame_count']:6d} frames")
        print()
        for enc_info in manifest['encoders']:
            print(f"encoder_{enc_info['side']:5s}: {enc_info['sample_count']:6d} samples")

        print("\n" + "=" * 70)
        print("✓ READY FOR VIEWER")
        print("=" * 70 + "\n")

        return 0

    except Exception as e:
        import traceback
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
