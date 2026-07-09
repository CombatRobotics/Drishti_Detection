#!/usr/bin/env python3
"""
Match Encoder Data to Detection Frames
======================================

Reads detection frame filenames and matches them to encoder data from CSV.
Creates JSON mapping detections to their corresponding encoder values.

Usage:
    python match_encoder_to_detections.py
"""

import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

# ============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# ============================================================================
ENCODER_CSV_PATH = r"/media/viraj/e84db5a4-80aa-41f1-a173-de4e78ab820d1/home/cri-pc-0/june-july-dhristi-bags/day_4/day4_dhristi_track_measurement_20260707_022848/encoder_data_right_wheel_encoder.csv"
DETECTION_FRAMES_DIR = r"/media/viraj/e84db5a4-80aa-41f1-a173-de4e78ab820d1/home/cri-pc-0/june-july-dhristi-bags/day_4/day4_dhristi_railhead_cameras_20260707_022848/right_data/run_20260707_194549/faults"
OUTPUT_JSON_PATH = r"/media/viraj/e84db5a4-80aa-41f1-a173-de4e78ab820d1/home/cri-pc-0/june-july-dhristi-bags/day_4/day4_dhristi_railhead_cameras_20260707_022848/right_data/run_20260707_194549/faults/faults/detections_with_encoder_right.json"
TIMESTAMP_TOLERANCE_NS = 100000000  # 100ms tolerance for matching


def load_encoder_csv(csv_path: str) -> dict:
    """Load encoder CSV and create timestamp lookup."""
    encoder_data = {}
    encoder_list = []

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                sec = int(row['seconds'])
                nsec = int(row['nanoseconds'])
                timestamp = sec + nsec / 1e9

                encoder_data[timestamp] = {
                    "seconds": sec,
                    "nanoseconds": nsec,
                    "ist_time": row['ist_time'],
                    "encoder_value": float(row['encoder_value'])
                }
                encoder_list.append(timestamp)

        encoder_list.sort()
        print(f"✓ Loaded {len(encoder_data)} encoder messages from CSV")
        print(f"  Time range: {datetime.fromtimestamp(encoder_list[0]).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(encoder_list[-1]).strftime('%Y-%m-%d %H:%M:%S')}")
        return encoder_data, encoder_list

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return {}, []


def find_closest_encoder(timestamp: float, encoder_data: dict, encoder_list: list, tolerance_ns: int = TIMESTAMP_TOLERANCE_NS) -> Optional[dict]:
    """Find closest encoder data to given timestamp."""
    if not encoder_list:
        return None

    # Binary search for closest timestamp
    import bisect
    idx = bisect.bisect_left(encoder_list, timestamp)

    candidates = []
    if idx > 0:
        candidates.append(encoder_list[idx - 1])
    if idx < len(encoder_list):
        candidates.append(encoder_list[idx])

    closest = min(candidates, key=lambda t: abs(t - timestamp))

    # Check tolerance
    if abs(closest - timestamp) * 1e9 > tolerance_ns:
        return None

    return {
        "closest_timestamp": closest,
        "data": encoder_data[closest],
        "difference_ns": int(abs(closest - timestamp) * 1e9)
    }


def extract_timestamp_from_filename(filename: str) -> Optional[Tuple[str, float, int, int]]:
    """
    Extract timestamp from frame filename.
    Expected format: frame_YYYY-MM-DD_HH-MM-SS.NNNNNN_DETECTIONS.jpg
    Returns: (ist_time, timestamp_float, sec, nsec) or None
    """
    # Pattern: frame_2026-07-09_12-34-56.706041_2.jpg
    pattern = r'frame_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.(\d{6})_(\d+)\.jpg'
    match = re.search(pattern, filename)

    if not match:
        return None

    date_str, time_str, microsec_str, detections = match.groups()

    # Parse to timestamp
    try:
        dt = datetime.strptime(f"{date_str} {time_str.replace('-', ':')}", "%Y-%m-%d %H:%M:%S")
        timestamp_sec = int(dt.timestamp())
        microsec = int(microsec_str)
        timestamp_nsec = microsec * 1000  # microsec to nanosec

        timestamp_float = timestamp_sec + timestamp_nsec / 1e9
        ist_time = f"{date_str} {time_str.replace('-', ':')}.{microsec_str}"

        return ist_time, timestamp_float, timestamp_sec, timestamp_nsec

    except Exception as e:
        print(f"  ⚠️  Could not parse timestamp from {filename}: {e}")
        return None


def scan_detection_frames(frames_dir: str) -> list:
    """Scan directory for detection frame files."""
    frames_path = Path(frames_dir)

    if not frames_path.exists():
        print(f"❌ Frames directory not found: {frames_dir}")
        return []

    # Find all .jpg files (detection frames)
    frame_files = sorted(frames_path.glob("frame_*.jpg"))

    print(f"✓ Found {len(frame_files)} detection frames")
    return frame_files


def main():
    """Main function."""

    print("\n" + "="*70)
    print("MATCH ENCODER DATA TO DETECTION FRAMES")
    print("="*70)

    # Load encoder CSV
    print(f"\n📊 Loading encoder CSV...")
    print(f"  Path: {ENCODER_CSV_PATH}")
    encoder_data, encoder_list = load_encoder_csv(ENCODER_CSV_PATH)

    if not encoder_data:
        print("❌ No encoder data loaded. Exiting.")
        return

    # Scan detection frames
    print(f"\n🔍 Scanning detection frames...")
    print(f"  Directory: {DETECTION_FRAMES_DIR}")
    frame_files = scan_detection_frames(DETECTION_FRAMES_DIR)

    if not frame_files:
        print("❌ No detection frames found. Exiting.")
        return

    # Match each frame to encoder data
    print(f"\n⚙️  Matching frames to encoder data...")
    detections_with_encoder = []
    matched_count = 0
    unmatched_count = 0

    for frame_file in frame_files:
        filename = frame_file.name

        # Extract timestamp from filename
        timestamp_info = extract_timestamp_from_filename(filename)
        if not timestamp_info:
            unmatched_count += 1
            continue

        ist_time, timestamp_float, sec, nsec = timestamp_info

        # Find matching encoder data
        encoder_info = find_closest_encoder(timestamp_float, encoder_data, encoder_list, TIMESTAMP_TOLERANCE_NS)

        detection_entry = {
            "frame_filename": filename,
            "frame_timestamp_sec": sec,
            "frame_timestamp_nanosec": nsec,
            "frame_timestamp_ist": ist_time,
            "encoder_data": None,
            "encoder_timestamp_match_ns": None
        }

        if encoder_info:
            detection_entry["encoder_data"] = encoder_info["data"]
            detection_entry["encoder_timestamp_match_ns"] = encoder_info["difference_ns"]
            matched_count += 1
        else:
            unmatched_count += 1

        detections_with_encoder.append(detection_entry)

    # Save results to JSON
    output_path = Path(OUTPUT_JSON_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_json = {
        "timestamp": datetime.now().isoformat(),
        "encoder_csv_path": ENCODER_CSV_PATH,
        "detection_frames_dir": DETECTION_FRAMES_DIR,
        "timestamp_tolerance_ns": TIMESTAMP_TOLERANCE_NS,
        "total_frames": len(detections_with_encoder),
        "matched_frames": matched_count,
        "unmatched_frames": unmatched_count,
        "detections": detections_with_encoder
    }

    try:
        with open(output_path, 'w') as f:
            json.dump(output_json, f, indent=2)

        print(f"\n✓ Results saved to JSON")
        print(f"  Path: {output_path}")

    except Exception as e:
        print(f"❌ Error saving JSON: {e}")
        return

    # Summary
    print("\n" + "="*70)
    print("✓ MATCHING COMPLETE")
    print("="*70)
    print(f"Total detection frames: {len(detections_with_encoder)}")
    print(f"Matched with encoder data: {matched_count}")
    print(f"Unmatched (no encoder within {TIMESTAMP_TOLERANCE_NS/1e9:.3f}s): {unmatched_count}")
    print(f"Match rate: {(matched_count/len(detections_with_encoder)*100):.1f}%\n")


if __name__ == "__main__":
    main()
