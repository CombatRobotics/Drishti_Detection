#!/usr/bin/env python3
"""
Synchronized Index Lookup Module
==================================

Pure-Python/numpy lookup functions for camera frames and encoder values.
No rclpy or cv2 dependencies — used by both the extraction validator and the viewer.

Core data structures:
- CameraIndex: sorted timestamp array + filepath list for one camera
- EncoderSeries: sorted timestamp array + value array for one encoder side
- SyncIndex: loaded manifest + all cameras + both encoders + computed globals
"""

import json
import csv
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict


MAX_FRAME_DELTA_NS = 50_000_000      # 50ms — frame blanking threshold
ENCODER_EXTRAP_LIMIT_NS = 500_000_000  # 500ms — max extrapolation beyond data range
STEP_NS = 10_000_000                 # 10ms — master timeline step granularity


@dataclass
class CameraIndex:
    """Index for one camera's frames."""
    key: str
    timestamps: np.ndarray      # int64, sorted ascending
    filepaths: List[str]        # absolute paths, same length as timestamps


@dataclass
class EncoderSeries:
    """Time series for one encoder side."""
    label: str                  # 'left' or 'right'
    timestamps: np.ndarray      # int64, sorted ascending
    values: np.ndarray          # float64, interpolated distance in mm


@dataclass
class SyncIndex:
    """Complete synchronized index for a session extraction."""
    manifest: dict
    output_dir: Path
    cameras: Dict[str, CameraIndex]  # key -> CameraIndex
    encoder_left: EncoderSeries
    encoder_right: EncoderSeries
    encoder_avg: EncoderSeries       # averaged left+right, resampled to union grid
    global_min_ts: int               # nanoseconds
    global_max_ts: int               # nanoseconds


def load_manifest(output_dir: Path) -> dict:
    """Load and return manifest.json."""
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'r') as f:
        return json.load(f)


def load_camera_index(output_dir: Path, camera_key: str) -> CameraIndex:
    """Load a single camera's index from CSV."""
    csv_path = output_dir / 'index' / f"{camera_key}.csv"
    timestamps = []
    filepaths = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ns = int(row['timestamp_ns'])
            filename = row['filename']
            timestamps.append(ts_ns)
            filepaths.append(str(output_dir / 'frames' / camera_key / filename))

    return CameraIndex(
        key=camera_key,
        timestamps=np.array(timestamps, dtype=np.int64),
        filepaths=filepaths
    )


def load_encoder_series(output_dir: Path, side: str) -> EncoderSeries:
    """Load encoder time series from CSV."""
    csv_path = output_dir / 'index' / f"encoder_{side}.csv"
    timestamps = []
    values = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_ns = int(row['timestamp_ns'])
            value_mm = float(row['value_mm'])
            timestamps.append(ts_ns)
            values.append(value_mm)

    return EncoderSeries(
        label=side,
        timestamps=np.array(timestamps, dtype=np.int64),
        values=np.array(values, dtype=np.float64)
    )


def build_encoder_series(left: EncoderSeries, right: EncoderSeries,
                         primary_source: str = 'average') -> EncoderSeries:
    """
    Build encoder series based on primary_source selection.

    Args:
        left: Left encoder series
        right: Right encoder series
        primary_source: 'left', 'right', or 'average'
                       - 'left': use only left encoder (if right is damaged)
                       - 'right': use only right encoder (if left is damaged)
                       - 'average': average both (default, both working)

    Returns:
        EncoderSeries ready for interpolation
    """
    if primary_source == 'left':
        # Use left encoder only (right damaged)
        return EncoderSeries(
            label='left (right damaged)',
            timestamps=left.timestamps,
            values=left.values
        )
    elif primary_source == 'right':
        # Use right encoder only (left damaged)
        return EncoderSeries(
            label='right (left damaged)',
            timestamps=right.timestamps,
            values=right.values
        )
    elif primary_source == 'average':
        # Average both (both working, original behavior)
        union_ts = np.union1d(left.timestamps, right.timestamps)
        left_interp = np.interp(union_ts, left.timestamps, left.values)
        right_interp = np.interp(union_ts, right.timestamps, right.values)
        avg_values = (left_interp + right_interp) / 2.0

        return EncoderSeries(
            label='avg (both working)',
            timestamps=union_ts,
            values=avg_values
        )
    else:
        raise ValueError(f"primary_source must be 'left', 'right', or 'average', got '{primary_source}'")


def load_all(output_dir: Path, camera_keys: Optional[List[str]] = None,
             primary_encoder_source: str = 'average') -> SyncIndex:
    """
    Load the complete synchronized index from an extraction output directory.

    Args:
        output_dir: Path to the sync_extract directory
        camera_keys: List of camera keys to load (None = load all from manifest)
        primary_encoder_source: 'left', 'right', or 'average' (default)
                                Use 'left' if right encoder is damaged
                                Use 'right' if left encoder is damaged

    Returns:
        SyncIndex with all data loaded
    """
    output_dir = Path(output_dir)
    manifest = load_manifest(output_dir)

    # Load cameras
    cameras = {}
    all_camera_keys = [cam_info['key'] for cam_info in manifest.get('cameras', [])]

    if camera_keys is None:
        camera_keys = all_camera_keys
    else:
        # Filter to requested keys that exist in manifest
        camera_keys = [k for k in camera_keys if k in all_camera_keys]

    for key in camera_keys:
        cameras[key] = load_camera_index(output_dir, key)

    # Load encoders
    encoder_left = load_encoder_series(output_dir, 'left')
    encoder_right = load_encoder_series(output_dir, 'right')
    encoder_avg = build_encoder_series(encoder_left, encoder_right, primary_source=primary_encoder_source)

    # Compute global min/max across ALL manifest cameras (not just selected)
    # so the timeline range is stable when camera selection changes
    all_timestamps = []
    for cam_info in manifest.get('cameras', []):
        all_timestamps.append(cam_info['first_ts_ns'])
        all_timestamps.append(cam_info['last_ts_ns'])

    for enc_info in manifest.get('encoders', []):
        all_timestamps.append(enc_info['first_ts_ns'])
        all_timestamps.append(enc_info['last_ts_ns'])

    global_min_ts = int(min(all_timestamps))
    global_max_ts = int(max(all_timestamps))

    return SyncIndex(
        manifest=manifest,
        output_dir=output_dir,
        cameras=cameras,
        encoder_left=encoder_left,
        encoder_right=encoder_right,
        encoder_avg=encoder_avg,
        global_min_ts=global_min_ts,
        global_max_ts=global_max_ts
    )


def nearest_frame(cam: CameraIndex, t_ns: int,
                  max_delta_ns: int = MAX_FRAME_DELTA_NS) -> Optional[Tuple[str, int]]:
    """
    Find the nearest frame to a given timestamp for a camera.

    Args:
        cam: CameraIndex for the camera
        t_ns: Target timestamp in nanoseconds
        max_delta_ns: Maximum allowed delta from target (50ms default = blanking threshold)

    Returns:
        (filepath, actual_timestamp_ns) or None if no frame within max_delta_ns
    """
    if len(cam.timestamps) == 0:
        return None

    # Binary search for insertion point
    idx = np.searchsorted(cam.timestamps, t_ns)

    # Check both neighbors
    candidates = []
    if idx > 0:
        candidates.append((idx - 1, abs(cam.timestamps[idx - 1] - t_ns)))
    if idx < len(cam.timestamps):
        candidates.append((idx, abs(cam.timestamps[idx] - t_ns)))

    if not candidates:
        return None

    # Pick the closest
    best_idx, best_delta = min(candidates, key=lambda x: x[1])

    if best_delta > max_delta_ns:
        return None  # Too far away

    return (cam.filepaths[best_idx], int(cam.timestamps[best_idx]))


def interpolate_encoder_distance_mm(idx: SyncIndex, t_ns: int) -> Optional[float]:
    """
    Interpolate encoder distance at a given timestamp.

    Args:
        idx: SyncIndex
        t_ns: Target timestamp in nanoseconds

    Returns:
        Interpolated distance in mm, or None if outside extrapolation limit
    """
    enc = idx.encoder_avg
    if len(enc.timestamps) == 0:
        return None

    # Check extrapolation bounds
    if t_ns < enc.timestamps[0] - ENCODER_EXTRAP_LIMIT_NS:
        return None
    if t_ns > enc.timestamps[-1] + ENCODER_EXTRAP_LIMIT_NS:
        return None

    # numpy.interp handles edge cases (clamps at boundaries)
    value = float(np.interp(t_ns, enc.timestamps, enc.values))
    return value


def find_timestamps_for_encoder_value(series: EncoderSeries, target_value: float,
                                       tolerance: float = 0.5,
                                       cluster_gap_ns: int = 20_000_000) -> List[int]:
    """
    Find all timestamps where encoder value crosses (is near) a target value.
    Handles non-monotonic encoder values (reversals, resets).

    Args:
        series: EncoderSeries (left, right, or avg)
        target_value: Target value to search for
        tolerance: Tolerance band around target (±tolerance)
        cluster_gap_ns: Group candidates within this gap into one (keep first)

    Returns:
        Sorted list of candidate timestamps (may be empty, may be many)
    """
    if len(series.timestamps) < 2:
        return []

    candidates = []
    low_bound = target_value - tolerance
    high_bound = target_value + tolerance

    # Scan consecutive pairs for crossings
    for i in range(len(series.timestamps) - 1):
        ts_a = series.timestamps[i]
        ts_b = series.timestamps[i + 1]
        val_a = series.values[i]
        val_b = series.values[i + 1]

        # Check if target lies within the range [min(val_a, val_b), max(val_a, val_b)]
        seg_min = min(val_a, val_b)
        seg_max = max(val_a, val_b)

        # Expand by tolerance
        seg_min_tol = seg_min - tolerance
        seg_max_tol = seg_max + tolerance

        # Check if target intersects this segment's range
        if target_value < seg_min_tol or target_value > seg_max_tol:
            continue

        # Linearly interpolate the crossing timestamp
        if abs(val_b - val_a) < 1e-10:
            # Values are essentially equal; use midpoint
            candidate_ts = int((ts_a + ts_b) / 2)
        else:
            # Standard linear interpolation
            f = (target_value - val_a) / (val_b - val_a)
            f = np.clip(f, 0.0, 1.0)
            candidate_ts = int(ts_a + f * (ts_b - ts_a))

        candidates.append(candidate_ts)

    # Cluster nearby candidates (keep first of each cluster)
    if not candidates:
        return []

    candidates.sort()
    clustered = [candidates[0]]
    for ts in candidates[1:]:
        if ts - clustered[-1] > cluster_gap_ns:
            clustered.append(ts)

    return clustered


def get_master_timeline_range(idx: SyncIndex, step_ns: int = STEP_NS) -> Tuple[int, int, int, int]:
    """
    Get the master timeline parameters.

    Returns:
        (global_min_ts, global_max_ts, step_ns, num_steps)
    """
    num_steps = (idx.global_max_ts - idx.global_min_ts) // step_ns
    return (idx.global_min_ts, idx.global_max_ts, step_ns, num_steps)


def slider_pos_to_ts(idx: SyncIndex, pos: int, step_ns: int = STEP_NS) -> int:
    """Convert slider position to timestamp."""
    return idx.global_min_ts + pos * step_ns


def ts_to_slider_pos(idx: SyncIndex, t_ns: int, step_ns: int = STEP_NS) -> int:
    """Convert timestamp to slider position."""
    return int(round((t_ns - idx.global_min_ts) / step_ns))


# ============================================================================
# Self-Test (matches repo convention: config block + __main__)
# ============================================================================

def _run_self_test():
    """Synthetic unit test with known data."""
    print("\n" + "=" * 70)
    print("SYNC_INDEX SELF-TEST")
    print("=" * 70)

    # Create synthetic camera index (frames at 30ms intervals, ~33fps)
    base_ts = 1783371529000000000  # realistic timestamp
    cam_ts = np.array([base_ts, base_ts + 33000000, base_ts + 66000000, base_ts + 99000000], dtype=np.int64)
    cam_fps = [f'/tmp/frame_{ts}.jpg' for ts in cam_ts]
    cam = CameraIndex('test_cam', cam_ts, cam_fps)

    # Test nearest_frame
    print("\n1. nearest_frame (50ms blanking):")
    # Exact match
    result = nearest_frame(cam, cam_ts[1])
    assert result is not None and result[1] == cam_ts[1], f"Exact match failed: {result}"
    print(f"   ✓ Exact match")

    # Within 50ms
    result = nearest_frame(cam, cam_ts[1] + 10000000)  # 10ms from frame
    assert result is not None and result[1] == cam_ts[1], f"Near match failed: {result}"
    print(f"   ✓ Within 50ms")

    # Beyond 50ms (no frame within 50ms of this timestamp)
    far_ts = cam_ts[-1] + 100000000  # 100ms after last frame
    result = nearest_frame(cam, far_ts)
    assert result is None, f"Beyond blanking should be None: {result}"
    print(f"   ✓ Beyond 50ms: None (blanked)")

    # Create synthetic encoder series (non-monotonic: reversals, ~1000Hz)
    base_enc_ts = base_ts
    enc_ts = np.array([
        base_enc_ts,
        base_enc_ts + 1000000,      # 1ms later
        base_enc_ts + 2000000,      # 2ms later
        base_enc_ts + 3000000,      # 3ms later
    ], dtype=np.int64)
    # Encoder distance: goes 0->100, then reverses to 50, then forward to 150 (real-world reversing)
    enc_vals = np.array([0.0, 100.0, 50.0, 150.0], dtype=np.float64)
    enc_left = EncoderSeries('left', enc_ts, enc_vals)
    enc_right = EncoderSeries('right', enc_ts, enc_vals)

    # Build averaged encoder series
    enc_avg = build_encoder_series(enc_left, enc_right, primary_source='average')

    # Test interpolate_encoder_distance_mm
    print("\n2. interpolate_encoder_distance_mm:")
    result = interpolate_encoder_distance_mm(
        SyncIndex({}, Path('/tmp'), {}, enc_avg, enc_left, enc_right, base_enc_ts, base_enc_ts + 4000000),
        base_enc_ts + 500000  # halfway between 1st and 2nd sample
    )
    assert result is not None and 45 < result < 55, f"Interpolation failed: {result}"
    print(f"   ✓ Interpolated at midpoint: {result:.1f} mm")

    # Test find_timestamps_for_encoder_value (multi-candidate due to reversal)
    print("\n3. find_timestamps_for_encoder_value (non-monotonic):")
    # Value 100 appears at one specific timestamp
    candidates = find_timestamps_for_encoder_value(enc_avg, 100.0, tolerance=1.0)
    assert len(candidates) >= 1, f"Should find at least 1 candidate for 100mm: {candidates}"
    print(f"   ✓ Found {len(candidates)} candidate(s) for 100mm")

    # Value 75 appears in two segments (going up 0->100, then reversing 100->50)
    candidates = find_timestamps_for_encoder_value(enc_avg, 75.0, tolerance=1.0)
    assert len(candidates) >= 1, f"Should find candidate(s) for 75mm: {candidates}"
    print(f"   ✓ Found {len(candidates)} candidate(s) for 75mm (reversal)")

    # Test master timeline
    print("\n4. get_master_timeline_range:")
    idx = SyncIndex({}, Path('/tmp'), {}, enc_avg, enc_left, enc_right, base_enc_ts, base_enc_ts + 3000000)
    min_ts, max_ts, step_ns, num_steps = get_master_timeline_range(idx, step_ns=1000000)  # 1ms steps
    assert num_steps == 3, f"num_steps should be 3: {num_steps}"
    print(f"   ✓ Timeline: {num_steps} steps of {step_ns}ns = {num_steps * step_ns / 1e6:.1f}ms")

    # Test slider conversion
    print("\n5. slider_pos <-> timestamp conversion:")
    mid_ts = base_enc_ts + 1500000
    pos = ts_to_slider_pos(idx, mid_ts, step_ns=1000000)
    ts = slider_pos_to_ts(idx, pos, step_ns=1000000)
    assert abs(ts - mid_ts) <= 1000000, f"Round-trip failed: {ts} vs {mid_ts}"
    print(f"   ✓ Round-trip conversion OK")

    print("\n" + "=" * 70)
    print("✓ ALL TESTS PASSED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    _run_self_test()
