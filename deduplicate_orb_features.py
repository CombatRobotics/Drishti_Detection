#!/usr/bin/env python3
"""
Feature-based deduplication using ORB (Oriented FAST and Rotated BRIEF).

Compares frames by matching distinctive keypoints and descriptors.
More robust to lighting, camera vibration, and compression than pixel/edge methods.
Optionally restrict feature extraction to specific regions using alpha masks.

========== HOW TO RUN ==========

1. Basic usage:
   python3 deduplicate_orb_features.py /path/to/input_folder

2. Custom output folder:
   python3 deduplicate_orb_features.py /path/to/input_folder --output /path/to/output

3. Adjust minimum match count (higher = stricter):
   python3 deduplicate_orb_features.py /path/to/input_folder --min-matches 250

4. Adjust number of ORB features:
   python3 deduplicate_orb_features.py /path/to/input_folder --n-features 500

5. Combine options:
   python3 deduplicate_orb_features.py /path/to/input_folder \
     --min-matches 200 \
     --n-features 450 \
     --window 4 \
     --output /custom/output

REGION MASK CONFIGURATION:
  - Set MASK_PATH in script configuration (line ~56)
  - Set to None to disable masking (full frame analysis)
  - Binary mask image (same size as input frames)
  - White (255) = region to analyze for features
  - Black (0) = ignore region
  - Features only extracted from white regions
  - Example: mask out rail edges, focus on defects only

Output:
  - Deduplicated frames saved to output folder
  - dedup_metadata_orb.json with match statistics
"""

import cv2
import os
import json
import numpy as np
from datetime import datetime
import argparse

# ==========================================================================
# CONFIGURATION - Edit these as needed
# ==========================================================================
MASK_PATH = r"/media/viraj/New Volume/Dhrishti/Detection_code/Mask.png"  # Path to region mask (set to None to disable masking)
N_FEATURES = 650             # Number of ORB features to detect
COMPARISON_WINDOW = 3        # Compare with ±N frames (2 = 5 total frames: -2,-1,0,+1,+2)
MIN_MATCH_COUNT = 50        # Minimum number of matching features to consider a duplicate (ONLY criterion, removed ratio threshold)
USE_CENTER_FEATURES = True   # If True, prefer frames with features closest to center
CENTER_FEATURE_RADIUS = None # Max distance from center (pixels). None = use all features
USE_GPU = True               # If True, use GPU acceleration where available


def create_orb_detector(n_features=N_FEATURES):
    """Create ORB feature detector."""
    return cv2.ORB_create(nfeatures=n_features)


def calculate_avg_center_distance(keypoints, image_h, image_w):
    """
    Calculate average distance of keypoints from image center.

    Args:
        keypoints: List of ORB keypoints
        image_h: Image height
        image_w: Image width

    Returns:
        float: Average distance from center (pixels)
    """
    if len(keypoints) == 0:
        return float('inf')

    center_x, center_y = image_w / 2.0, image_h / 2.0
    distances = []

    for kp in keypoints:
        x, y = kp.pt
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        distances.append(distance)

    return float(np.mean(distances))


def filter_keypoints_by_center_distance(keypoints, descriptors, image_h, image_w,
                                       use_center=USE_CENTER_FEATURES,
                                       radius=CENTER_FEATURE_RADIUS):
    """
    Filter keypoints to only those closest to image center.

    Args:
        keypoints: List of ORB keypoints
        descriptors: Numpy array of descriptors
        image_h: Image height
        image_w: Image width
        use_center: If True, apply center filtering
        radius: Max distance from center (pixels). None = use all

    Returns:
        tuple: (filtered_keypoints, filtered_descriptors)
    """
    if not use_center or len(keypoints) == 0:
        return keypoints, descriptors

    center_x, center_y = image_w / 2.0, image_h / 2.0

    # Calculate distances from center
    indices_to_keep = []
    for i, kp in enumerate(keypoints):
        x, y = kp.pt
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)

        if radius is None or distance <= radius:
            indices_to_keep.append(i)

    if len(indices_to_keep) == 0:
        return keypoints, descriptors

    # Filter keypoints and descriptors
    filtered_keypoints = [keypoints[i] for i in indices_to_keep]
    if descriptors is not None:
        filtered_descriptors = descriptors[indices_to_keep]
    else:
        filtered_descriptors = None

    return filtered_keypoints, filtered_descriptors


def detect_and_compute_features(image, mask=None):
    """
    Detect ORB keypoints and compute descriptors.

    Args:
        image: Input image (BGR)
        mask: Optional binary mask (white = detect features, black = ignore)

    Returns:
        tuple: (keypoints, descriptors)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Apply mask if provided
    if mask is not None:
        # Ensure mask is same size as image
        if mask.shape != gray.shape:
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))
        # Convert mask to binary
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    else:
        mask = None

    # Detect keypoints and compute descriptors
    orb = create_orb_detector()
    keypoints, descriptors = orb.detectAndCompute(gray, mask)

    # Filter by center distance if enabled
    if USE_CENTER_FEATURES:
        keypoints, descriptors = filter_keypoints_by_center_distance(
            keypoints, descriptors, h, w, USE_CENTER_FEATURES, CENTER_FEATURE_RADIUS
        )

    return keypoints, descriptors


def match_features(descriptors1, descriptors2):
    """
    Match ORB descriptors between two frames using Brute Force matcher.

    Args:
        descriptors1: Descriptors from frame 1
        descriptors2: Descriptors from frame 2

    Returns:
        tuple: (match_count, good_matches)
    """
    if descriptors1 is None or descriptors2 is None:
        return 0, []

    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return 0, []

    # Brute force matcher with Hamming distance (for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort by distance (lower is better)
    matches = sorted(matches, key=lambda x: x.distance)

    return len(matches), matches


def calculate_match_ratio(keypoints1, keypoints2, matches):
    """
    Calculate match ratio: (good matches) / (avg keypoints in both frames)

    Args:
        keypoints1: Keypoints from frame 1
        keypoints2: Keypoints from frame 2
        matches: List of good matches

    Returns:
        float: Match ratio (0-1)
    """
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        return 0.0

    avg_keypoints = (len(keypoints1) + len(keypoints2)) / 2.0
    match_ratio = len(matches) / avg_keypoints if avg_keypoints > 0 else 0.0

    return match_ratio


def is_duplicate_frame(current_keypoints, current_descriptors, history,
                       match_ratio_threshold, min_match_count,
                       image_h=None, image_w=None):
    """
    Compare current frame against saved history using feature matching.
    If duplicate found, compare center distances and return which should be kept.

    Args:
        current_keypoints: Keypoints from current frame
        current_descriptors: Descriptors from current frame
        history: List of dicts with frame data
        match_ratio_threshold: Threshold for duplicate detection
        min_match_count: Minimum matches required
        image_h: Image height (for center distance calculation)
        image_w: Image width (for center distance calculation)

    Returns:
        tuple: (is_duplicate, details)
    """
    if not history:
        return False, {
            "matched_frame": None,
            "match_count": 0,
            "match_ratio": 0.0,
            "kp_count": len(current_keypoints),
            "keep_current": True,
            "reason": "no_history"
        }

    if current_descriptors is None or len(current_keypoints) < 3:
        return False, {
            "matched_frame": None,
            "match_count": 0,
            "match_ratio": 0.0,
            "kp_count": len(current_keypoints),
            "keep_current": True,
            "reason": "no_descriptors"
        }

    best_match_count = 0
    best_match_ratio = 0.0
    best_match_frame = None
    best_match_idx = None

    for idx, record in enumerate(history):
        match_count, matches = match_features(
            current_descriptors,
            record["descriptors"]
        )

        if match_count < min_match_count:
            continue

        match_ratio = calculate_match_ratio(
            current_keypoints,
            record["keypoints"],
            matches
        )

        if match_ratio > best_match_ratio:
            best_match_ratio = match_ratio
            best_match_count = match_count
            best_match_frame = record["filename"]
            best_match_idx = idx

        # Early exit if we found a duplicate
        if best_match_ratio >= match_ratio_threshold:
            break

    is_dup = best_match_ratio >= match_ratio_threshold

    # If duplicate found, compare center distances
    keep_current = True
    reason = "unique"

    if is_dup and USE_CENTER_FEATURES and image_h is not None and image_w is not None:
        current_center_dist = calculate_avg_center_distance(
            current_keypoints, image_h, image_w
        )
        history_center_dist = history[best_match_idx].get("center_distance", float('inf'))

        # Keep the frame with features closer to center (lower distance is better)
        if history_center_dist < current_center_dist:
            keep_current = False
            reason = f"matched_frame_closer_to_center ({history_center_dist:.0f}px vs {current_center_dist:.0f}px)"
        else:
            reason = f"current_frame_closer_to_center ({current_center_dist:.0f}px vs {history_center_dist:.0f}px)"
    elif is_dup:
        reason = "matched_duplicate"

    return is_dup, {
        "matched_frame": best_match_frame,
        "match_count": best_match_count,
        "match_ratio": float(best_match_ratio),
        "kp_count": len(current_keypoints),
        "keep_current": keep_current,
        "reason": reason
    }


def deduplicate_orb_features(input_folder, output_folder, mask_path=None,
                             n_features=N_FEATURES,
                             comparison_window=COMPARISON_WINDOW,
                             min_match_count=MIN_MATCH_COUNT,
                             use_gpu=USE_GPU):
    """
    Deduplicate frames using ORB feature matching with ±N frame comparison window.

    Uses two-pass approach:
    1. Extract features from all frames
    2. Compare each frame with ±N surrounding frames

    Args:
        input_folder: Path to folder containing images
        output_folder: Path to save unique frames
        mask_path: Optional path to binary mask image
        n_features: Number of ORB features to detect
        match_ratio_threshold: Threshold for duplicate detection
        comparison_window: Compare frame with ±N frames (e.g., 3 = frame 10 compares with 7-13)
        min_match_count: Minimum matches required
        use_gpu: Use GPU acceleration where available
    """

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Load mask if provided
    mask = None
    if mask_path:
        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask file not found: {mask_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not load mask: {mask_path}")
        print(f"✓ Mask loaded: {mask_path} (shape: {mask.shape})")

    # Get all image files
    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if not image_files:
        raise ValueError(f"No image files found in {input_folder}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    print("\n" + "=" * 70)
    print("ORB FEATURE-BASED DEDUPLICATION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input folder: {input_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  N ORB features: {n_features}")
    print(f"  Min match count (ONLY criterion): {min_match_count}")
    print(f"  Comparison window: ±{comparison_window} frames")
    print(f"  GPU enabled: {use_gpu}")
    print(f"  Region mask: {'Yes' if mask is not None else 'No (full frame)'}")
    print(f"  Total frames: {len(image_files)}")
    print(f"\n" + "-" * 70)
    print(f"PASS 1: Extracting features from all frames...")
    print("-" * 70 + "\n")

    # PASS 1: Load all images and extract features
    all_frames = []
    for idx, filename in enumerate(image_files):
        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)

        if image is None:
            print(f"⚠️  Frame {idx+1:5d}/{len(image_files)} | SKIPPED (unreadable): {filename}")
            all_frames.append({
                "filename": filename,
                "image": None,
                "keypoints": None,
                "descriptors": None,
                "status": "skipped",
                "reason": "unreadable"
            })
            continue

        # Detect features
        keypoints, descriptors = detect_and_compute_features(image, mask)

        if descriptors is None or len(keypoints) < 3:
            all_frames.append({
                "filename": filename,
                "image": image,
                "keypoints": None,
                "descriptors": None,
                "status": "skipped",
                "reason": "no_features"
            })
            continue

        # Calculate center distance
        center_dist = calculate_avg_center_distance(
            keypoints, image.shape[0], image.shape[1]
        )

        all_frames.append({
            "filename": filename,
            "image": image,
            "keypoints": keypoints,
            "descriptors": descriptors,
            "status": "candidate",
            "center_distance": center_dist,
            "kp_count": len(keypoints)
        })

        if (idx + 1) % 100 == 0:
            print(f"   Extracted: {idx + 1}/{len(image_files)} frames")

    print(f"✓ Feature extraction complete: {len(all_frames)} frames loaded\n")

    # PASS 2: Deduplication with ±N frame window
    print("-" * 70)
    print(f"PASS 2: Deduplication (±{comparison_window} frame window)...")
    print("-" * 70 + "\n")

    unique_count = 0
    duplicate_count = 0
    skipped_count = 0
    processed_count = 0
    metadata = []
    to_keep = [False] * len(all_frames)  # Track which frames to keep

    for idx, frame_data in enumerate(all_frames):
        if frame_data["status"] == "skipped":
            skipped_count += 1
            metadata.append({
                "filename": frame_data["filename"],
                "status": "skipped",
                "reason": frame_data["reason"]
            })
            continue

        processed_count += 1
        current_kp = frame_data["keypoints"]
        current_desc = frame_data["descriptors"]
        current_filename = frame_data["filename"]

        if current_kp is None or current_desc is None:
            continue

        # Get comparison window: current frame ± comparison_window frames
        start_idx = max(0, idx - comparison_window)
        end_idx = min(len(all_frames), idx + comparison_window + 1)
        comparison_frames = all_frames[start_idx:end_idx]

        # Check against all frames in the window
        is_dup = False
        best_match = None
        best_details = None

        for comp_idx, comp_frame in enumerate(comparison_frames):
            if comp_frame["status"] == "skipped" or comp_frame["keypoints"] is None:
                continue

            actual_idx = start_idx + comp_idx
            if actual_idx == idx:  # Skip comparing with itself
                continue

            match_count, matches = match_features(
                current_desc,
                comp_frame["descriptors"]
            )

            if match_count < min_match_count:
                continue

            # Only check match count (ratio threshold removed)
            if match_count >= min_match_count:
                is_dup = True
                best_match = (actual_idx, comp_frame, match_count)
                break  # Found duplicate

        # Decide whether to keep this frame
        if is_dup and best_match:
            matched_idx, matched_frame, match_count = best_match
            current_center = frame_data["center_distance"]
            matched_center = matched_frame["center_distance"]

            # Keep frame with features closer to center
            if current_center <= matched_center:
                to_keep[idx] = True
                reason = f"KEEP (closer to center: {current_center:.0f}px vs {matched_center:.0f}px)"
                print(
                    f"✓ Frame {idx+1:5d} | {current_filename} - {reason}"
                )
            else:
                to_keep[idx] = False
                duplicate_count += 1
                reason = f"matched frame {matched_idx+1} is closer to center"
                print(
                    f"✗ Frame {idx+1:5d} | {current_filename} - DUPLICATE ({reason})"
                )
                metadata.append({
                    "filename": current_filename,
                    "status": "duplicate",
                    "matched_to": matched_frame["filename"],
                    "match_count": match_count
                })
                continue
        else:
            to_keep[idx] = True

        # Save unique frame
        unique_count += 1
        output_path = os.path.join(output_folder, frame_data["filename"])
        cv2.imwrite(output_path, frame_data["image"])

        metadata.append({
            "filename": current_filename,
            "status": "saved",
            "kp_count": frame_data["kp_count"],
            "center_distance": frame_data["center_distance"]
        })

        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{len(all_frames)} processed, {unique_count} unique")

    # Calculate statistics
    dedup_percentage = (duplicate_count / processed_count * 100) if processed_count > 0 else 0.0

    print("\n" + "=" * 70)
    print(f"DEDUPLICATION COMPLETE (±{comparison_window} Frame Bidirectional Window)")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Total frames processed: {processed_count}")
    print(f"  Unique frames saved: {unique_count}")
    print(f"  Duplicate frames skipped: {duplicate_count}")
    print(f"  Unreadable/no-feature frames: {skipped_count}")
    print(f"  Reduction rate: {dedup_percentage:.1f}%")
    print(f"  Center-focus enabled: {USE_CENTER_FEATURES}")
    print(f"  Comparison window: ±{comparison_window} frames (total {2*comparison_window + 1} frames per comparison)")
    print(f"  GPU acceleration: {'Enabled' if use_gpu else 'Disabled'}")
    print(f"\n📁 Unique frames saved to: {output_folder}")

    # Save metadata
    metadata_path = os.path.join(output_folder, "dedup_metadata_orb.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "input_folder": input_folder,
            "output_folder": output_folder,
            "method": "ORB Feature Matching (MIN_MATCH_COUNT only, no ratio threshold)",
            "n_features": n_features,
            "min_match_count": min_match_count,
            "comparison_window": comparison_window,
            "total_window_size": 2 * comparison_window + 1,
            "center_focus_enabled": USE_CENTER_FEATURES,
            "center_feature_radius": CENTER_FEATURE_RADIUS,
            "gpu_enabled": use_gpu,
            "region_mask": mask_path if mask_path else None,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_processed": processed_count,
                "unique_saved": unique_count,
                "duplicates_skipped": duplicate_count,
                "unreadable_skipped": skipped_count,
                "reduction_percentage": dedup_percentage
            },
            "frames": metadata
        }, f, indent=2)

    print(f"📄 Metadata saved to: {metadata_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deduplicate frames using ORB feature matching (robust to lighting, vibration, compression)",
        epilog="""
Examples:
  python3 deduplicate_orb_features.py /path/to/frames
  python3 deduplicate_orb_features.py /path/to/frames --output /custom/output
  python3 deduplicate_orb_features.py /path/to/frames --match-ratio 0.20 --n-features 500
  python3 deduplicate_orb_features.py /path/to/frames --match-ratio 0.75 --history 15

CONFIGURATION (edit in script):
  - MASK_PATH: Set to your mask image path, or None for full frame
  - N_FEATURES: Number of ORB features (default: 500)
  - MATCH_RATIO_THRESHOLD: Duplicate threshold (default: 0.15)
  - MAX_HISTORY: Compare against last N frames (default: 15)

REGION MASK:
  - Binary image (white=255, black=0)
  - Same size as input frames
  - White regions: extract features from here
  - Black regions: ignore
  - Example: focus on defect areas, ignore rail edges

MATCH RATIO GUIDE:
  - 0.10 = loose, may over-deduplicate
  - 0.15 = moderate, catch most duplicates
  - 0.20 = strict, only similar frames
  - 0.25+ = very strict, only nearly identical
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input_folder", help="Folder containing frames (jpg/png/bmp)")
    parser.add_argument("--output", default=None,
                        help="Output folder (default: {input_folder}_dedup_orb)")
    parser.add_argument("--n-features", type=int, default=N_FEATURES,
                        help=f"Number of ORB features to detect (default: {N_FEATURES})")
    parser.add_argument("--window", type=int, default=COMPARISON_WINDOW,
                        help=f"Comparison window: ±N frames (default: {COMPARISON_WINDOW}, e.g., 4 = frames for frame at index)")
    parser.add_argument("--min-matches", type=int, default=MIN_MATCH_COUNT,
                        help=f"Minimum feature matches required for duplicate detection (ONLY criterion, default: {MIN_MATCH_COUNT})")
    parser.add_argument("--gpu", action="store_true", default=USE_GPU,
                        help="Enable GPU acceleration (default: enabled)")

    args = parser.parse_args()

    input_folder = args.input_folder.rstrip("/")
    parent_folder = os.path.dirname(input_folder)
    base_name = os.path.basename(input_folder)
    output_folder = args.output or os.path.join(parent_folder, f"{base_name}_dedup_orb")

    deduplicate_orb_features(
        input_folder,
        output_folder,
        mask_path=MASK_PATH,
        n_features=args.n_features,
        comparison_window=args.window,
        min_match_count=args.min_matches,
        use_gpu=args.gpu
    )
