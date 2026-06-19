#!/usr/bin/env python3
"""
Pixel-level deduplication using SSIM (Structural Similarity Index).

Compares frames at the pixel level using SSIM instead of bounding box IoU.
This detects visually similar/duplicate frames regardless of detection accuracy.

========== HOW TO RUN ==========

1. Basic usage (deduplicate all frames):
   python3 deduplicate_pixel_level.py /path/to/input_folder

2. Custom output folder:
   python3 deduplicate_pixel_level.py /path/to/input_folder --output /path/to/output

3. Adjust SSIM threshold (higher = stricter deduplication):
   python3 deduplicate_pixel_level.py /path/to/input_folder --ssim-threshold 0.90

4. Adjust comparison window (±N frames):
   python3 deduplicate_pixel_level.py /path/to/input_folder --window 3

5. With region mask (compare only specific area):
   python3 deduplicate_pixel_level.py /path/to/input_folder --mask /path/to/mask.png

6. Combine options:
   python3 deduplicate_pixel_level.py /path/to/input_folder \
     --ssim-threshold 0.85 \
     --window 4 \
     --mask /path/to/mask.png \
     --output /custom/output

REGION MASK CONFIGURATION:
  - Set MASK_PATH in script configuration (line ~51)
  - Set to None to disable masking (full frame analysis)
  - Binary mask image (same size as input frames)
  - White (255) = region to compare
  - Black (0) = ignore region
  - Example: mask out rail edges, focus on defect comparison only

Output:
  - Deduplicated frames saved to output folder
  - dedup_metadata_pixel.json with statistics and frame-by-frame analysis

Note:
  - SSIM = 1.0 means identical frames
  - SSIM = 0.0 means completely different frames
  - Threshold 0.85+ = very similar (good for duplicates)
  - Masking reduces noise from non-ROI areas
"""

import cv2
import os
import json
import numpy as np
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
import argparse

# ==========================================================================
# CONFIGURATION - Edit these as needed
# ==========================================================================
MASK_PATH = r"/media/viraj/New Volume/Dhrishti/Detection_code/Mask.png"  # Path to region mask (set to None to disable masking)
SSIM_THRESHOLD = 0.92      # SSIM threshold to mark as duplicate (0-1, higher = stricter)
COMPARISON_WINDOW = 3       # Compare with ±N frames (3 = 7 total frames: -3,-2,-1,0,+1,+2,+3)


def load_mask(mask_path=None):
    """Load and prepare mask if provided."""
    if mask_path is None or not os.path.isfile(mask_path):
        return None
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    return mask


def apply_mask_to_image(image, mask):
    """Apply mask to image (set masked regions to 0)."""
    if mask is None:
        return image
    if mask.shape != image.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    masked_image = image.copy()
    masked_image[mask == 0] = 0
    return masked_image


def calculate_ssim(img1, img2, mask=None):
    """
    Calculate Structural Similarity Index (SSIM) between two images.

    Args:
        img1: numpy array (BGR image from OpenCV)
        img2: numpy array (BGR image from OpenCV)
        mask: Optional binary mask (white = compare, black = ignore)

    Returns:
        float: SSIM score between -1 and 1 (1 = identical)
    """
    # Convert to grayscale for SSIM calculation
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Ensure same dimensions
    if gray1.shape != gray2.shape:
        h, w = min(gray1.shape[0], gray2.shape[0]), min(gray1.shape[1], gray2.shape[1])
        gray1 = gray1[:h, :w]
        gray2 = gray2[:h, :w]

    # Apply mask if provided
    if mask is not None:
        gray1 = apply_mask_to_image(gray1, mask)
        gray2 = apply_mask_to_image(gray2, mask)

    # Calculate SSIM
    score = ssim(gray1, gray2, data_range=255)
    return score


def is_duplicate_frame(current_idx, current_image, frame_data, ssim_threshold,
                       comparison_window, mask=None):
    """
    Compare current frame against frames in ±window using SSIM.

    Args:
        current_idx: Index of current frame
        current_image: Current frame (numpy array)
        frame_data: List of dicts with frame data (indexed by position)
        ssim_threshold: SSIM threshold for marking as duplicate
        comparison_window: Number of frames to check before/after (±N)
        mask: Optional binary mask for region restriction

    Returns:
        tuple: (is_duplicate, details)
    """
    best_ssim = 0.0
    best_match_idx = None
    best_match_filename = None

    # Calculate window boundaries
    start_idx = max(0, current_idx - comparison_window)
    end_idx = min(len(frame_data), current_idx + comparison_window + 1)

    # Compare against all frames in window (except current)
    for idx in range(start_idx, end_idx):
        if idx == current_idx:
            continue

        if frame_data[idx] is None:
            continue

        score = calculate_ssim(current_image, frame_data[idx]["image"], mask=mask)
        if score > best_ssim:
            best_ssim = score
            best_match_idx = idx
            best_match_filename = frame_data[idx]["filename"]

        # Early exit if we found a duplicate
        if best_ssim >= ssim_threshold:
            break

    is_dup = best_ssim >= ssim_threshold
    return is_dup, {
        "matched_idx": best_match_idx,
        "matched_frame": best_match_filename,
        "ssim": float(best_ssim)
    }


def deduplicate_pixel_level(input_folder, output_folder,
                            ssim_threshold=SSIM_THRESHOLD,
                            comparison_window=COMPARISON_WINDOW,
                            mask_path=MASK_PATH):
    """
    Deduplicate frames using pixel-level SSIM comparison with sliding window.

    Args:
        input_folder: Path to folder containing images
        output_folder: Path to save unique frames
        ssim_threshold: SSIM threshold for duplicate detection
        comparison_window: Number of frames before/after to compare (±N)
        mask_path: Optional path to region mask
    """

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Get all image files
    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ])

    if not image_files:
        raise ValueError(f"No image files found in {input_folder}")

    # Load mask if provided
    mask = load_mask(mask_path)
    mask_enabled = mask is not None

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    print("\n" + "=" * 70)
    print("PIXEL-LEVEL DEDUPLICATION (SSIM-based with Sliding Window)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Input folder: {input_folder}")
    print(f"  Output folder: {output_folder}")
    print(f"  SSIM threshold: {ssim_threshold}")
    print(f"  Comparison window: ±{comparison_window} frames")
    print(f"  Region masking: {'✓ Enabled' if mask_enabled else '✗ Disabled'}")
    if mask_enabled:
        print(f"  Mask path: {mask_path}")
    print(f"  Total frames: {len(image_files)}")
    print(f"\n" + "-" * 70)
    print("PASS 1: Loading all frames...")
    print("-" * 70 + "\n")

    # PASS 1: Load all frames into memory
    frame_data = []
    skipped_count = 0

    for idx, filename in enumerate(image_files):
        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)

        if image is None:
            print(f"⚠️  Frame {idx+1:5d}/{len(image_files)} | SKIPPED (unreadable): {filename}")
            frame_data.append(None)
            skipped_count += 1
            continue

        frame_data.append({
            "filename": filename,
            "image": image.copy()
        })

        if (idx + 1) % 100 == 0:
            print(f"   Loaded {idx + 1}/{len(image_files)} frames")

    print(f"\n✓ Pass 1 complete: Loaded {len(image_files) - skipped_count}/{len(image_files)} frames")
    print(f"\n" + "-" * 70)
    print("PASS 2: Deduplicating with sliding window...")
    print("-" * 70 + "\n")

    unique_count = 0
    duplicate_count = 0
    processed_count = 0
    metadata = []

    for idx, frame_info in enumerate(frame_data):
        if frame_info is None:
            metadata.append({
                "filename": image_files[idx],
                "status": "skipped",
                "reason": "unreadable"
            })
            continue

        processed_count += 1
        filename = frame_info["filename"]

        # Check for duplicates against frames in ±window
        is_dup, details = is_duplicate_frame(
            idx,
            frame_info["image"],
            frame_data,
            ssim_threshold,
            comparison_window,
            mask=mask
        )

        if is_dup:
            duplicate_count += 1
            print(
                f"✗ Frame {idx+1:5d}/{len(image_files)} | DUPLICATE: {filename} "
                f"(SSIM={details['ssim']:.3f}, matched_to={details['matched_frame']})"
            )
            metadata.append({
                "filename": filename,
                "status": "duplicate",
                "matched_idx": details["matched_idx"],
                "matched_to": details["matched_frame"],
                "ssim": details["ssim"]
            })
            continue

        # Save unique frame
        unique_count += 1
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, frame_info["image"])

        print(f"✓ Frame {idx+1:5d}/{len(image_files)} | SAVED unique frame: {filename}")
        metadata.append({
            "filename": filename,
            "status": "saved"
        })

        # Print progress every 100 frames
        if (idx + 1) % 100 == 0:
            print(f"   Progress: {idx + 1}/{len(image_files)}, {unique_count} unique, {duplicate_count} duplicates")

    # Calculate statistics
    dedup_percentage = (duplicate_count / processed_count * 100) if processed_count > 0 else 0.0

    print("\n" + "=" * 70)
    print("DEDUPLICATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Total frames processed: {processed_count}")
    print(f"  Unique frames saved: {unique_count}")
    print(f"  Duplicate frames skipped: {duplicate_count}")
    print(f"  Unreadable frames skipped: {skipped_count}")
    print(f"  Reduction rate: {dedup_percentage:.1f}%")
    print(f"\n📁 Unique frames saved to: {output_folder}")

    # Save metadata
    metadata_path = os.path.join(output_folder, "dedup_metadata_pixel.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "input_folder": input_folder,
            "output_folder": output_folder,
            "method": "SSIM (Structural Similarity Index) - Sliding Window ±N",
            "ssim_threshold": ssim_threshold,
            "comparison_window": comparison_window,
            "resize_enabled": False,
            "masking_enabled": mask_enabled,
            "mask_path": mask_path if mask_enabled else None,
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
        description="Deduplicate frames using pixel-level SSIM (Structural Similarity Index) with sliding window",
        epilog="""
Examples:
  python3 deduplicate_pixel_level.py /path/to/frames
  python3 deduplicate_pixel_level.py /path/to/frames --output /custom/output
  python3 deduplicate_pixel_level.py /path/to/frames --ssim-threshold 0.90 --window 3
  python3 deduplicate_pixel_level.py /path/to/frames --ssim-threshold 0.85 --mask /path/to/mask.png
  python3 deduplicate_pixel_level.py /path/to/frames --ssim-threshold 0.85 --window 4 --mask /path/to/mask.png

SSIM Threshold Guidelines:
  0.95+ = Extremely strict (only nearly identical frames)
  0.85-0.95 = Strict (catches close duplicates, recommended)
  0.70-0.85 = Moderate (catches similar frames)
  <0.70 = Very loose (may remove non-duplicates)

Window Explanation:
  --window 3 compares frame N against frames [N-3, N-2, N-1, N+1, N+2, N+3]
  Higher window = more comparisons, slower but more accurate
  Lower window = faster but may miss duplicates

Masking:
  White (255) = region to compare
  Black (0) = region to ignore
  Useful for ignoring constant backgrounds
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("input_folder", help="Folder containing frames (jpg/png/bmp)")
    parser.add_argument("--output", default=None,
                        help="Output folder (default: {input_folder}_dedup_pixel)")
    parser.add_argument("--ssim-threshold", type=float, default=SSIM_THRESHOLD,
                        help=f"SSIM threshold for duplicate detection 0-1 (default: {SSIM_THRESHOLD})")
    parser.add_argument("--window", type=int, default=COMPARISON_WINDOW,
                        help=f"Comparison window: ±N frames (default: {COMPARISON_WINDOW})")
    parser.add_argument("--mask", default=MASK_PATH,
                        help="Optional region mask image (white=compare, black=ignore)")

    args = parser.parse_args()

    input_folder = args.input_folder.rstrip("/")
    parent_folder = os.path.dirname(input_folder)
    base_name = os.path.basename(input_folder)
    output_folder = args.output or os.path.join(parent_folder, f"{base_name}_dedup_pixel")

    deduplicate_pixel_level(
        input_folder,
        output_folder,
        ssim_threshold=args.ssim_threshold,
        comparison_window=args.window,
        mask_path=args.mask
    )
