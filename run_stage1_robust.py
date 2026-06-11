"""
Stage 1 Runner (Robust) — processes a directory of images with robust line fitting.

Usage:
  python3 run_stage1_robust.py /home/viraj/faults/
  python3 run_stage1_robust.py /home/viraj/faults/ --workers 4 --max_frames 100
"""

import cv2
import json
import os
import sys
import argparse
from datetime import datetime
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(__file__))
from stage1_deskew_robust import process_frame

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "output_robust")
JPEG_QUALITY = 95
# ─────────────────────────────────────────────────────────────────────────────


def get_default_output_root(input_path):
    parent = os.path.dirname(input_path.rstrip("/"))
    base = os.path.basename(input_path.rstrip("/"))
    return os.path.join(parent, f"{base}_stage1_robust")


def make_run_dir(output_root=None):
    if output_root is None:
        output_root = OUTPUT_BASE
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_root, f"run_{ts}")
    frames_dir = os.path.join(run_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    return run_dir, frames_dir


def run_single_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    output_root = os.path.dirname(image_path)
    run_dir, frames_dir = make_run_dir(output_root)
    meta_path = os.path.join(run_dir, "metadata.jsonl")

    print(f"\n{'='*55}")
    print(f"  Stage 1 — Rail Deskew (Robust, single image)")
    print(f"{'='*55}")
    print(f"  Input   : {os.path.basename(image_path)}")
    print(f"  Output  : {run_dir}")
    print(f"{'='*55}\n")

    deskewed, meta = process_frame(frame)
    meta["source_image"] = image_path

    if deskewed is not None:
        fname = os.path.basename(image_path)
        fpath = os.path.join(frames_dir, fname)
        cv2.imwrite(fpath, deskewed, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        meta["saved_as"] = fname
        print(f"  QC PASS — saved: {fpath}")
    else:
        meta["saved_as"] = None
        print(f"  QC FAIL — {meta['fail_reason']}")

    with open(meta_path, "w") as f:
        f.write(json.dumps(meta, indent=2) + "\n")

    print(f"\n  Metadata:")
    for k, v in meta.items():
        print(f"    {k:<30} {v}")
    print(f"\n  Log: {meta_path}")
    print(f"{'='*55}\n")


def _process_image_worker(args):
    """
    Worker function for parallel processing. Runs in a separate process.
    Returns (metadata_dict, success).
    """
    img_path, frames_dir, jpeg_quality = args
    fname = os.path.basename(img_path)

    frame = cv2.imread(img_path)
    if frame is None:
        return {"source_image": img_path, "saved_as": None, "fail_reason": "unreadable"}, False

    deskewed, meta = process_frame(frame)
    meta["source_image"] = img_path

    if deskewed is not None:
        try:
            cv2.imwrite(os.path.join(frames_dir, fname), deskewed,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
            meta["saved_as"] = fname
            return meta, True
        except Exception as e:
            meta["saved_as"] = None
            meta["fail_reason"] = f"write_error: {str(e)}"
            return meta, False
    else:
        meta["saved_as"] = None
        return meta, False


def run_directory(directory_path, max_frames=None, num_workers=None, output_root=None):
    """
    Process all .jpg images in a directory using multiprocessing.
    """
    if not os.path.isdir(directory_path):
        print(f"ERROR: Directory not found: {directory_path}")
        sys.exit(1)

    # Find all JPEG files
    image_files = sorted([
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ])

    if not image_files:
        print(f"ERROR: No JPEG images found in: {directory_path}")
        sys.exit(1)

    if max_frames:
        image_files = image_files[:max_frames]

    if num_workers is None:
        num_workers = cpu_count()

    if output_root is None:
        output_root = get_default_output_root(directory_path)
    run_dir, frames_dir = make_run_dir(output_root)
    meta_path = os.path.join(run_dir, "metadata.jsonl")

    print(f"\n{'='*60}")
    print(f"  Stage 1 — Rail Deskew (Robust, Parallel)")
    print(f"{'='*60}")
    print(f"  Input dir  : {directory_path}")
    print(f"  Images     : {len(image_files)}")
    print(f"  Workers    : {num_workers} (CPU cores: {cpu_count()})")
    print(f"  Output     : {run_dir}")
    print(f"{'='*60}\n")

    # Prepare work items
    work_items = [
        (img_path, frames_dir, JPEG_QUALITY)
        for img_path in image_files
    ]

    n_pass = 0
    n_fail = 0

    # Process in parallel using worker pool
    with Pool(processes=num_workers) as pool:
        with open(meta_path, "w") as meta_f:
            for idx, (meta, success) in enumerate(
                pool.imap_unordered(_process_image_worker, work_items, chunksize=4),
                1
            ):
                if success:
                    n_pass += 1
                else:
                    n_fail += 1

                meta_f.write(json.dumps(meta) + "\n")

                if idx % 50 == 0:
                    pct = 100.0 * idx / len(image_files)
                    print(f"  [{idx:>5}/{len(image_files)}] {pct:5.1f}%  "
                          f"pass={n_pass}  fail={n_fail}")

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"  Processed : {len(image_files)}  |  pass={n_pass}  fail={n_fail}")
    print(f"  Success rate : {100.0 * n_pass / len(image_files):.1f}%")
    print(f"  Output    : {run_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1 Robust — Rail deskew runner")
    parser.add_argument("input_folder", help="Path to directory of images")
    parser.add_argument("--output", default=None,
                        help="Optional output root folder")
    parser.add_argument("--workers", type=int, default=None,
                        help=f"Number of parallel workers (default: all CPU cores)")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Stop after N frames (for testing)")
    args = parser.parse_args()

    output_root = args.output or get_default_output_root(args.input_folder)
    run_directory(args.input_folder, max_frames=args.max_frames,
                  num_workers=args.workers, output_root=output_root)
