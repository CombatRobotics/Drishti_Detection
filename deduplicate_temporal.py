#!/usr/bin/env python3
"""
Technique: temporal fault deduplication using YOLO re-detection.

This script deduplicates repeated rail-fault frames by re-running YOLO on each
candidate frame and comparing the highest-confidence detection against the last
saved unique frame. If the current detection overlaps the previous unique
frame with IoU above the threshold and occurs within the temporal duplicate
window, it is treated as a duplicate.

This preserves one representative frame per temporal fault event.
"""

import cv2
import os
import json
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION - Edit these as needed
# ============================================================================
MODEL_PATH = r"/media/viraj/New Volume/Dhrishti/YOLO/v4.1/faultdetection_v4.1.pt"
BBOX_OVERLAP_THRESHOLD = 0.85  # Minimum IoU to consider a detection a duplicate
CONFIDENCE_THRESHOLD = 0.425   # YOLO confidence threshold for re-detection
MAX_FRAME_GAP = 90             # Maximum frame distance to treat detections as temporal duplicates


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union (IoU) for two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    intersection_w = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    intersection_h = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection_area = intersection_w * intersection_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)
    union_area = area1 + area2 - intersection_area

    return intersection_area / union_area if union_area > 0 else 0.0


def extract_best_bbox(result):
    """Return the highest-confidence bounding box from a YOLO result."""
    boxes = result.boxes
    if len(boxes) == 0:
        return None
    best_index = int(boxes.conf.argmax().item())
    return boxes.xyxy[best_index].cpu().numpy().tolist()


def deduplicate_temporal(input_folder, model_path, output_folder,
                          bbox_threshold=BBOX_OVERLAP_THRESHOLD,
                          conf_threshold=CONFIDENCE_THRESHOLD,
                          max_frame_gap=MAX_FRAME_GAP):
    """Deduplicate detected fault frames using temporal overlap."""
    print("\n" + "=" * 70)
    print("TEMPORAL FAULT DEDUPLICATION")
    print("=" * 70)

    if not os.path.isdir(input_folder):
        print(f"❌ Error: Input folder not found: {input_folder}")
        return
    if not os.path.isfile(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        return

    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    if not image_files:
        print(f"❌ Error: No image files found in {input_folder}")
        return

    print(f"\n📁 Input folder: {input_folder}")
    print(f"📁 Output folder: {output_folder}")
    print(f"🤖 Model: {os.path.basename(model_path)}")
    print(f"📊 Total images to process: {len(image_files)}")
    print(f"⚙️  IoU threshold: {bbox_threshold}")
    print(f"⚙️  Confidence threshold: {conf_threshold}")
    print(f"⚙️  Max frame gap: {max_frame_gap}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    print(f"\n✓ Model loaded on {device}")

    os.makedirs(output_folder, exist_ok=True)

    last_saved_bbox = None
    last_saved_frame = None
    unique_count = 0
    duplicate_count = 0
    skipped_count = 0
    processed_count = 0
    metadata = []

    for idx, filename in enumerate(image_files, start=1):
        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)
        if image is None:
            print(f"⚠️  Frame {idx:4d}/{len(image_files)} | SKIPPED (unreadable): {filename}")
            skipped_count += 1
            metadata.append({"filename": filename, "status": "skipped", "reason": "unreadable"})
            continue

        processed_count += 1
        results = model(image, conf=conf_threshold, device=device)
        bbox = extract_best_bbox(results[0])

        if bbox is None:
            print(f"⚠️  Frame {idx:4d}/{len(image_files)} | SKIPPED (no detections): {filename}")
            skipped_count += 1
            metadata.append({"filename": filename, "status": "skipped", "reason": "no_detection"})
            continue

        frame_gap = idx - last_saved_frame if last_saved_frame is not None else max_frame_gap + 1
        iou_score = 0.0
        is_duplicate = False

        if last_saved_bbox is not None and frame_gap <= max_frame_gap:
            iou_score = calculate_iou(last_saved_bbox, bbox)
            is_duplicate = iou_score >= bbox_threshold

        if is_duplicate:
            duplicate_count += 1
            print(f"✗ Frame {idx:4d}/{len(image_files)} | DUPLICATE (IoU={iou_score:.3f}, gap={frame_gap}) | {filename}")
            metadata.append({
                "filename": filename,
                "status": "duplicate",
                "iou": iou_score,
                "frame_gap": frame_gap
            })
        else:
            unique_count += 1
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, image)
            last_saved_bbox = bbox
            last_saved_frame = idx
            print(f"✓ Frame {idx:4d}/{len(image_files)} | UNIQUE | saved {filename} | IoU={iou_score:.3f}")
            metadata.append({
                "filename": filename,
                "status": "saved",
                "iou": iou_score,
                "frame_gap": frame_gap
            })

    print("\n" + "=" * 70)
    print("DEDUPLICATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Total images processed: {processed_count}")
    print(f"  Unique frames saved: {unique_count}")
    print(f"  Duplicate frames skipped: {duplicate_count}")
    print(f"  Skipped frames: {skipped_count}")
    print(f"  Reduction rate: {(duplicate_count / processed_count * 100) if processed_count else 0:.1f}%")
    print(f"\n📁 Output folder: {output_folder}")

    metadata_path = os.path.join(output_folder, "deduplication_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "input_folder": input_folder,
            "output_folder": output_folder,
            "model": model_path,
            "iou_threshold": bbox_threshold,
            "confidence_threshold": conf_threshold,
            "max_frame_gap": max_frame_gap,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_processed": processed_count,
                "unique_saved": unique_count,
                "duplicates_skipped": duplicate_count,
                "skipped": skipped_count,
                "reduction_rate": (duplicate_count / processed_count * 100) if processed_count else 0
            },
            "frames": metadata
        }, f, indent=2)

    print(f"📄 Metadata saved to: {metadata_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate detected fault frames using temporal overlap and YOLO re-detection"
    )
    parser.add_argument("input_folder", help="Path to detected frames folder")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO model path")
    parser.add_argument("--output", default=None, help="Output folder path")
    parser.add_argument("--threshold", "-t", type=float, default=BBOX_OVERLAP_THRESHOLD,
                        help="IoU threshold for duplicate matching")
    parser.add_argument("--confidence", "-c", type=float, default=CONFIDENCE_THRESHOLD,
                        help="YOLO confidence threshold")
    parser.add_argument("--gap", "-g", type=int, default=MAX_FRAME_GAP,
                        help="Maximum frame gap for duplicate matching")

    args = parser.parse_args()

    input_folder = args.input_folder.rstrip("/")
    parent_folder = os.path.dirname(input_folder)
    base_name = os.path.basename(input_folder)
    output_folder = args.output or os.path.join(parent_folder, f"{base_name}_dedup_temporal")

    deduplicate_temporal(
        input_folder,
        args.model,
        output_folder,
        bbox_threshold=args.threshold,
        conf_threshold=args.confidence,
        max_frame_gap=args.gap
    )
