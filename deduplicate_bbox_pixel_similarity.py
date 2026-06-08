#!/usr/bin/env python3
"""
Technique: temporal bbox pixel similarity deduplication.

This script deduplicates fault detection frames by comparing only the pixel
content inside the detected bounding box over a small temporal window.

It groups candidate frames within +/-5 frames, compares the normalized pixel
appearance inside each detection bbox, and chooses the single representative
frame that is closest to the image center from each duplicate group.

Detection confidence is explicitly ignored during duplicate comparison;
frame grouping is driven by bbox appearance and temporal proximity only.
"""

import cv2
import os
import json
import torch
import numpy as np
from ultralytics import YOLO
from datetime import datetime

# ==========================================================================
# CONFIGURATION - Edit these as needed
# ==========================================================================
MODEL_PATH = r"/media/viraj/New Volume/Dhrishti/YOLO/v4.1/faultdetection_v4.1.pt"
CONFIDENCE_THRESHOLD = 0.425   # YOLO confidence threshold for re-detection
PIXEL_SIMILARITY_THRESHOLD = 0.70  # Similarity score threshold for bbox crop match
TEMPORAL_WINDOW = 8            # Frames before/after to compare for duplicate grouping
CROP_SIZE = (128, 128)         # Resize bbox pixels before similarity comparison


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


def crop_and_normalize(frame, bbox):
    """Crop the bbox region, convert to grayscale, and resize to fixed size."""
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_resized = cv2.resize(crop_gray, CROP_SIZE, interpolation=cv2.INTER_AREA)
    return crop_resized


def pixel_similarity(crop_a, crop_b):
    """Compute a brightness-invariant similarity between two bbox crop images."""
    if crop_a is None or crop_b is None or crop_a.shape != crop_b.shape:
        return 0.0

    crop_a = crop_a.astype("float32")
    crop_b = crop_b.astype("float32")
    crop_a = (crop_a - crop_a.mean()) / (crop_a.std() + 1e-6)
    crop_b = (crop_b - crop_b.mean()) / (crop_b.std() + 1e-6)

    correlation = np.mean(crop_a * crop_b)
    similarity = (correlation + 1.0) / 2.0
    return max(0.0, min(1.0, similarity))


def bbox_center_distance(bbox, frame_width, frame_height):
    """Compute Euclidean distance from bbox center to image center."""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    fx = frame_width / 2.0
    fy = frame_height / 2.0
    return ((cx - fx) ** 2 + (cy - fy) ** 2) ** 0.5


def extract_primary_detection(result):
    """Return the largest detection bbox and class from YOLO result.

    Confidence is not used for duplicate grouping; we use the largest detected
    bbox as the representative defect region to reduce sensitivity to score
    variation across nearby frames.
    """
    boxes = result.boxes
    if len(boxes) == 0:
        return None, None, None

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()

    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    best_index = int(areas.argmax())
    bbox = xyxy[best_index].tolist()
    cls_id = int(classes[best_index])
    confidence = float(confidences[best_index])
    return bbox, cls_id, confidence


def group_duplicate_frames(frames, threshold, window):
    """Group frames into duplicate clusters using a small temporal window.

    Grouping is based on class and bbox pixel similarity only; confidence scores
    are ignored for duplicate decisions.
    """
    n = len(frames)
    visited = [False] * n
    groups = []

    for i in range(n):
        if visited[i] or frames[i]["crop"] is None:
            continue

        group = [i]
        visited[i] = True
        queue = [i]

        while queue:
            idx = queue.pop(0)
            for j in range(max(0, idx - window), min(n, idx + window + 1)):
                if visited[j] or j == idx or frames[j]["crop"] is None:
                    continue

                if frames[idx]["class"] != frames[j]["class"]:
                    continue

                sim = pixel_similarity(frames[idx]["crop"], frames[j]["crop"])
                if sim >= threshold:
                    visited[j] = True
                    group.append(j)
                    queue.append(j)

        groups.append(sorted(group))

    return groups


def deduplicate_bbox_pixel_similarity(input_folder, model_path, output_folder,
                                      similarity_threshold=PIXEL_SIMILARITY_THRESHOLD,
                                      temporal_window=TEMPORAL_WINDOW,
                                      confidence_threshold=CONFIDENCE_THRESHOLD):
    """Deduplicate frames using bbox pixel similarity in a temporal window."""
    print("\n" + "=" * 70)
    print("BBOX PIXEL SIMILARITY DEDUPLICATION")
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
    print(f"⚙️  Similarity threshold: {similarity_threshold}")
    print(f"⚙️  Temporal window: ±{temporal_window} frames")
    print(f"⚙️  Confidence threshold: {confidence_threshold}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    print(f"\n✓ Model loaded on {device}")

    frames = []
    skipped_count = 0

    for idx, filename in enumerate(image_files, start=1):
        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)
        if image is None:
            print(f"⚠️  Frame {idx:4d}/{len(image_files)} | SKIPPED (unreadable): {filename}")
            skipped_count += 1
            continue

        results = model(image, conf=confidence_threshold, device=device)
        bbox, cls_id, confidence = extract_primary_detection(results[0])

        if bbox is None:
            print(f"⚠️  Frame {idx:4d}/{len(image_files)} | SKIPPED (no detection): {filename}")
            skipped_count += 1
            frames.append({
                "filename": filename,
                "crop": None,
                "class": None,
                "confidence": 0.0,
                "center_distance": float("inf"),
                "frame_index": idx
            })
            continue

        crop = crop_and_normalize(image, bbox)
        center_distance = bbox_center_distance(bbox, image.shape[1], image.shape[0])
        frames.append({
            "filename": filename,
            "bbox": bbox,
            "class": cls_id,
            "confidence": confidence,
            "crop": crop,
            "center_distance": center_distance,
            "frame_index": idx
        })

    groups = group_duplicate_frames(frames, similarity_threshold, temporal_window)

    os.makedirs(output_folder, exist_ok=True)
    unique_folder = os.path.join(output_folder, "representative_frames")
    os.makedirs(unique_folder, exist_ok=True)

    selected_indices = set()
    metadata = []

    for group in groups:
        if not group:
            continue

        best_idx = min(group, key=lambda i: frames[i]["center_distance"])
        selected_indices.add(best_idx)
        member_info = [
            {
                "filename": frames[i]["filename"],
                "frame_index": frames[i]["frame_index"],
                "class": frames[i]["class"],
                "confidence": frames[i]["confidence"],
                "center_distance": frames[i]["center_distance"]
            }
            for i in group
        ]
        metadata.append({
            "group": group,
            "selected_index": best_idx,
            "selected_frame": frames[best_idx]["filename"],
            "selected_center_distance": frames[best_idx]["center_distance"],
            "members": member_info
        })

    saved_count = 0
    duplicate_count = 0

    for idx, frame in enumerate(frames):
        if idx in selected_indices:
            image = cv2.imread(os.path.join(input_folder, frame["filename"]))
            output_path = os.path.join(unique_folder, frame["filename"]) if frame["crop"] is not None else os.path.join(unique_folder, frame["filename"])
            cv2.imwrite(output_path, image)
            saved_count += 1
        elif frame["crop"] is not None:
            duplicate_count += 1

    print("\n" + "=" * 70)
    print("DEDUPLICATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Total frames scanned: {len(frames)}")
    print(f"  Representative frames saved: {saved_count}")
    print(f"  Duplicate frames skipped: {duplicate_count}")
    print(f"  Skipped frames: {skipped_count}")
    print(f"\n📁 Representative frames written to: {unique_folder}")

    metadata_path = os.path.join(output_folder, "bbox_pixel_similarity_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "input_folder": input_folder,
            "output_folder": output_folder,
            "model": model_path,
            "similarity_threshold": similarity_threshold,
            "temporal_window": temporal_window,
            "confidence_threshold": confidence_threshold,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_frames": len(frames),
                "representative_saved": saved_count,
                "duplicates_skipped": duplicate_count,
                "skipped": skipped_count
            },
            "groups": metadata
        }, f, indent=2)

    print(f"📄 Metadata saved to: {metadata_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate saved fault detection frames using bbox pixel similarity and center selection"
    )
    parser.add_argument("input_folder", help="Path to detected frames folder")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO model path")
    parser.add_argument("--output", default=None, help="Output folder path")
    parser.add_argument("--similarity", type=float, default=PIXEL_SIMILARITY_THRESHOLD,
                        help="Pixel similarity threshold for duplicate grouping")
    parser.add_argument("--window", type=int, default=TEMPORAL_WINDOW,
                        help="Temporal window size for duplicate grouping")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help="YOLO confidence threshold")

    args = parser.parse_args()

    input_folder = args.input_folder.rstrip("/")
    parent_folder = os.path.dirname(input_folder)
    base_name = os.path.basename(input_folder)
    output_folder = args.output or os.path.join(parent_folder, f"{base_name}_dedup_bbox_pixels")

    deduplicate_bbox_pixel_similarity(
        input_folder,
        args.model,
        output_folder,
        similarity_threshold=args.similarity,
        temporal_window=args.window,
        confidence_threshold=args.confidence
    )
