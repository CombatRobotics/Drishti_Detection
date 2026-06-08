#!/usr/bin/env python3
"""
Technique: spatio-temporal fault deduplication using YOLO detections.

This script deduplicates overlapping rail-fault detections by comparing
re-detected fault boxes across consecutive frames and maintaining fault
track records. It preserves one representative frame per unique fault event
instead of saving every frame that contains the same physical defect.
"""

import cv2
import os
import json
import torch
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# ==========================================================================
# CONFIGURATION - Edit these as needed
# ==========================================================================
MODEL_PATH = r"/media/viraj/New Volume/Dhrishti/YOLO/v4.1/faultdetection_v4.1.pt"
IOU_THRESHOLD = 0.75            # Minimum IoU to treat a detection as the same fault
CONFIDENCE_THRESHOLD = 0.425    # YOLO confidence threshold for re-detection
TRACK_AGE_THRESHOLD = 25     # Maximum frame gap to keep a track alive


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


def extract_detections(result):
    """Extract bounding boxes, class ids, and confidence scores from a YOLO result."""
    detections = []
    boxes = result.boxes
    if len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()

    for coords, cls_id, conf in zip(xyxy, classes, confidences):
        detections.append({
            "xyxy": coords.tolist(),
            "class": int(cls_id),
            "confidence": float(conf)
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def match_detection_to_track(detection, tracks, iou_threshold):
    """Match a detection to the best existing track based on class and IoU."""
    best_track = None
    best_iou = 0.0

    for track in tracks:
        if track["class"] != detection["class"]:
            continue

        iou = calculate_iou(detection["xyxy"], track["bbox"])
        if iou >= iou_threshold and iou > best_iou:
            best_iou = iou
            best_track = track

    return best_track, best_iou


def prune_old_tracks(tracks, current_frame, max_age):
    """Remove tracks that have not been seen for longer than max_age frames."""
    return [track for track in tracks if current_frame - track["last_seen_frame"] <= max_age]


def deduplicate_spatiotemporal_faults(input_folder, model_path, output_folder,
                                      iou_threshold=IOU_THRESHOLD,
                                      conf_threshold=CONFIDENCE_THRESHOLD,
                                      track_age=TRACK_AGE_THRESHOLD):
    """Deduplicate detected fault frames using spatio-temporal detection association."""
    print("\n" + "=" * 70)
    print("SPATIO-TEMPORAL FAULT DEDUPLICATION")
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
    print(f"⚙️  IoU threshold: {iou_threshold}")
    print(f"⚙️  Confidence threshold: {conf_threshold}")
    print(f"⚙️  Track age threshold: {track_age} frames")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    print(f"\n✓ Model loaded on {device}")

    os.makedirs(output_folder, exist_ok=True)
    unique_folder = os.path.join(output_folder, "unique_fault_frames")
    os.makedirs(unique_folder, exist_ok=True)

    tracks = []
    unique_frame_count = 0
    duplicate_frame_count = 0
    skipped_count = 0
    processed_count = 0
    metadata = []
    track_id_counter = 0

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
        detections = extract_detections(results[0])

        if not detections:
            print(f"⚠️  Frame {idx:4d}/{len(image_files)} | SKIPPED (no detections): {filename}")
            skipped_count += 1
            metadata.append({"filename": filename, "status": "skipped", "reason": "no_detections"})
            tracks = prune_old_tracks(tracks, idx, track_age)
            continue

        tracks = prune_old_tracks(tracks, idx, track_age)
        frame_has_new_fault = False
        matched_infos = []

        for detection in detections:
            track, best_iou = match_detection_to_track(detection, tracks, iou_threshold)
            if track is not None:
                track["last_seen_frame"] = idx
                track["bbox"] = detection["xyxy"]
                track["frames"].append({
                    "frame_index": idx,
                    "filename": filename,
                    "iou": best_iou,
                    "confidence": detection["confidence"]
                })
                matched_infos.append({
                    "status": "duplicate",
                    "track_id": track["id"],
                    "iou": best_iou,
                    "class": detection["class"]
                })
            else:
                frame_has_new_fault = True
                track_id_counter += 1
                new_track = {
                    "id": track_id_counter,
                    "class": detection["class"],
                    "bbox": detection["xyxy"],
                    "last_seen_frame": idx,
                    "rep_filename": filename,
                    "rep_frame_index": idx,
                    "rep_confidence": detection["confidence"],
                    "frames": [{
                        "frame_index": idx,
                        "filename": filename,
                        "iou": 1.0,
                        "confidence": detection["confidence"]
                    }]
                }
                tracks.append(new_track)
                matched_infos.append({
                    "status": "new",
                    "track_id": new_track["id"],
                    "class": detection["class"],
                    "confidence": detection["confidence"]
                })

        if frame_has_new_fault:
            unique_frame_count += 1
            output_path = os.path.join(unique_folder, filename)
            cv2.imwrite(output_path, image)
            print(f"✓ Frame {idx:4d}/{len(image_files)} | NEW FAULT FOUND | saved as representative frame: {filename}")
            metadata.append({
                "filename": filename,
                "status": "saved",
                "new_tracks": [info for info in matched_infos if info["status"] == "new"],
                "duplicates": [info for info in matched_infos if info["status"] == "duplicate"]
            })
        else:
            duplicate_frame_count += 1
            print(f"✗ Frame {idx:4d}/{len(image_files)} | DUPLICATE FRAME - no new fault track")
            metadata.append({
                "filename": filename,
                "status": "duplicate",
                "matches": matched_infos
            })

    print("\n" + "=" * 70)
    print("DEDUPLICATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Total images processed: {processed_count}")
    print(f"  Unique representative frames saved: {unique_frame_count}")
    print(f"  Duplicate frames skipped: {duplicate_frame_count}")
    print(f"  Skipped frames: {skipped_count}")
    print(f"  Unique fault tracks: {len(tracks)}")

    metadata_path = os.path.join(output_folder, "spatiotemporal_dedup_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "input_folder": input_folder,
            "output_folder": output_folder,
            "model": model_path,
            "iou_threshold": iou_threshold,
            "confidence_threshold": conf_threshold,
            "track_age_threshold": track_age,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_processed": processed_count,
                "unique_saved": unique_frame_count,
                "duplicates_skipped": duplicate_frame_count,
                "skipped": skipped_count,
                "unique_tracks": len(tracks)
            },
            "tracks": [
                {
                    "track_id": track["id"],
                    "class": track["class"],
                    "rep_filename": track["rep_filename"],
                    "rep_frame_index": track["rep_frame_index"],
                    "rep_confidence": track["rep_confidence"],
                    "frames_seen": len(track["frames"]),
                    "last_seen_frame": track["last_seen_frame"]
                }
                for track in tracks
            ],
            "frames": metadata
        }, f, indent=2)

    print(f"\n📁 Unique frames written to: {unique_folder}")
    print(f"📄 Metadata saved to: {metadata_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate saved fault detection frames using spatio-temporal detection tracking"
    )
    parser.add_argument("input_folder", help="Path to detected frames folder")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO model path")
    parser.add_argument("--output", default=None, help="Output folder path")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD,
                        help="IoU threshold for matching same fault")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help="YOLO confidence threshold")
    parser.add_argument("--age", type=int, default=TRACK_AGE_THRESHOLD,
                        help="Maximum frame gap for an active fault track")

    args = parser.parse_args()

    input_folder = args.input_folder.rstrip("/")
    parent_folder = os.path.dirname(input_folder)
    base_name = os.path.basename(input_folder)
    output_folder = args.output or os.path.join(parent_folder, f"{base_name}_dedup_spatiotemporal")

    deduplicate_spatiotemporal_faults(
        input_folder,
        args.model,
        output_folder,
        iou_threshold=args.iou,
        conf_threshold=args.confidence,
        track_age=args.age
    )
