#!/usr/bin/env python3
"""
Interactive GUI tuner for spatial similarity deduplication.

This script uses the same detection and frame similarity logic as
`deduplicate_spatial_similarity.py`, but runs in a GUI so you can tune
thresholds live and see how many frames become duplicates.

Usage:
  python3 dedup_spatial_similarity_tuner.py /path/to/input_folder

The input folder is the directory containing the detected fault frame images.
"""

import argparse
import os
import cv2
import json
import numpy as np
import torch
from ultralytics import YOLO
from datetime import datetime
from deduplicate_spatial_similarity import (
    MODEL_PATH,
    BBOX_OVERLAP_THRESHOLD,
    SIMILARITY_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    MAX_HISTORY,
    extract_detections,
    calculate_spatial_similarity,
)

DEFAULT_MODEL_PATH = MODEL_PATH
DEFAULT_CONFIDENCE = CONFIDENCE_THRESHOLD
DEFAULT_IOU = BBOX_OVERLAP_THRESHOLD
DEFAULT_SIMILARITY = SIMILARITY_THRESHOLD
DEFAULT_HISTORY = MAX_HISTORY
MAX_GROUPS = 200
MAX_HISTORY_SLIDER = 20


def bbox_center_distance(bbox, width, height):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    fx = width / 2.0
    fy = height / 2.0
    return float(np.hypot(cx - fx, cy - fy))


def load_frames(input_folder, model_path, confidence_threshold):
    image_files = sorted([
        f for f in os.listdir(input_folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not image_files:
        raise ValueError(f"No image files found in {input_folder}")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    frames = []

    print(f"Running YOLO detection on {len(image_files)} frames...")
    for idx, filename in enumerate(image_files, start=1):
        filepath = os.path.join(input_folder, filename)
        image = cv2.imread(filepath)
        if image is None:
            frames.append({
                "filename": filename,
                "image": None,
                "detections": None,
                "center_distance": float("inf"),
                "frame_index": idx,
            })
            continue

        results = model(image, conf=confidence_threshold, device=device)
        detections = extract_detections(results[0])
        center_distance = float("inf")
        if detections:
            bbox = detections[0]["xyxy"]
            center_distance = bbox_center_distance(bbox, image.shape[1], image.shape[0])

        frames.append({
            "filename": filename,
            "image": image,
            "detections": detections,
            "center_distance": center_distance,
            "frame_index": idx,
        })

        if idx % 50 == 0 or idx == len(image_files):
            print(f"  {idx}/{len(image_files)} frames processed")
    return frames


def build_summary_image(total, saved, duplicate, skipped, iou, similarity, history_len, confidence, group_count):
    width, height = 620, 280
    img = np.zeros((height, width, 3), dtype=np.uint8)
    lines = [
        f"IoU overlap threshold: {iou:.2f}",
        f"Frame similarity threshold: {similarity:.2f}",
        f"History compare depth: {history_len}",
        f"YOLO confidence: {confidence:.2f}",
        f"Total frames: {total}",
        f"Unique saved: {saved}",
        f"Duplicate skipped: {duplicate}",
        f"Skipped/missing: {skipped}",
        f"Duplicate groups: {group_count}",
    ]
    y = 30
    for line in lines:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (230, 230, 230), 2, cv2.LINE_AA)
        y += 30

    bar_x = 20
    bar_y = y + 10
    bar_h = 24
    maxv = max(total, 1)
    for label, value, color in [
        ("unique", saved, (84, 255, 159)),
        ("duplicate", duplicate, (60, 140, 255)),
        ("skipped", skipped, (203, 203, 203)),
    ]:
        length = int((value / maxv) * (width - 120))
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + length, bar_y + bar_h), color, -1)
        cv2.putText(img, f"{label}: {value}", (bar_x + 8, bar_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        bar_y += bar_h + 12
    return img


def build_group_preview(frames, groups, selected_group, folder):
    if not groups or selected_group < 0 or selected_group >= len(groups):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "No group selected", (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (240, 240, 240), 2)
        return img
    indices = groups[selected_group]
    rep_idx = indices[0]
    rep_frame = frames[rep_idx]
    rep_img = rep_frame["image"]
    if rep_img is None:
        rep_img = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        h, w = rep_img.shape[:2]
        scale = min(640 / w, 480 / h, 1.0)
        rep_img = cv2.resize(rep_img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        y_off = (480 - rep_img.shape[0]) // 2
        x_off = (640 - rep_img.shape[1]) // 2
        canvas[y_off : y_off + rep_img.shape[0], x_off : x_off + rep_img.shape[1]] = rep_img
        rep_img = canvas
    cv2.putText(rep_img, f"Group {selected_group} / {len(groups)-1}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2)
    cv2.putText(rep_img, f"Rep: {rep_frame['filename']}", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(rep_img, f"Members: {len(indices)}", (12, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return rep_img


def build_spatial_groups(frames, iou_threshold, similarity_threshold, history_len):
    groups = []
    unique_refs = []
    group_by_unique = {}
    duplicate_count = 0

    for i, frame in enumerate(frames):
        if frame["detections"] is None or len(frame["detections"]) == 0:
            continue

        duplicate = False
        for unique_idx in unique_refs[-history_len:]:
            sim, _ = calculate_spatial_similarity(
                frame["detections"],
                frames[unique_idx]["detections"],
                iou_threshold,
            )
            if sim >= similarity_threshold:
                duplicate = True
                group_id = group_by_unique[unique_idx]
                groups[group_id].append(i)
                duplicate_count += 1
                break

        if not duplicate:
            group_id = len(groups)
            groups.append([i])
            unique_refs.append(i)
            group_by_unique[i] = group_id

    return groups, len(unique_refs), duplicate_count


def main():
    parser = argparse.ArgumentParser(description="Interactive spatial similarity dedup GUI tuner")
    parser.add_argument("input_folder", help="Path to folder containing detected fault images")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLO model path")
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE, help="YOLO confidence threshold")
    args = parser.parse_args()

    input_folder = args.input_folder.rstrip("/")
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(input_folder)

    frames = load_frames(input_folder, args.model, args.confidence)
    total_frames = len(frames)
    skipped_frames = sum(1 for f in frames if f["detections"] is None or len(f["detections"]) == 0)

    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Summary", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 500, 240)
    cv2.resizeWindow("Summary", 640, 280)
    cv2.resizeWindow("Preview", 640, 520)

    cv2.createTrackbar("IoU x100", "Controls", int(DEFAULT_IOU * 100), 100, lambda x: None)
    cv2.createTrackbar("Similarity x100", "Controls", int(DEFAULT_SIMILARITY * 100), 100, lambda x: None)
    cv2.createTrackbar("History", "Controls", DEFAULT_HISTORY, MAX_HISTORY_SLIDER, lambda x: None)
    cv2.createTrackbar("Group idx", "Controls", 0, MAX_GROUPS, lambda x: None)

    last_params = (-1, -1, -1)
    groups = []
    selected_group = 0
    unique_saved = 0
    duplicate_count = 0

    while True:
        iou = cv2.getTrackbarPos("IoU x100", "Controls") / 100.0
        similarity = cv2.getTrackbarPos("Similarity x100", "Controls") / 100.0
        history_len = max(1, cv2.getTrackbarPos("History", "Controls"))
        selected_group = cv2.getTrackbarPos("Group idx", "Controls")

        if (iou, similarity, history_len) != last_params:
            groups, unique_saved, duplicate_count = build_spatial_groups(frames, iou, similarity, history_len)
            last_params = (iou, similarity, history_len)
            print(f"Updated: IoU={iou:.2f}, sim={similarity:.2f}, history={history_len}, unique={unique_saved}, groups={len(groups)}")

        if selected_group >= len(groups):
            selected_group = max(0, len(groups) - 1)
            cv2.setTrackbarPos("Group idx", "Controls", selected_group)

        summary_img = build_summary_image(
            total=total_frames,
            saved=unique_saved,
            duplicate=duplicate_count,
            skipped=skipped_frames,
            iou=iou,
            similarity=similarity,
            history_len=history_len,
            confidence=args.confidence,
            group_count=len(groups),
        )
        preview_img = build_group_preview(frames, groups, selected_group, input_folder)

        cv2.imshow("Summary", summary_img)
        cv2.imshow("Preview", preview_img)

        key = cv2.waitKey(100)
        if key == 27 or key == ord("q"):
            break
        if key == ord("s"):
            out_dir = os.path.join(input_folder, f"spatial_tuner_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "tuner_params.json"), "w") as f:
                json.dump({
                    "iou": iou,
                    "similarity": similarity,
                    "history": history_len,
                    "confidence": args.confidence,
                    "group_index": selected_group,
                    "timestamp": datetime.now().isoformat(),
                }, f, indent=2)
            print(f"Saved snapshot metadata to {out_dir}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
