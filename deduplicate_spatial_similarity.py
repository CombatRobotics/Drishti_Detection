import cv2
import os
import json
import torch
from ultralytics import YOLO
from datetime import datetime

# Technique: spatial similarity deduplication using YOLO re-detection
# This script compares all detected boxes between frames, computing box IoU and a frame-level similarity score.
# Frames are marked duplicate when spatial matching to recent saved frames exceeds the similarity threshold.

# ==========================================================================
# CONFIGURATION - Edit these as needed
# ==========================================================================
MODEL_PATH = r"/media/viraj/New Volume/Dhrishti/YOLO/v4.1/faultdetection_v4.1.pt"
BBOX_OVERLAP_THRESHOLD = 0.05  # Minimum IoU to consider two boxes overlapping
SIMILARITY_THRESHOLD = 0.1    # Minimum spatial similarity score to treat frames as duplicates
CONFIDENCE_THRESHOLD = 0.2   # YOLO confidence threshold for re-detection
MAX_HISTORY = 3                # Number of previous saved frames to compare against


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) for two bounding boxes.

    Args:
        bbox1 (list): [x1, y1, x2, y2]
        bbox2 (list): [x1, y1, x2, y2]

    Returns:
        float: IoU score between 0.0 and 1.0.
    """
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
    """
    Extract bounding boxes, classes, and confidence scores from a YOLO result.

    Args:
        result: A single YOLO result object.

    Returns:
        list[dict]: List of detections with fields 'xyxy', 'class', and 'confidence'.
    """
    detections = []
    boxes = result.boxes
    if len(boxes) == 0:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    for coords, cls_id, conf in zip(xyxy, classes, confs):
        detections.append({
            "xyxy": coords.tolist(),
            "class": int(cls_id),
            "confidence": float(conf)
        })
    return detections


def calculate_spatial_similarity(current_detections, previous_detections, overlap_threshold):
    """
    Compute a spatial similarity score between two sets of detections.

    This measures how many current detections overlap with previous detections
    of the same class using IoU.

    Args:
        current_detections (list[dict]): Detections for the current frame.
        previous_detections (list[dict]): Detections for the comparison frame.
        overlap_threshold (float): IoU threshold to count an overlap as a match.

    Returns:
        tuple[float, float]: (similarity_score, average_iou)
            similarity_score: matched boxes / max(len(current), len(previous))
            average_iou: average best IoU for current detections
    """
    if not current_detections or not previous_detections:
        return 0.0, 0.0

    matched = 0
    total_iou = 0.0

    for current in current_detections:
        best_iou = 0.0
        for previous in previous_detections:
            if current["class"] != previous["class"]:
                continue
            iou = calculate_iou(current["xyxy"], previous["xyxy"])
            best_iou = max(best_iou, iou)

        total_iou += best_iou
        if best_iou >= overlap_threshold:
            matched += 1

    similarity_score = matched / max(len(current_detections), len(previous_detections))
    average_iou = total_iou / len(current_detections)
    return similarity_score, average_iou


def is_duplicate_frame(current_detections, history, overlap_threshold, similarity_threshold):
    """
    Compare the current frame against saved history to decide if it is a duplicate.

    Args:
        current_detections (list[dict]): Detections for the current frame.
        history (list[dict]): List of previous saved frames with detections.
        overlap_threshold (float): IoU threshold for box overlap.
        similarity_threshold (float): Frame similarity threshold for duplicate marking.

    Returns:
        tuple[bool, dict]: (is_duplicate, details)
    """
    for record in history:
        similarity, avg_iou = calculate_spatial_similarity(
            current_detections,
            record["detections"],
            overlap_threshold
        )
        if similarity >= similarity_threshold:
            return True, {
                "matched_frame": record["filename"],
                "similarity": similarity,
                "average_iou": avg_iou
            }
    return False, {"similarity": 0.0, "average_iou": 0.0}


def deduplicate_spatial_overlap(input_folder, model_path, output_folder,
                                 bbox_overlap_threshold=BBOX_OVERLAP_THRESHOLD,
                                 similarity_threshold=SIMILARITY_THRESHOLD,
                                 conf_threshold=CONFIDENCE_THRESHOLD,
                                 max_history=MAX_HISTORY):
    """
    Deduplicate detected frames using spatial similarity and bounding-box overlap.

    Args:
        input_folder (str): Folder containing detected frames.
        model_path (str): Path to YOLO model.
        output_folder (str): Folder to write unique frames.
        bbox_overlap_threshold (float): IoU threshold per-box overlap.
        similarity_threshold (float): Similarity threshold for duplicate frames.
        conf_threshold (float): YOLO confidence threshold.
        max_history (int): Number of previous unique frames to compare.
    """
    print("\n" + "=" * 70)
    print("SPATIAL OVERLAP DEDUPLICATION PIPELINE")
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
    print(f"⚙️  IoU overlap threshold: {bbox_overlap_threshold}")
    print(f"⚙️  Similarity threshold: {similarity_threshold}")
    print(f"⚙️  Confidence threshold: {conf_threshold}")
    print(f"⚙️  Lookback history: {max_history} frames")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    print(f"\n✓ Model loaded on {device}")

    os.makedirs(output_folder, exist_ok=True)

    saved_history = []
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
        detections = extract_detections(results[0])

        if not detections:
            print(f"⚠️  Frame {idx:4d}/{len(image_files)} | SKIPPED (no detections): {filename}")
            skipped_count += 1
            metadata.append({"filename": filename, "status": "skipped", "reason": "no_detections"})
            continue

        duplicate, details = is_duplicate_frame(
            detections,
            saved_history[-max_history:],
            bbox_overlap_threshold,
            similarity_threshold
        )

        if duplicate:
            duplicate_count += 1
            print(
                f"✗ Frame {idx:4d}/{len(image_files)} | DUPLICATE: {filename} "
                f"(similarity={details['similarity']:.3f}, avg_iou={details['average_iou']:.3f}, "
                f"matched_to={details['matched_frame']})"
            )
            metadata.append({
                "filename": filename,
                "status": "duplicate",
                "matched_to": details["matched_frame"],
                "similarity": details["similarity"],
                "average_iou": details["average_iou"]
            })
            continue

        unique_count += 1
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)
        saved_history.append({
            "filename": filename,
            "detections": detections
        })
        print(f"✓ Frame {idx:4d}/{len(image_files)} | SAVED unique frame: {filename}")
        metadata.append({
            "filename": filename,
            "status": "saved",
            "detections": len(detections)
        })

    dedup_percentage = (duplicate_count / processed_count * 100) if processed_count > 0 else 0.0

    print("\n" + "=" * 70)
    print("DEDUPLICATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Summary:")
    print(f"  Total images processed: {processed_count}")
    print(f"  Unique images saved: {unique_count}")
    print(f"  Duplicate images skipped: {duplicate_count}")
    print(f"  Skipped images: {skipped_count}")
    print(f"  Reduction rate: {dedup_percentage:.1f}%")
    print(f"\n📁 Unique frames saved to: {output_folder}")

    metadata_path = os.path.join(output_folder, "deduplication_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({
            "input_folder": input_folder,
            "output_folder": output_folder,
            "model": model_path,
            "bbox_overlap_threshold": bbox_overlap_threshold,
            "similarity_threshold": similarity_threshold,
            "confidence_threshold": conf_threshold,
            "max_history": max_history,
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_processed": processed_count,
                "unique_saved": unique_count,
                "duplicates_skipped": duplicate_count,
                "skipped": skipped_count,
                "reduction_percentage": dedup_percentage
            },
            "frames": metadata
        }, f, indent=2)

    print(f"📄 Metadata saved to: {metadata_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Deduplicate detected frames using spatial similarity and box overlap"
    )
    parser.add_argument("input_folder", help="Path to detected frames folder")
    parser.add_argument("--model", default=MODEL_PATH, help="YOLO model path")
    parser.add_argument("--output", default=None, help="Output folder path")
    parser.add_argument("--iou", type=float, default=BBOX_OVERLAP_THRESHOLD,
                        help="Box IoU overlap threshold")
    parser.add_argument("--similarity", type=float, default=SIMILARITY_THRESHOLD,
                        help="Frame similarity threshold")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help="YOLO confidence threshold")
    parser.add_argument("--history", type=int, default=MAX_HISTORY,
                        help="Number of previous unique frames to compare")

    args = parser.parse_args()

    input_folder = args.input_folder.rstrip("/")
    parent_folder = os.path.dirname(input_folder)
    base_name = os.path.basename(input_folder)
    output_folder = args.output or os.path.join(parent_folder, f"{base_name}_dedup_spatial")

    deduplicate_spatial_overlap(
        input_folder,
        args.model,
        output_folder,
        bbox_overlap_threshold=args.iou,
        similarity_threshold=args.similarity,
        conf_threshold=args.confidence,
        max_history=args.history
    )
