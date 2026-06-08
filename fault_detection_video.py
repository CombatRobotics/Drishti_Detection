import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import os
from datetime import datetime
import time  # For measuring inference time
# import argparse  # Commented out since we're using hardcoded paths

# ============================================================================
# CONFIGURATION VARIABLES - Edit these as needed
# ============================================================================
VIDEO_PATH = r"/media/viraj/New Volume/Dhrishti/Day 6/ace_delhi_6_10m20260124_050136/ace_delhi_6_10m20260124_050136.avi"  # Path to your input video file
MODEL_PATH = r"/media/viraj/New Volume/Dhrishti/YOLO/v4.1/faultdetection_v4.1.pt"                          # Path to your PyTorch model (.pt file)
PLAYBACK_SPEED = 1.0                                              # Playback speed multiplier (0.25x = slower, 1.0x = normal, 2.0x = faster)
OUTPUT_DIR = r"/home/viraj/inference_trial"              # Base directory for saving detected frames
CONFIDENCE_THRESHOLD = 0.425                                   # Detection confidence threshold (0.0-1.0)


def load_model(model_path):
    """
    Load the YOLO model from the .pt file using ultralytics library.

    Args:
        model_path (str): Path to the YOLO model file (.pt format)

    Returns:
        YOLO: Loaded YOLO model object ready for inference
    """
    model = YOLO(model_path)
    return model


def preprocess_frame(frame, input_size=(1024, 626)):
    """
    Preprocess the frame for model input.
    Converts greyscale to RGB and applies necessary transformations.

    Args:
        frame (np.ndarray): Input frame from video
        input_size (tuple): Target size for resizing (width, height)

    Returns:
        torch.Tensor: Preprocessed frame tensor ready for model inference

    Note:
        YOLO models typically handle preprocessing internally, so this function
        is kept for compatibility but may not be used if YOLO does its own preprocessing.
    """
    # Check if frame is greyscale (single channel) or RGB (3 channels)
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        # Greyscale frame - convert to 3-channel by duplicating channels
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        # RGB frame - convert from BGR (OpenCV default) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model's expected input size
    frame_resized = cv2.resize(frame_rgb, input_size)

    # Apply normalization transforms (ImageNet standard normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert to tensor and normalize to [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    # Add batch dimension (model expects batch of images)
    frame_tensor = transform(frame_resized).unsqueeze(0)
    return frame_tensor


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Used to detect duplicate detections across consecutive frames.

    Args:
        bbox1 (list): First bounding box [x1, y1, x2, y2]
        bbox2 (list): Second bounding box [x1, y1, x2, y2]

    Returns:
        float: IoU score between 0 and 1 (1.0 = identical boxes, 0.0 = no overlap)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Calculate intersection area
    intersection_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    intersection_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    intersection_area = intersection_x * intersection_y

    # Calculate union area
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = area1 + area2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    return iou


def get_bbox_center_distance(bbox, frame_width, frame_height):
    """
    Calculate distance from bounding box center to frame center.
    Lower distance = defect more centered in frame.

    Args:
        bbox (list): Bounding box [x1, y1, x2, y2]
        frame_width (int): Width of frame
        frame_height (int): Height of frame

    Returns:
        float: Euclidean distance from bbox center to frame center
    """
    x1, y1, x2, y2 = bbox
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    # Calculate Euclidean distance
    distance = np.sqrt((bbox_center_x - frame_center_x) ** 2 +
                       (bbox_center_y - frame_center_y) ** 2)
    return distance


def postprocess_output(results, frame):
    """
    Post-process the YOLO detection results and draw bounding boxes on the frame.

    Args:
        results (list): YOLO inference results containing detected boxes
        frame (np.ndarray): Original frame to annotate

    Returns:
        np.ndarray: Frame with detection bounding boxes and labels drawn
    """
    # Use YOLO's built-in plot function to draw all detections on the frame
    annotated_frame = results[0].plot()  # Get the first result and plot detections
    return annotated_frame


def main(video_path, model_path, playback_speed=1.0, output_dir=None):
    """
    Main function to run fault detection on video frames.

    Args:
        video_path (str): Path to input video file
        model_path (str): Path to YOLO model (.pt file)
        playback_speed (float): Playback speed multiplier (default: 1.0)
        output_dir (str): Directory to save detected frames (optional)
    """

    # ========================================================================
    # START TIMING: Measure total execution time
    # ========================================================================
    total_start_time = time.time()

    # ========================================================================
    # STEP 1: Load Model and Check Device
    # ========================================================================
    print("\n" + "="*70)
    print("FAULT DETECTION VIDEO PROCESSOR")
    print("="*70)

    model = load_model(model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device.startswith("cuda"):
        print(f"\n✓ Running inference on GPU ({device})")
    else:
        print("\n⚠ Running inference on CPU (CUDA not available)")

    # ========================================================================
    # STEP 2: Open Video File and Read Metadata
    # ========================================================================
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_path}")
        return

    # Auto-detect FPS from input video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("⚠ Warning: Could not detect video FPS, defaulting to 30")
        video_fps = 30

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps}")
    print(f"  Total frames: {total_frames}")

    # ========================================================================
    # STEP 3: Create Output Directory Structure
    # ========================================================================
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        faults_dir = os.path.join(run_dir, "faults")
        os.makedirs(faults_dir, exist_ok=True)
        print(f"  Saving detected frames to: {run_dir}")
    else:
        run_dir = None
        faults_dir = None

    # ========================================================================
    # STEP 4: Calculate Frame Display Timing
    # ========================================================================
    # Frame delay = (1000ms per frame / FPS) / playback_speed
    delay = int((1000 / video_fps) / playback_speed)  # milliseconds

    print(f"\nProcessing video at {video_fps} FPS with {playback_speed}x playback speed")
    print("Press 'q' to quit\n")
    print("-"*70)

    # ========================================================================
    # STEP 5: Main Processing Loop - Frame-by-Frame Inference
    # ========================================================================

    # Tracking variables for performance metrics
    frame_count = 0
    inference_times = []  # List to store inference time for each frame
    detection_count = 0   # Total detections across all frames

    while True:
        # Read next frame from video
        ret, frame = cap.read()
        if not ret:
            print("\n✓ End of video or error reading frame")
            break

        frame_count += 1

        # ====================================================================
        # RUN INFERENCE with timing
        # ====================================================================
        inference_start = time.time()  # Start timing inference

        # Run YOLO inference on frame (automatic preprocessing by YOLO)
        results = model(frame, conf=CONFIDENCE_THRESHOLD, device=device)

        inference_end = time.time()  # End timing inference
        inference_time_ms = (inference_end - inference_start) * 1000  # Convert to milliseconds
        inference_times.append(inference_time_ms)

        # ====================================================================
        # POST-PROCESS: Draw Bounding Boxes
        # ====================================================================
        processed_frame = postprocess_output(results, frame)

        # ====================================================================
        # SAVE DETECTIONS
        # ====================================================================
        if run_dir and len(results[0].boxes) > 0:
            current_frame_index = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            detections = len(results[0].boxes)
            detection_count += detections
            confidence = results[0].boxes.conf.max().item() if detections > 0 else 0

            filename = f"frame_{current_frame_index:04d}_{detections}_{confidence:.2f}.jpg"
            faults_path = os.path.join(faults_dir, filename)
            cv2.imwrite(faults_path, frame)

            print(f"Frame {current_frame_index:4d} | Detections: {detections} | Confidence: {confidence:.2f} | SAVED | Inference: {inference_time_ms:.2f}ms")

        # ====================================================================
        # DISPLAY FRAME
        # ====================================================================
        cv2.imshow('Fault Detection', processed_frame)

        # Control playback speed and handle user input
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("\n⚠ Processing stopped by user (pressed 'q')")
            break

    # ========================================================================
    # STEP 6: Cleanup and Print Performance Metrics
    # ========================================================================
    cap.release()
    cv2.destroyAllWindows()

    # Stop timing total execution
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    # Calculate and display inference time statistics
    print("\n" + "="*70)
    print("PROCESSING COMPLETE - PERFORMANCE METRICS")
    print("="*70)

    if inference_times:
        avg_inference_time = sum(inference_times) / len(inference_times)
        min_inference_time = min(inference_times)
        max_inference_time = max(inference_times)
        total_inference_time = sum(inference_times)

        # Calculate video duration
        video_duration_seconds = frame_count / video_fps if video_fps > 0 else 0
        video_duration_minutes = video_duration_seconds / 60
        video_duration_hours = video_duration_minutes / 60

        # Calculate processing overhead
        overhead_time = total_elapsed_time - (total_inference_time / 1000)
        overhead_percentage = (overhead_time / total_elapsed_time * 100) if total_elapsed_time > 0 else 0

        print(f"\n📹 Video Information:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Video FPS: {video_fps}")
        print(f"  Original video duration: {video_duration_seconds:.2f} seconds ({video_duration_minutes:.2f} min)")
        if video_duration_hours > 0:
            print(f"                            {video_duration_hours:.2f} hours")

        print(f"\n🔍 Detection Summary:")
        print(f"  Total frames processed: {frame_count}")
        print(f"  Total detections found: {detection_count}")

        print(f"\n⏱️  Inference Time Statistics:")
        print(f"  Average per frame: {avg_inference_time:.2f} ms")
        print(f"  Minimum per frame: {min_inference_time:.2f} ms")
        print(f"  Maximum per frame: {max_inference_time:.2f} ms")
        print(f"  Total inference time: {total_inference_time/1000:.2f} seconds ({total_inference_time/1000/60:.2f} min)")

        print(f"\n⏳ Total Execution Time:")
        print(f"  Total elapsed time (wall clock): {total_elapsed_time:.2f} seconds ({total_elapsed_time/60:.2f} min)")
        print(f"  Processing overhead (I/O, display, etc): {overhead_time:.2f} seconds ({overhead_percentage:.1f}%)")

        # Calculate speedup
        speedup_factor = video_duration_seconds / total_elapsed_time if total_elapsed_time > 0 else 0
        print(f"\n📊 Processing Efficiency:")
        print(f"  Processing speed: {speedup_factor:.2f}x (processed {speedup_factor:.2f} seconds of video per second)")

        # Calculate FPS achieved
        achieved_fps = 1000 / avg_inference_time if avg_inference_time > 0 else 0
        print(f"  Achieved inference FPS: {achieved_fps:.2f} fps")
        print(f"  Expected video playback FPS: {video_fps}")

        if achieved_fps >= video_fps:
            print(f"  ✓ Real-time capable: YES (inference fast enough for {video_fps} fps)")
        else:
            required_speedup = video_fps / achieved_fps
            print(f"  ✓ Real-time capable: NO (would need {required_speedup:.2f}x faster inference)")

    print("="*70 + "\n")


if __name__ == "__main__":
    """
    Script entry point.

    Configuration variables are defined at the top of the file.
    You can either use the hardcoded paths above, or uncomment the argparse
    section below to use command line arguments instead.
    """

    # Option 1: Command line arguments (uncomment to enable)
    # parser = argparse.ArgumentParser(description='Run fault detection on video frames')
    # parser.add_argument('video_path', help='Path to the input video file')
    # parser.add_argument('model_path', help='Path to the PyTorch model (.pt file)')
    # parser.add_argument('--playback_speed', type=float, default=1.0,
    #                     help='Playback speed multiplier (default: 1.0)')
    # args = parser.parse_args()
    # main(args.video_path, args.model_path, args.playback_speed)

    # Option 2: Using hardcoded configuration variables (currently active)
    main(VIDEO_PATH, MODEL_PATH, playback_speed=PLAYBACK_SPEED, output_dir=OUTPUT_DIR)
