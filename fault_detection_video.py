import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
import os
from datetime import datetime
# import argparse  # Commented out since we're using hardcoded paths

# Configuration variables - Edit these as needed
VIDEO_PATH = r"D:\rosbags_17thJan\ace_delhi2_20260116_033518\ace_delhi2_20260116_03351820260117_172917.avi"  # Path to your input video file
MODEL_PATH = r"D:\Dhrishti\YOLO\v4.1\faultdetection_v4.1.pt"                          # Path to your PyTorch model (.pt file)
PLAYBACK_SPEED = 1.5                                              # Playback speed multiplier (0.25x = slower, 1.0x = normal, 2.0x = faster)
OUTPUT_DIR = r"D:\Dhrishti\Drishti_DMRC_data\Output_jan_2026\ace_delhi2_20260116_03351820260117_172917"              # Base directory for saving detected frames
CONFIDENCE_THRESHOLD = 0.5                                    # Detection confidence threshold

def load_model(model_path):
    """
    Load the YOLO model from the .pt file using ultralytics.
    """
    model = YOLO(model_path)
    return model

def preprocess_frame(frame, input_size=(1024, 626)):
    """
    Preprocess the frame for model input.
    Handles both RGB and greyscale video input.
    Adjust based on your model's expected input format.
    """
    # Check if frame is greyscale (single channel) or RGB (3 channels)
    if len(frame.shape) == 2 or frame.shape[2] == 1:
        # Greyscale frame - convert to 3-channel by duplicating
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    else:
        # RGB frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Resize to model's input size
    frame_resized = cv2.resize(frame_rgb, input_size)

    # Convert to tensor and normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    frame_tensor = transform(frame_resized).unsqueeze(0)
    return frame_tensor

def postprocess_output(results, frame):
    """
    Post-process the YOLO results and overlay detections on the frame.
    """
    # YOLO results can plot directly on the frame
    annotated_frame = results[0].plot()  # Get the first result and plot detections
    return annotated_frame

def main(video_path, model_path, playback_speed=1.0, output_dir=None):
    # Load the model
    model = load_model(model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.to(device)
    if device.startswith("cuda"):
        print(f"Running inference on GPU ({device})")
    else:
        print("Running inference on CPU (CUDA not available)")

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Auto-detect FPS from input video
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("Warning: Could not detect video FPS, defaulting to 30")
        video_fps = 30
    
    # Get video info for display
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nVideo Info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps}")
    print(f"  Total frames: {total_frames}")

    # Create output directory for this run
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

    # Calculate delay between frames based on video's native FPS and playback speed
    delay = int((1000 / video_fps) / playback_speed)  # milliseconds

    print(f"\nProcessing video at {video_fps} FPS ({playback_speed}x speed)")
    print("Press 'q' to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame")
            break

        # Run inference using YOLO (no preprocessing needed for YOLO)
        results = model(frame, conf=CONFIDENCE_THRESHOLD, device=device)  # Adjust confidence threshold as needed

        # Post-process and overlay results on frame
        processed_frame = postprocess_output(results, frame)

        # Save frame if detections are found
        if run_dir and len(results[0].boxes) > 0:
            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            detections = len(results[0].boxes)
            confidence = results[0].boxes.conf.max().item() if detections > 0 else 0

            class_names = results[0].names
            saved_labels = set()
            for cls_id in results[0].boxes.cls.cpu().numpy().astype(int):
                label = class_names.get(int(cls_id), f"class_{cls_id}")
                if label in saved_labels:
                    continue
                saved_labels.add(label)
                class_dir = os.path.join(run_dir, label)
                os.makedirs(class_dir, exist_ok=True)
                filename = f"frame_{frame_count:04d}_{detections}_{confidence:.2f}.jpg"
                filepath = os.path.join(class_dir, filename)
                cv2.imwrite(filepath, processed_frame)
                if faults_dir:
                    faults_path = os.path.join(faults_dir, filename)
                    cv2.imwrite(faults_path, frame)
            print(f"Saved frame {frame_count} with {detections} detections (conf: {confidence:.2f})")

        # Display the frame
        cv2.imshow('Fault Detection', processed_frame)

        # Wait for key press and control frame rate
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Configuration variables are defined at the top of the file
    # You can either use the hardcoded paths above, or use command line arguments
    # Uncomment the lines below to use command line arguments instead

    # parser = argparse.ArgumentParser(description='Run fault detection on video frames')
    # parser.add_argument('video_path', help='Path to the input video file')
    # parser.add_argument('model_path', help='Path to the PyTorch model (.pt file)')
    # parser.add_argument('--playback_speed', type=float, default=1.0,
    #                     help='Playback speed multiplier (default: 1.0)')
    # args = parser.parse_args()
    # main(args.video_path, args.model_path, args.playback_speed)

    # Using hardcoded paths
    main(VIDEO_PATH, MODEL_PATH, playback_speed=PLAYBACK_SPEED, output_dir=OUTPUT_DIR)
