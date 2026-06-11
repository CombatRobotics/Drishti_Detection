#!/usr/bin/env python3
"""
Production-Grade Fault Detection Pipeline
==========================================

Reliable video frame processing using OpenCV frame-by-frame decoding.
Counts ACTUAL decoded frames (not metadata) for reproducibility across systems.

Key Features:
- Reliable frame counting (works with any codec)
- Production-grade error handling
- Comprehensive performance metrics
- Reproducible across systems (local, DGX, cloud)
- Real-time display with adjustable speed
"""

import os
import argparse
import cv2
import torch
import numpy as np
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
import time
from typing import Tuple, Optional
from ultralytics import YOLO

# ============================================================================
# CONFIGURATION
# ============================================================================
VIDEO_PATH = r"/media/viraj/New Volume/Dhrishti/Day 6/ace_delhi_6_10m20260124_050136/ace_delhi_6_10m20260124_050136.avi"
MODEL_PATH = r"/media/viraj/New Volume/Dhrishti/YOLO/v4.1/faultdetection_v4.1.engine"
PLAYBACK_SPEED = 1.0
OUTPUT_DIR = r"/home/viraj/inference_trial2"
CONFIDENCE_THRESHOLD = 0.425
DETERMINISM_LEVEL = 2  # 0 = none, 1 = basic, 2 = GPU-level, 3 = maximum


def setup_determinism(level: int):
    """
    Apply determinism settings based on selected level.

    Levels:
      0 - No determinism (baseline runtime behavior)
      1 - Basic determinism (fixed seeds + cudnn deterministic)
      2 - GPU-level determinism (CUDA workspace sync/config)
      3 - Maximum determinism (all additional flags)
    """
    print(f"\n⚙️  Determinism level: {level}")

    if level == 0:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print("  - No deterministic seeding or CUDA flags")
        return

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("  - torch.manual_seed(42)")
    print("  - np.random.seed(42)")
    print("  - torch.cuda.manual_seed_all(42)")
    print("  - cudnn.deterministic = True")
    print("  - cudnn.benchmark = False")

    if level >= 2:
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        print("  - CUDA_LAUNCH_BLOCKING=1")
        print("  - CUBLAS_WORKSPACE_CONFIG=:16:8")

    if level >= 3:
        os.environ["CUDNN_DETERMINISTIC"] = "1"
        os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
        print("  - CUDNN_DETERMINISTIC=1")
        print("  - TF_CUDNN_USE_AUTOTUNE=0")


def detect_system_info() -> dict:
    """Detect basic system and GPU information."""
    info = {}
    try:
        info['platform'] = platform.platform()
        info['hostname'] = platform.node()
        info['python_version'] = platform.python_version()
    except Exception:
        pass

    info['torch_version'] = getattr(torch, '__version__', None)
    info['cuda_available'] = torch.cuda.is_available()

    gpus = []
    if info['cuda_available']:
        try:
            info['cuda_version'] = torch.version.cuda
        except Exception:
            info['cuda_version'] = None
        try:
            count = torch.cuda.device_count()
            for i in range(count):
                try:
                    name = torch.cuda.get_device_name(i)
                except Exception:
                    name = None
                try:
                    mem = torch.cuda.get_device_properties(i).total_memory
                except Exception:
                    mem = None
                gpus.append({'index': i, 'name': name, 'total_memory_bytes': mem})
        except Exception:
            pass
    else:
        # Fallback to nvidia-smi if available
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True, stderr=subprocess.DEVNULL)
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(',')]
                if parts:
                    g = {'name': parts[0]}
                    if len(parts) > 1:
                        g['memory'] = parts[1]
                    gpus.append(g)
        except Exception:
            pass

    info['gpus'] = gpus
    return info


# ============================================================================
# RELIABLE VIDEO FRAME READER (Production Grade)
# ============================================================================

class ReliableVideoReader:
    """
    Production-grade video frame reader using OpenCV.

    Principle: Count ACTUAL decoded frames, never rely on metadata.

    This ensures consistent behavior across all systems and codecs
    (local, DGX, cloud, etc.) regardless of video encoding issues.
    """

    def __init__(self, video_path: str):
        """
        Initialize video reader.

        Args:
            video_path: Path to video file

        Raises:
            RuntimeError: If video cannot be opened
        """
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        # Get metadata (for information only, not for counting)
        self.metadata_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.metadata_frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.metadata_fps <= 0:
            self.metadata_fps = 30

        # Real counters (what matters!)
        self.actual_frame_count = 0
        self.read_errors = 0

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], int]:
        """
        Read next real frame from video.

        Returns:
            Tuple of (success, frame, frame_index)
            - success: True if frame was successfully read
            - frame: numpy array or None
            - frame_index: current count of successfully decoded frames
        """
        ret, frame = self.cap.read()

        if ret and frame is not None and frame.size > 0:
            self.actual_frame_count += 1
            return True, frame, self.actual_frame_count
        else:
            if ret:  # Frame read but corrupted
                self.read_errors += 1
            return False, None, self.actual_frame_count

    def get_info(self) -> dict:
        """Get video information."""
        return {
            "filename": self.video_path.name,
            "resolution": f"{self.width}x{self.height}",
            "fps": self.metadata_fps,
            "metadata_frame_count": self.metadata_frame_count,
        }

    def get_actual_frame_count(self) -> int:
        """Get count of successfully decoded frames."""
        return self.actual_frame_count

    def get_read_errors(self) -> int:
        """Get count of corrupted/failed frame reads."""
        return self.read_errors

    def close(self):
        """Release video capture."""
        self.cap.release()


# ============================================================================
# DETECTION UTILITIES
# ============================================================================

def postprocess_output(results, frame: np.ndarray) -> np.ndarray:
    """Draw detection results on frame."""
    return results[0].plot()


def load_model(model_path: str) -> YOLO:
    """Load YOLO model."""
    return YOLO(model_path)


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def main(video_path: str, model_path: str, playback_speed: float = 1.0,
         output_dir: Optional[str] = None, determinism_level: int = DETERMINISM_LEVEL):
    """
    Main fault detection pipeline.

    Args:
        video_path: Path to input video
        model_path: Path to YOLO model
        playback_speed: Display speed multiplier
        output_dir: Output directory for saving frames
        determinism_level: Selected determinism level (0-3)
    """

    total_start_time = time.time()

    print("\n" + "="*70)
    print("FAULT DETECTION - PRODUCTION PIPELINE")
    print("="*70)

    setup_determinism(determinism_level)

    # ========================================================================
    # STEP 1: Load Model
    # ========================================================================
    print("\n📦 Loading model...")
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    model_extension = Path(model_path).suffix.lower().lstrip('.')
    model_format = {
        'pt': 'PyTorch',
        'onnx': 'ONNX',
        'engine': 'TensorRT Engine'
    }.get(model_extension, model_extension.upper())

    # Determine device
    is_pytorch_model = model_extension == 'pt'
    is_onnx_model = model_extension == 'onnx'

    if is_onnx_model:
        device = "cpu"
        print("⚠️  ONNX uses CPU inference (CUDA library compatibility)")
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if is_pytorch_model:
        model.to(device)

    print(f"✓ Model type: {model_format}")
    if device.startswith("cuda"):
        print(f"✓ Device: GPU ({device})")
    else:
        print(f"✓ Device: CPU")

    # ========================================================================
    # STEP 2: Open Video (Reliable Frame Reading)
    # ========================================================================
    print("\n📹 Opening video...")
    try:
        video = ReliableVideoReader(video_path)
    except Exception as e:
        print(f"❌ Error opening video: {e}")
        return

    video_info = video.get_info()
    print(f"✓ File: {video_info['filename']}")
    print(f"✓ Resolution: {video_info['resolution']}")
    print(f"✓ FPS: {video_info['fps']}")
    print(f"✓ Metadata claims: {video_info['metadata_frame_count']} frames")
    print(f"  (Will count actual decoded frames)")

    fps = video.metadata_fps

    # ========================================================================
    # STEP 3: Create Output Directory
    # ========================================================================
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(output_dir) / f"run_{timestamp}"
        faults_dir = run_dir / "faults"
        faults_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n💾 Output directory: {run_dir}")
    else:
        run_dir = None
        faults_dir = None

    # ========================================================================
    # STEP 4: Display Configuration
    # ========================================================================
    frame_delay = int((1000 / fps) / playback_speed) if fps > 0 else 33

    print(f"\n⚙️  Configuration:")
    print(f"  Playback speed: {playback_speed}x")
    print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
    print(f"\nPress 'q' to quit\n")
    print("-"*70)

    # ========================================================================
    # STEP 5: Main Processing Loop
    # ========================================================================

    frame_idx = 0
    detection_count = 0
    frames_with_detections = 0
    inference_times = []

    while True:
        success, frame, actual_frame_idx = video.read_frame()

        if not success:
            if frame_idx > 0:
                print(f"\n✓ End of video")
            break

        frame_idx += 1

        # Run inference with timing
        inf_start = time.time()
        results = model(frame, conf=CONFIDENCE_THRESHOLD, device=device)
        inf_time = (time.time() - inf_start) * 1000
        inference_times.append(inf_time)

        # Process detections
        if len(results[0].boxes) > 0:
            num_detections = len(results[0].boxes)
            detection_count += num_detections
            frames_with_detections += 1

            # Save frame
            if faults_dir:
                confidence = float(results[0].boxes.conf.max())
                filename = f"frame_{actual_frame_idx:06d}_{num_detections}_{confidence:.2f}.jpg"
                faults_path = faults_dir / filename
                cv2.imwrite(str(faults_path), frame)

            print(f"Frame {actual_frame_idx:6d} | {num_detections:2d} det | {inf_time:7.2f}ms | SAVED")

        # Display frame
        processed_frame = postprocess_output(results, frame)
        cv2.imshow('Fault Detection', processed_frame)

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            print("\n⚠️  Stopped by user")
            break

    # ========================================================================
    # STEP 6: Cleanup
    # ========================================================================
    video.close()
    cv2.destroyAllWindows()
    total_end_time = time.time()
    total_elapsed_time = total_end_time - total_start_time

    # ========================================================================
    # STEP 7: Performance Metrics
    # ========================================================================
    actual_frame_count = video.get_actual_frame_count()
    read_errors = video.get_read_errors()

    print("\n" + "="*70)
    print("PERFORMANCE METRICS")
    print("="*70)

    # Frame count analysis
    print(f"\n📊 Frame Count:")
    print(f"  Metadata claimed: {video_info['metadata_frame_count']:,}")
    print(f"  Actually decoded: {actual_frame_count:,}")
    print(f"  Read errors: {read_errors}")

    if actual_frame_count != video_info['metadata_frame_count']:
        discrepancy = actual_frame_count / video_info['metadata_frame_count']
        print(f"  Discrepancy: {discrepancy:.2f}x (codec/metadata issue)")

    # Detections
    print(f"\n🔍 Detections:")
    print(f"  Total: {detection_count:,}")
    print(f"  Frames with detections: {frames_with_detections}")

    # Inference timing
    if inference_times:
        avg_inf = np.mean(inference_times)
        min_inf = np.min(inference_times)
        max_inf = np.max(inference_times)
        total_inf = sum(inference_times)

        print(f"\n⏱️  Inference Timing:")
        print(f"  Average: {avg_inf:.2f} ms/frame")
        print(f"  Min/Max: {min_inf:.2f} / {max_inf:.2f} ms")
        print(f"  Total: {total_inf/1000:.2f} seconds")

        video_duration = actual_frame_count / fps if fps > 0 else 0
        achieved_fps = 1000 / avg_inf if avg_inf > 0 else 0

        print(f"\n📈 Efficiency:")
        print(f"  Video duration: {video_duration:.2f}s")
        print(f"  Achieved FPS: {achieved_fps:.1f} fps")
        print(f"  Required FPS: {fps:.1f} fps")
        if achieved_fps >= fps:
            print(f"  ✓ Real-time: YES")
        else:
            print(f"  ✓ Real-time: NO ({fps/achieved_fps:.2f}x speedup needed)")

        print(f"\n⏳ Execution Time:")
        print(f"  Total: {total_elapsed_time:.2f}s")
        print(f"  Speedup: {video_duration / total_elapsed_time:.2f}x")

    # Save metrics to JSON only
    if run_dir:
        json_file = run_dir / "metrics.json"

        # gather system info and determinism settings
        system_info = detect_system_info()
        determinism_info = {
            "level": determinism_level,
            "cudnn_deterministic": bool(getattr(torch.backends.cudnn, 'deterministic', None)),
            "cudnn_benchmark": bool(getattr(torch.backends.cudnn, 'benchmark', None)),
            "seeds": {
                "torch": 42,
                "numpy": 42,
                "cuda_all": 42,
            },
            "env": {
                k: os.environ.get(k) for k in (
                    'CUDA_LAUNCH_BLOCKING', 'CUBLAS_WORKSPACE_CONFIG', 'CUDNN_DETERMINISTIC', 'TF_CUDNN_USE_AUTOTUNE'
                ) if os.environ.get(k) is not None
            }
        }

        # compute durations
        total_inference_seconds = (sum(inference_times) / 1000) if inference_times else 0
        video_duration_seconds = (actual_frame_count / fps) if fps > 0 else 0
        processing_overhead_seconds = total_elapsed_time - total_inference_seconds
        processing_overhead_percent = (processing_overhead_seconds / total_elapsed_time * 100) if total_elapsed_time > 0 else 0

        model_file = Path(model_path).name if model_path else None
        model_ext = Path(model_path).suffix if model_path else None

        metrics_data = {
            "timestamp": datetime.now().isoformat(),
            "model": {
                "file": model_file,
                "type": model_ext,
                "path": model_path,
                "confidence_threshold": CONFIDENCE_THRESHOLD,
            },
            "video": {
                "filename": video_info.get('filename'),
                "resolution": video_info.get('resolution'),
                "fps": video_info.get('fps'),
                "metadata_frame_count": video_info.get('metadata_frame_count'),
                "total_frames_processed": actual_frame_count,
                "original_duration_seconds": round(video_duration_seconds, 2),
                "original_duration_readable": f"{video_duration_seconds:.2f} seconds ({video_duration_seconds/60:.2f} min)",
                "original_duration_hours": round(video_duration_seconds/3600, 2),
            },
            "detection_summary": {
                "total_frames_processed": actual_frame_count,
                "total_detections_found": detection_count,
            },
            "inference": {
                "avg_ms": float(np.mean(inference_times)) if inference_times else 0,
                "min_ms": float(np.min(inference_times)) if inference_times else 0,
                "max_ms": float(np.max(inference_times)) if inference_times else 0,
                "total_seconds": total_inference_seconds,
            },
            "execution": {
                "total_elapsed_seconds": total_elapsed_time,
                "processing_overhead_seconds": processing_overhead_seconds,
                "processing_overhead_percent": round(processing_overhead_percent, 2),
                "device": device,
            },
            "system": system_info,
            "determinism": determinism_info,
        }

        with open(json_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)

        print(f"✓ JSON data saved to: {json_file}")

    print("\n" + "="*70 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Fault detection video pipeline with determinism control")
    parser.add_argument("--video-path", default=VIDEO_PATH, help="Input video path")
    parser.add_argument("--model-path", default=MODEL_PATH, help="YOLO model path")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory for metrics and frames")
    parser.add_argument("--playback-speed", type=float, default=PLAYBACK_SPEED, help="Playback speed multiplier")
    parser.add_argument("--determinism-level", type=int, choices=[0, 1, 2, 3], default=DETERMINISM_LEVEL,
                        help="Determinism level: 0=none, 1=basic, 2=GPU-level, 3=maximum")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        video_path=args.video_path,
        model_path=args.model_path,
        playback_speed=args.playback_speed,
        output_dir=args.output_dir,
        determinism_level=args.determinism_level,
    )
