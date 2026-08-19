# Dhrishti Rail Fault Detection & Processing Pipeline

A comprehensive computer vision pipeline for detecting, deduplicating, and geometrically standardizing rail head defects from video footage. Integrates YOLO-based detection, spatial deduplication, geometric normalization, and Adobe Lightroom for batch editing.

## Overview

This project implements a complete end-to-end pipeline for rail inspection:

1. **Fault Detection** — Extract defective frames from video using YOLO
2. **Deduplication** — Remove spatial and temporal redundancy
3. **Geometric Normalization** — Deskew, center, and standardize resolution
4. **Lightroom Integration** — Batch import standardized frames for editing

---

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

**Required:**
- **FFmpeg** — Essential for MJPEG video preprocessing (ensures reliable frame-by-frame decoding across all video codecs)
  - Install on Ubuntu: `sudo apt-get install ffmpeg`
  - Install on macOS: `brew install ffmpeg`
  - Install on Windows: Download from https://ffmpeg.org/download.html

**Supported Model Formats for Inference:**
- **PyTorch** (`.pt` files) — Full YOLO model format
- **TensorRT Engine** (`.engine` files) — Optimized inference format for NVIDIA GPUs

**Model Conversion (PyTorch → TensorRT):**
- ONNX tools are included for model conversion workflow
- Inference via ONNX is **not supported** — Use PyTorch or TensorRT instead

---

## Stage 1: Fault Detection from Video

**Script:** `fault_detection_video.py`

Processes a video file using a YOLO fault detection model with automatic MJPEG preprocessing for reliable frame-by-frame decoding. Runs inference on every frame and displays results in real-time. Automatically saves frames with detections to timestamped output directories.

**Key Innovation: MJPEG Preprocessing**
- Automatically transcodes input video to MJPEG format via FFmpeg
- Ensures OpenCV reads every frame exactly once (no frame loss)
- Handles all video codecs (H.264, MPEG-4, HEVC, AV1, etc.) with consistent results
- Solves metadata ambiguity issues that cause frame count discrepancies

### Configuration

Edit the configuration variables at the top of `fault_detection_video.py`:

```python
VIDEO_PATH = r"D:\Drishti\basler_1767303407.avi"  # Input video file (any format)
MODEL_PATH = r"D:\Drishti\model.pt"               # YOLO model (.pt or .engine)
OUTPUT_DIR = r"D:\Drishti\Output"                 # Output directory for detected frames
CONFIDENCE_THRESHOLD = 0.425                       # Detection confidence threshold
DETERMINISM_LEVEL = 2                             # 0=none, 1=basic, 2=GPU-level, 3=maximum
```

### Features

- **MJPEG preprocessing** — Automatic transcoding for deterministic frame reading
- **Codec-agnostic** — Works with any video format (AVI, MP4, MOV, MKV, etc.)
- **Real-time YOLO inference** with live bounding box visualization
- **GPU-accelerated** — Supports PyTorch and TensorRT Engine models
- **Deterministic results** — Reproducible across systems (local, DGX, cloud)
- **Organized output** — Timestamped folders with comprehensive metrics

### Output

```
Output/run_20260102_143022/
├── frame_001_detection.jpg
├── frame_002_detection.jpg
└── ...
```

Each frame contains:
- Original frame with YOLO bounding boxes overlaid
- Detection class labels and confidence scores

### Controls

- Press 'q' to quit video processing

---

## Stage 2: Deduplication

After fault detection, the extracted frames often contain duplicate or near-duplicate detections due to temporal overlap in video frames. This stage intelligently deduplicates based on spatial, temporal, or combined spatiotemporal information while preserving unique defects.

### Deduplication Scripts

- **Spatial Similarity** (`deduplicate_spatial_similarity_with_category.py`) — Removes spatially similar detections within a threshold distance, optionally filtering by defect category
- **Temporal Deduplication** (`deduplicate_temporal.py`) — Removes consecutive frame duplicates
- **Spatiotemporal Clustering** (`deduplicate_spatiotemporal_faults.py`) — Combines spatial and temporal information for holistic deduplication

### Usage

**Spatial deduplication (all categories):**
```bash
python3 deduplicate_spatial_similarity_with_category.py /path/to/detected_frames/
```

**Filter by defect category:**
```bash
python3 deduplicate_spatial_similarity_with_category.py /path/to/detected_frames/ --category crack
```

**Temporal deduplication:**
```bash
python3 deduplicate_temporal.py /path/to/detected_frames/
```

**Spatiotemporal deduplication:**
```bash
python3 deduplicate_spatiotemporal_faults.py /path/to/detected_frames/
```

### Output

```
deduplicated_output/run_20260102_143022/
├── frame_001_unique.jpg
├── frame_005_unique.jpg
└── metadata.jsonl
```

---

## Stage 3: Geometric Normalization (Work in Progress)

**Status:** This stage is currently under development. Scripts for deskewing and centering deduplicated frames are available in `stage1_deskew_robust.py` and related utilities.

---

## Stage 4: Adobe Lightroom Integration

The output from the deskewed and centered script is used as input for editing in Adobe Lightroom.

---

## GPU Determinism

When running deep learning models on NVIDIA GPUs, GPU-to-GPU determinism is enabled to ensure consistent results across different NVIDIA cards. This means that the same input will produce identical outputs when executed on any NVIDIA GPU, which is critical for reproducibility and reliability in fault detection workflows.

Floating-point operations on GPUs can vary slightly between different hardware configurations due to differences in how reductions and operations are implemented. NVIDIA provides mechanisms to control this behavior. For more information on controlling floating-point determinism in NVIDIA CUDA, see: https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/#:~:text=CUB%20in%20NVIDIA%20CUDA%20Core,levels%20via%20the%20execution%20environment

---

## Usage Examples

### Basic fault detection (with MJPEG preprocessing):
```bash
python3 fault_detection_video.py
```

### Fault detection with custom paths:
```bash
python3 fault_detection_video.py \
  --video-path /path/to/video.avi \
  --model-path /path/to/model.pt \
  --output-dir /path/to/output \
  --confidence-threshold 0.425 \
  --determinism-level 2
```

### Complete pipeline:
```bash
# 1. Extract faults from video (with automatic MJPEG preprocessing)
python3 fault_detection_video.py

# 2. Deduplicate detections (choose one strategy)
python3 deduplicate_spatial_similarity_with_category.py ./Output/run_20260102_143022/
# or use temporal deduplication
python3 deduplicate_temporal.py ./Output/run_20260102_143022/
# or use spatiotemporal clustering
python3 deduplicate_spatiotemporal_faults.py ./Output/run_20260102_143022/

# 3. Extract frames for geometric normalization (work in progress)
python3 video_frame_extractor.py

# 4. (Optional) Lightroom integration
python3 post-process/Lightroom_API/lightroom_api.py  # Authenticate first
```

---

## Notes

- Frame rate is automatically extracted from video metadata using OpenCV
- Deduplication strategies can be chosen based on your data characteristics (spatial, temporal, or spatiotemporal)
- The pipeline is designed for rail inspection but generalizes to other defect detection scenarios
- Processing speed scales with available CPU cores