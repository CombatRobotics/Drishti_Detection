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

**Optional:**
- **FFmpeg** — If you want to cross-check frame decoding from videos independently, you can install FFmpeg on your system
  - Install on Ubuntu: `sudo apt-get install ffmpeg`
  - Install on macOS: `brew install ffmpeg`
  - Install on Windows: Download from https://ffmpeg.org/download.html

---

## Stage 1: Fault Detection from Video

**Script:** `fault_detection_video.py`

Processes a video file using a YOLO fault detection model, running inference on each frame and displaying results in real-time. Automatically saves frames with detections to timestamped output directories. Frame rate is automatically extracted from the video metadata using OpenCV, which handles inconsistent frame rates gracefully.

### Configuration

Edit the configuration variables at the top of `fault_detection_video.py`:

```python
VIDEO_PATH = r"D:\Drishti\basler_1767303407.mp4"  # Input video file
MODEL_PATH = r"D:\Drishti\135epochs.pt"          # YOLO model (.pt file)
OUTPUT_DIR = r"D:\Drishti\Output"                # Output directory for detected frames
```

### Features

- **Automatic frame extraction** using OpenCV (works with inconsistent frame rates)
- **Real-time YOLO inference** with live bounding box visualization
- **Organized output** — Timestamped folders (e.g., `run_20260102_143022`)

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

## Complete Pipeline Example

```bash
# 1. Extract faults from video
python3 fault_detection_video.py

# 2. Deduplicate detections (choose one strategy)
python3 deduplicate_spatial_similarity_with_category.py ./Output/run_20260102_143022/
# or
python3 deduplicate_temporal.py ./Output/run_20260102_143022/
# or
python3 deduplicate_spatiotemporal_faults.py ./Output/run_20260102_143022/

# 3. (Optional) Lightroom integration
python3 post-process/Lightroom_API/lightroom_api.py  # Authenticate first
```

---

## Notes

- Frame rate is automatically extracted from video metadata using OpenCV
- Deduplication strategies can be chosen based on your data characteristics (spatial, temporal, or spatiotemporal)
- The pipeline is designed for rail inspection but generalizes to other defect detection scenarios
- Processing speed scales with available CPU cores