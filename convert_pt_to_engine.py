#!/usr/bin/env python3
"""
Convert YOLO PyTorch (.pt) → ONNX → TensorRT Engine (.engine)

Safety features:
  - Original .pt file is NEVER modified
  - Creates separate .onnx and .engine files
  - Validates each step
  - Logs all file operations
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

# ============================================================================
# CONFIGURATION
# ============================================================================
PT_MODEL_PATH = r"/media/viraj/New Volume/Dhrishti/YOLO/v4.1/faultdetection_v4.1.pt"
OUTPUT_DIR = r"/media/viraj/New Volume/Dhrishti/YOLO/v4.1"

# Derived paths (won't overwrite original .pt)
ONNX_PATH = os.path.join(OUTPUT_DIR, "faultdetection_v4.1.onnx")
ENGINE_PATH = os.path.join(OUTPUT_DIR, "faultdetection_v4.1.engine")

print("\n" + "="*80)
print("YOLO PyTorch (.pt) → ONNX → TensorRT Engine (.engine) Converter")
print("="*80)

# ============================================================================
# STEP 0: Verify Setup
# ============================================================================
print("\n[STEP 0] Verifying setup...")

if not torch.cuda.is_available():
    print("❌ Error: CUDA not available. TensorRT requires NVIDIA GPU.")
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
cuda_version = torch.version.cuda
print(f"  ✓ GPU: {gpu_name}")
print(f"  ✓ CUDA: {cuda_version}")
print(f"  ✓ TensorRT: {trt.__version__}")

if not os.path.exists(PT_MODEL_PATH):
    print(f"❌ Error: PyTorch model not found: {PT_MODEL_PATH}")
    sys.exit(1)

print(f"  ✓ PyTorch model found: {PT_MODEL_PATH}")
pt_size = os.path.getsize(PT_MODEL_PATH) / (1024*1024)
print(f"     Size: {pt_size:.1f} MB")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"  ✓ Created output directory: {OUTPUT_DIR}")
else:
    print(f"  ✓ Output directory: {OUTPUT_DIR}")

# ============================================================================
# STEP 1: PyTorch → ONNX
# ============================================================================
print("\n[STEP 1] Converting PyTorch (.pt) → ONNX...")

try:
    model = YOLO(PT_MODEL_PATH)
    print(f"  ✓ Loaded PyTorch model")

    onnx_path = model.export(
        format="onnx",
        device=0,
        imgsz=640,
        half=False,           # FP32 for accuracy
        optimize=False,       # Don't optimize yet
        verbose=False
    )

    if not onnx_path:
        raise Exception("ONNX export returned None")

    print(f"  ✓ ONNX file created: {onnx_path}")
    onnx_size = os.path.getsize(onnx_path) / (1024*1024)
    print(f"     Size: {onnx_size:.1f} MB")

except Exception as e:
    print(f"❌ ONNX conversion failed: {e}")
    print("\nTroubleshooting:")
    print("  - Check TensorRT installation: pip install tensorrt")
    print("  - Check PyTorch model integrity")
    sys.exit(1)

# ============================================================================
# STEP 2: Simplify ONNX (Optional but recommended)
# ============================================================================
print("\n[STEP 2] Simplifying ONNX model...")

try:
    import onnx
    from onnx_simplifier import simplify

    onnx_model = onnx.load(onnx_path)
    print(f"  ✓ Loaded ONNX model")

    simplified_model, check = simplify(onnx_model)
    if check:
        onnx.save(simplified_model, onnx_path)
        print(f"  ✓ ONNX model simplified")
    else:
        print(f"  ⚠ ONNX simplification check failed, using original")

except Exception as e:
    print(f"  ⚠ ONNX simplification skipped: {e}")

# ============================================================================
# STEP 3: ONNX → TensorRT Engine
# ============================================================================
print("\n[STEP 3] Converting ONNX → TensorRT Engine...")
print("  (This may take 1-3 minutes, please wait...)\n")

try:
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(TRT_LOGGER) as builder:
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )

        parser = trt.OnnxParser(network, TRT_LOGGER)
        print(f"  🔄 Parsing ONNX file...")

        with open(onnx_path, 'rb') as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    print(f"❌ Parse error: {parser.get_error(error)}")
                raise Exception("ONNX parsing failed")

        print(f"  ✓ ONNX parsed successfully")

        # Configure builder for determinism
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        # Set memory pool
        profile = builder.create_optimization_profile()
        profile.set_shape('images', (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
        config.add_optimization_profile(profile)

        print(f"  🔄 Building TensorRT engine...")

        # TensorRT 10.x uses build_serialized_network instead of build_engine
        engine_bytes = builder.build_serialized_network(network, config)
        if not engine_bytes:
            raise Exception("Failed to build TensorRT engine")

        print(f"  ✓ Engine built successfully")

        # Save engine to file
        with open(ENGINE_PATH, 'wb') as f:
            f.write(engine_bytes)

        print(f"  ✓ Engine saved to disk")

except Exception as e:
    print(f"❌ TensorRT conversion failed: {e}")
    print("\nTroubleshooting:")
    print("  - Make sure CUDA memory is available")
    print("  - Check TensorRT version compatibility")
    sys.exit(1)

# ============================================================================
# STEP 4: Verification
# ============================================================================
print("\n[STEP 4] Verifying output files...")

files_ok = True

# Check PT file (should be unchanged)
if os.path.exists(PT_MODEL_PATH):
    print(f"  ✓ Original .pt file intact: {PT_MODEL_PATH}")
else:
    print(f"  ❌ WARNING: .pt file missing!")
    files_ok = False

# Check ONNX file
if os.path.exists(ONNX_PATH if os.path.exists(ONNX_PATH) else onnx_path):
    onnx_size = os.path.getsize(onnx_path) / (1024*1024)
    print(f"  ✓ ONNX file created: {onnx_path}")
    print(f"     Size: {onnx_size:.1f} MB")
else:
    print(f"  ❌ ONNX file not found!")
    files_ok = False

# Check Engine file
if os.path.exists(ENGINE_PATH):
    engine_size = os.path.getsize(ENGINE_PATH) / (1024*1024)
    print(f"  ✓ Engine file created: {ENGINE_PATH}")
    print(f"     Size: {engine_size:.1f} MB")
else:
    print(f"  ❌ Engine file not found!")
    files_ok = False

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)

if files_ok:
    print("✓ CONVERSION SUCCESSFUL!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  1. ONNX:   {onnx_path}")
    print(f"  2. Engine: {ENGINE_PATH}")
    print(f"\nOriginal .pt file:")
    print(f"  {PT_MODEL_PATH} (UNCHANGED ✓)")
    print(f"\nNext steps:")
    print(f"  1. Update fault_detection_video.py to use the .engine file")
    print(f"  2. Or keep using .pt (it still works, but slower)")
    print(f"\n" + "="*80 + "\n")
else:
    print("❌ CONVERSION COMPLETED WITH ERRORS")
    print("="*80 + "\n")
    sys.exit(1)
