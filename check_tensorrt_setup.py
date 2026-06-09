#!/usr/bin/env python3
"""
Comprehensive TensorRT Installation & Setup Verification
Run this to check if your system is ready for model conversion
"""

import sys
from pathlib import Path

print("\n" + "="*70)
print("TensorRT Installation & Setup Verification")
print("="*70)

checks_passed = 0
checks_total = 0

# ============================================================================
# Check: TensorRT
# ============================================================================
checks_total += 1
try:
    import tensorrt as trt
    print(f"\n✓ TensorRT: {trt.__version__}")
    checks_passed += 1
except ImportError as e:
    print(f"\n✗ TensorRT: NOT INSTALLED")
    print(f"  Error: {e}")
    print(f"  Fix: pip install tensorrt")

# ============================================================================
# Check: PyTorch & CUDA
# ============================================================================
checks_total += 1
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"\n✓ PyTorch: {torch.__version__}")
    print(f"  - CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
        try:
            print(f"  - cuDNN version: {torch.backends.cudnn.version()}")
        except:
            print(f"  - cuDNN version: Unknown")
        checks_passed += 1
    else:
        print(f"  ✗ CUDA not available (GPU not detected)")
except ImportError as e:
    print(f"\n✗ PyTorch: NOT INSTALLED")
    print(f"  Error: {e}")

# ============================================================================
# Check: PyCUDA
# ============================================================================
checks_total += 1
try:
    import pycuda
    import pycuda.driver as cuda
    cuda.init()
    device = cuda.Device(0)
    print(f"\n✓ PyCUDA: {pycuda.VERSION}")
    print(f"  - Device: {device.name()}")
    checks_passed += 1
except ImportError as e:
    print(f"\n✗ PyCUDA: NOT INSTALLED")
    print(f"  Error: {e}")
    print(f"  Fix: pip install pycuda")
except Exception as e:
    print(f"\n✗ PyCUDA: ERROR")
    print(f"  Error: {e}")

# ============================================================================
# Check: ONNX
# ============================================================================
checks_total += 1
try:
    import onnx
    print(f"\n✓ ONNX: {onnx.__version__}")
    checks_passed += 1
except ImportError as e:
    print(f"\n✗ ONNX: NOT INSTALLED")
    print(f"  Error: {e}")
    print(f"  Fix: pip install onnx")

# ============================================================================
# Check: ONNX Simplifier
# ============================================================================
checks_total += 1
try:
    import onnxsim
    print(f"\n✓ ONNX-Simplifier: installed")
    checks_passed += 1
except ImportError as e:
    print(f"\n✗ ONNX-Simplifier: NOT INSTALLED")
    print(f"  Error: {e}")
    print(f"  Fix: pip install onnx-simplifier")

# ============================================================================
# Check: ONNX Runtime
# ============================================================================
checks_total += 1
try:
    import onnxruntime as rt
    print(f"\n✓ ONNX Runtime: {rt.__version__}")
    print(f"  - Providers: {rt.get_available_providers()}")
    checks_passed += 1
except ImportError as e:
    print(f"\n✗ ONNX Runtime: NOT INSTALLED")
    print(f"  Error: {e}")
    print(f"  Fix: pip install onnxruntime")

# ============================================================================
# Check: Ultralytics
# ============================================================================
checks_total += 1
try:
    from ultralytics import YOLO
    print(f"\n✓ Ultralytics YOLO: installed")
    checks_passed += 1
except ImportError as e:
    print(f"\n✗ Ultralytics: NOT INSTALLED")
    print(f"  Error: {e}")
    print(f"  Fix: pip install ultralytics")

# ============================================================================
# Check: OpenCV
# ============================================================================
checks_total += 1
try:
    import cv2
    print(f"\n✓ OpenCV: {cv2.__version__}")
    checks_passed += 1
except ImportError as e:
    print(f"\n✗ OpenCV: NOT INSTALLED")
    print(f"  Error: {e}")
    print(f"  Fix: pip install opencv-python")

# ============================================================================
# Check: NumPy
# ============================================================================
checks_total += 1
try:
    import numpy as np
    print(f"\n✓ NumPy: {np.__version__}")
    checks_passed += 1
except ImportError as e:
    print(f"\n✗ NumPy: NOT INSTALLED")
    print(f"  Error: {e}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print(f"Results: {checks_passed}/{checks_total} checks passed")
print("="*70)

if checks_passed == checks_total:
    print("\n✓ SUCCESS! Your system is ready for model conversion.")
    print("\nNext steps:")
    print("  1. Run: python convert_pt_to_engine.py")
    print("  2. Wait for conversion to complete")
    print("  3. Use the .engine file in fault_detection_video.py")
    print("\n" + "="*70 + "\n")
    sys.exit(0)

else:
    print(f"\n✗ FAILED! {checks_total - checks_passed} dependencies missing.")
    print("\nTo fix, run:")
    print("  pip install -r requirements.txt")
    print("\nOr install missing packages individually:")
    print("  pip install tensorrt pycuda onnx onnx-simplifier onnxruntime")
    print("\n" + "="*70 + "\n")
    sys.exit(1)
