import cv2
import os
from datetime import datetime

# Configuration variables - Edit these as needed
VIDEO_PATH = r"/media/viraj/New Volume/Dhrishti/Day 6/ace_delhi_6_10m20260124_050136/ace_delhi_6_10m20260124_050136.avi"  # Path to your input video file
OUTPUT_DIR = r"/media/viraj/New Volume/Dhrishti/Day 6/ace_delhi_6_10m20260124_050136/True_frame_count"                  # Base directory for saving extracted frames
IMAGE_FORMAT = "png"                                                            # Output image format: jpg, png, bmp
JPEG_QUALITY = 100                                                               # JPEG quality (1-100), only used if IMAGE_FORMAT is jpg
USE_CUDA = True                                                                 # Use CUDA decode if available


def detect_gpu_info():
    """Detect GPU information for CUDA video decoding."""
    try:
        if hasattr(cv2, "cuda"):
            cuda_device_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_device_count > 0:
                return f"{cuda_device_count} CUDA device(s) available"
            else:
                return "No CUDA devices detected"
        else:
            return "CUDA module not available in OpenCV"
    except Exception as e:
        return f"Error detecting GPU: {e}"


def extract_frames(video_path, output_dir, image_format="jpg", jpeg_quality=100):
    """
    Extract all frames from a video file and save them to the output directory.

    Supports both CPU and GPU (CUDA) video decoding for better performance.

    Args:
        video_path: Path to the input video file
        output_dir: Base directory for saving extracted frames
        image_format: Output image format (jpg, png, bmp)
        jpeg_quality: JPEG quality (1-100), only used for jpg format
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found: {video_path}")
        return

    print("\n" + "=" * 60)
    print("VIDEO FRAME EXTRACTOR")
    print("=" * 60)

    # Display GPU information
    print(f"\n🖥️  System Info:")
    print(f"  {detect_gpu_info()}")
    if USE_CUDA:
        print(f"  CUDA decode: ENABLED (will attempt GPU acceleration)")
    else:
        print(f"  CUDA decode: DISABLED")

    # Open video capture (prefer CUDA decode if available)
    use_cuda = False
    cuda_reader = None
    cap = None

    if USE_CUDA:
        try:
            if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                print("✓ CUDA device available, attempting hardware-accelerated video decode...")
                cuda_reader = cv2.cudacodec.createVideoReader(video_path)
                use_cuda = True
                print("✓ Using CUDA video decode")
            else:
                print("⚠ No CUDA device found, using CPU decode")
        except Exception as exc:
            print(f"⚠ CUDA decode failed ({exc}), falling back to CPU decode")
            use_cuda = False

    if not use_cuda:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video file {video_path}")
            return
        print("✓ Using CPU video decode")
    
    # Get video properties (use a metadata-only CPU capture when decoding on GPU)
    video_fps = 0
    metadata_frame_count = 0
    width = 0
    height = 0
    if use_cuda:
        meta_cap = cv2.VideoCapture(video_path)
        if meta_cap.isOpened():
            video_fps = meta_cap.get(cv2.CAP_PROP_FPS)
            metadata_frame_count = int(meta_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(meta_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(meta_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta_cap.release()
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        metadata_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_fps <= 0:
        print("Warning: Could not detect video FPS, defaulting to 30")
        video_fps = 30
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directory with video name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(output_dir, f"{video_name}_{timestamp}")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"\n📹 Video Info:")
    print(f"  File: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Metadata claims: {metadata_frame_count} frames")
    print(f"  (Will count actual decoded frames)")

    print(f"\n💾 Output:")
    print(f"  Directory: {frames_dir}")
    print(f"  Format: {image_format.upper()}")
    if image_format.lower() == "jpg":
        print(f"  JPEG Quality: {jpeg_quality}")

    print("\n" + "-" * 60)
    print("Extracting frames...")
    print("-" * 60)
    
    frame_count = 0
    saved_count = 0
    pending_frame = None

    if use_cuda and (width == 0 or height == 0):
        ret, gpu_frame = cuda_reader.nextFrame()
        if not ret:
            print("Error: Could not read from CUDA decoder")
            return
        pending_frame = gpu_frame.download()
        height, width = pending_frame.shape[:2]
    
    while True:
        if pending_frame is not None:
            frame = pending_frame
            pending_frame = None
            ret = True
        elif use_cuda:
            ret, gpu_frame = cuda_reader.nextFrame()
            if not ret:
                break
            frame = gpu_frame.download()
        else:
            ret, frame = cap.read()
            if not ret:
                break
        
        frame_count += 1

        # Generate filename with zero-padded frame number (fixed 6-digit padding)
        filename = f"frame_{frame_count:06d}.{image_format}"
        filepath = os.path.join(frames_dir, filename)

        # Save the frame
        if image_format.lower() == "jpg":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        elif image_format.lower() == "png":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(filepath, frame)

        saved_count += 1

        # Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"  Extracted: {frame_count} frames")
    
    # Clean up
    if cap is not None:
        cap.release()
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Metadata claimed: {metadata_frame_count}")
    print(f"  Actually decoded: {saved_count}")
    if saved_count != metadata_frame_count:
        discrepancy = saved_count / metadata_frame_count if metadata_frame_count > 0 else 0
        print(f"  Discrepancy: {discrepancy:.2f}x (codec/metadata issue)")
    print(f"  Output directory: {frames_dir}")
    print(f"  Frames per second in source: {video_fps:.2f}")
    print("=" * 60 + "\n")
    
    return frames_dir


def main():
    """Main function to run the frame extraction."""
    extract_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        image_format=IMAGE_FORMAT,
        jpeg_quality=JPEG_QUALITY
    )


if __name__ == "__main__":
    main()

