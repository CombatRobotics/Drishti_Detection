import cv2
import os
from datetime import datetime

# Configuration variables - Edit these as needed
VIDEO_PATH = r"D:\rosbags_17thJan\ace_delhi_1_return_20260115_044439\ace_delhi_1_return_20260115_04443920260117_125830.avi"  # Path to your input video file
OUTPUT_DIR = r"D:\rosbags_17thJan\ace_delhi_1_return_20260115_044439\Frames"                   # Base directory for saving extracted frames
IMAGE_FORMAT = "png"                                                            # Output image format: jpg, png, bmp
JPEG_QUALITY = 95                                                               # JPEG quality (1-100), only used if IMAGE_FORMAT is jpg
USE_CUDA = True                                                                 # Use CUDA decode if available


def extract_frames(video_path, output_dir, image_format="jpg", jpeg_quality=95):
    """
    Extract all frames from a video file and save them to the output directory.
    
    Args:
        video_path: Path to the input video file
        output_dir: Base directory for saving extracted frames
        image_format: Output image format (jpg, png, bmp)
        jpeg_quality: JPEG quality (1-100), only used for jpg format
    """
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Open video capture (prefer CUDA decode if available)
    use_cuda = False
    cuda_reader = None
    cap = None
    if USE_CUDA and hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        try:
            cuda_reader = cv2.cudacodec.createVideoReader(video_path)
            use_cuda = True
            print("Using CUDA video decode")
        except Exception as exc:
            print(f"CUDA decode unavailable, falling back to CPU ({exc})")
            use_cuda = False

    if not use_cuda:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
    
    # Get video properties
    if use_cuda:
        video_fps = 0
    else:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("Warning: Could not detect video FPS, defaulting to 30")
        video_fps = 30
    
    if use_cuda:
        total_frames = 0
        width = 0
        height = 0
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / video_fps if video_fps > 0 else 0
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output directory with video name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    frames_dir = os.path.join(output_dir, f"{video_name}_{timestamp}")
    os.makedirs(frames_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("VIDEO FRAME EXTRACTOR")
    print("=" * 60)
    print(f"\nVideo Info:")
    print(f"  File: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {video_fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"\nOutput:")
    print(f"  Directory: {frames_dir}")
    print(f"  Format: {image_format.upper()}")
    if image_format.lower() == "jpg":
        print(f"  JPEG Quality: {jpeg_quality}")
    print("\n" + "-" * 60)
    print("Extracting frames...")
    print("-" * 60)
    
    frame_count = 0
    saved_count = 0
    
    while True:
        if use_cuda:
            ret, gpu_frame = cuda_reader.nextFrame()
            if not ret:
                break
            frame = gpu_frame.download()
        else:
            ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Generate filename with zero-padded frame number
        # Calculate padding based on total frames
        padding = len(str(total_frames)) if total_frames > 0 else 6
        filename = f"frame_{frame_count:0{padding}d}.{image_format}"
        filepath = os.path.join(frames_dir, filename)
        
        # Save the frame
        if image_format.lower() == "jpg":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        elif image_format.lower() == "png":
            cv2.imwrite(filepath, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        else:
            cv2.imwrite(filepath, frame)
        
        saved_count += 1
        
        # Print progress every 100 frames or at specific percentages
        if frame_count % 100 == 0 or (total_frames > 0 and frame_count == total_frames):
            if total_frames > 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)")
            else:
                print(f"  Progress: {frame_count} frames")
    
    # Clean up
    if cap is not None:
        cap.release()
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  Total frames extracted: {saved_count}")
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

