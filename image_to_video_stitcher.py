import os
import subprocess
import re

# Configuration variables - Edit these as needed
FFMPEG_PATH = r"C:\Users\Viraj\Downloads\ffmpeg-8.0-full_build\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
IMAGE_FOLDER = r"D:\Path\To\Your\Image\Folder"         # Path to the folder containing the images
FIRST_FRAME_NAME = "Image.0001.png"                     # Name of the first frame (e.g., "Image.0001.png", "frame_0001.jpg")
TOTAL_FRAMES = 100                                      # Total number of frames
FRAME_RATE = 30                                         # Output video frame rate (FPS)
OUTPUT_VIDEO_NAME = "stitched_video.mp4"                # Output video filename (will be saved in IMAGE_FOLDER)


def parse_frame_pattern(first_frame_name):
    """
    Parse the first frame name to extract the naming pattern for ffmpeg.
    
    Supports patterns like:
    - Image.0001.png -> Image.%04d.png
    - frame_0001.jpg -> frame_%04d.jpg
    - MySequence_001.png -> MySequence_%03d.png
    - 0001.png -> %04d.png
    
    Returns:
        tuple: (pattern, start_number, extension)
    """
    # Find the numeric part in the filename
    match = re.search(r'(\d+)(?=\.[^.]+$)', first_frame_name)
    
    if not match:
        raise ValueError(f"Could not find frame number in filename: {first_frame_name}")
    
    number_str = match.group(1)
    start_number = int(number_str)
    num_digits = len(number_str)
    
    # Get the prefix (everything before the number)
    prefix = first_frame_name[:match.start()]
    
    # Get the extension (everything after the number)
    extension = first_frame_name[match.end():]
    
    # Create the ffmpeg pattern
    pattern = f"{prefix}%0{num_digits}d{extension}"
    
    return pattern, start_number, extension


def stitch_images_to_video(image_folder, first_frame_name, total_frames, frame_rate, 
                           output_video_name, ffmpeg_path):
    """
    Stitch images into a video using ffmpeg.
    
    Args:
        image_folder: Path to the folder containing the images
        first_frame_name: Name of the first frame
        total_frames: Total number of frames
        frame_rate: Output video frame rate
        output_video_name: Name for the output video file
        ffmpeg_path: Path to ffmpeg executable
    """
    # Check if image folder exists
    if not os.path.exists(image_folder):
        print(f"Error: Image folder not found: {image_folder}")
        return False
    
    # Check if ffmpeg exists
    if not os.path.exists(ffmpeg_path):
        print(f"Error: FFmpeg not found: {ffmpeg_path}")
        return False
    
    # Check if first frame exists
    first_frame_path = os.path.join(image_folder, first_frame_name)
    if not os.path.exists(first_frame_path):
        print(f"Error: First frame not found: {first_frame_path}")
        return False
    
    # Parse the naming pattern
    try:
        pattern, start_number, extension = parse_frame_pattern(first_frame_name)
    except ValueError as e:
        print(f"Error: {e}")
        return False
    
    # Full input pattern path
    input_pattern = os.path.join(image_folder, pattern)
    
    # Output video path (same folder as images)
    output_path = os.path.join(image_folder, output_video_name)
    
    print("\n" + "=" * 60)
    print("IMAGE TO VIDEO STITCHER (Roboflow Compatible)")
    print("=" * 60)
    print(f"\nInput:")
    print(f"  Image folder: {image_folder}")
    print(f"  First frame: {first_frame_name}")
    print(f"  Detected pattern: {pattern}")
    print(f"  Start number: {start_number}")
    print(f"  Total frames: {total_frames}")
    print(f"  Frame rate: {frame_rate} FPS")
    print(f"\nOutput:")
    print(f"  Video file: {output_path}")
    print(f"  Codec: H.264 (libx264)")
    print(f"  Format: MP4")
    print("\n" + "-" * 60)
    print("Running FFmpeg...")
    print("-" * 60 + "\n")
    
    # Build ffmpeg command
    # -y: Overwrite output file without asking
    # -framerate: Input frame rate
    # -start_number: Start frame number
    # -i: Input pattern
    # -frames:v: Number of frames to encode
    # -c:v libx264: Use H.264 codec
    # -pix_fmt yuv420p: Pixel format for compatibility
    # -crf 18: Quality (lower = better, 18-23 is good)
    # -preset medium: Encoding speed/quality tradeoff
    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate", str(frame_rate),
        "-start_number", str(start_number),
        "-i", input_pattern,
        "-frames:v", str(total_frames),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "medium",
        output_path
    ]
    
    print("Command:")
    print(" ".join(f'"{c}"' if " " in c else c for c in cmd))
    print()
    
    try:
        # Run ffmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=image_folder
        )
        
        if result.returncode == 0:
            print("\n" + "=" * 60)
            print("STITCHING COMPLETE!")
            print("=" * 60)
            
            # Get output file size
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                size_mb = file_size / (1024 * 1024)
                print(f"\nOutput video: {output_path}")
                print(f"File size: {size_mb:.2f} MB")
                duration = total_frames / frame_rate
                print(f"Duration: {duration:.2f} seconds")
                print(f"\nReady for upload to Roboflow!")
            
            print("=" * 60 + "\n")
            return True
        else:
            print("\n" + "=" * 60)
            print("ERROR: FFmpeg failed!")
            print("=" * 60)
            print(f"\nStderr:\n{result.stderr}")
            print("=" * 60 + "\n")
            return False
            
    except Exception as e:
        print(f"\nError running FFmpeg: {e}")
        return False


def main():
    """Main function to run the image stitching."""
    stitch_images_to_video(
        image_folder=IMAGE_FOLDER,
        first_frame_name=FIRST_FRAME_NAME,
        total_frames=TOTAL_FRAMES,
        frame_rate=FRAME_RATE,
        output_video_name=OUTPUT_VIDEO_NAME,
        ffmpeg_path=FFMPEG_PATH
    )


if __name__ == "__main__":
    main()

