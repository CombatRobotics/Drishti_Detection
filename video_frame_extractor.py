import cv2
import os
import subprocess
import tempfile
from datetime import datetime

# Configuration variables - Edit these as needed
VIDEO_PATH = r"/media/viraj/New Volume/Dhrishti/Day 6/ace_delhi_6_10m20260124_050136/ace_delhi_6_10m20260124_050136.avi"  # Path to your input video file
OUTPUT_DIR = r"/media/viraj/New Volume/Dhrishti/Day 6/ace_delhi_6_10m20260124_050136/True_frame_count"                  # Base directory for saving extracted frames
IMAGE_FORMAT = "png"                                                            # Output image format: jpg, png, bmp
JPEG_QUALITY = 100                                                               # JPEG quality (1-100), only used if IMAGE_FORMAT is jpg
TARGET_CODEC = "mpeg4"                                                          # Target codec for preprocessing (deterministic, codec-agnostic)
FORCE_MPEG4 = True                                                               # If True, transcode input to MPEG-4 before reading


def probe_video_codec(video_path):
    """Return the codec name and profile for the primary video stream."""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,profile,codec_long_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return None

        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if len(lines) < 3:
            return None

        return {
            "codec_name": lines[0],
            "codec_long_name": lines[1],
            "profile": lines[2],
        }
    except FileNotFoundError:
        return None


def transcode_to_mpeg4(video_path):
    """Transcode the video to MPEG-4 codec using ffmpeg.

    This ensures deterministic, codec-agnostic frame extraction.
    MPEG-4 is a standard codec that works consistently across all systems.
    """
    try:
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".avi")
        tmp_file.close()
        output_path = tmp_file.name

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            video_path,
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-vtag",
            "xvid",
            "-an",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠ Warning: ffmpeg MPEG-4 transcode failed, falling back to original video file.")
            print(result.stderr.strip())
            return video_path

        return output_path
    except FileNotFoundError:
        print("⚠ Warning: ffmpeg is not installed or not found in PATH. Skipping MPEG-4 preprocessing.")
        return video_path


def ensure_mpeg4_input(video_path):
    """Return a video path that is encoded as MPEG-4 for deterministic frame extraction."""
    if not FORCE_MPEG4:
        return video_path

    info = probe_video_codec(video_path)
    if info is None:
        print("⚠ Could not probe video codec. Proceeding with original file.")
        return video_path

    codec = info.get("codec_name", "")
    profile = info.get("profile", "")
    print(f"Input codec: {codec}, profile: {profile}")

    if codec.lower() == TARGET_CODEC and profile.lower() != "simple profile":
        print("✓ Input already uses MPEG-4. Using original file.")
        return video_path

    print("🔄 Transcoding input to MPEG-4 for deterministic frame extraction...")
    return transcode_to_mpeg4(video_path)


def extract_frames(video_path, output_dir, image_format="jpg", jpeg_quality=100):
    """
    Extract all frames from a video file and save them to the output directory.

    Uses FFmpeg preprocessing to transcode to MPEG-4 for deterministic, codec-agnostic extraction.

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

    print("\n" + "=" * 70)
    print("VIDEO FRAME EXTRACTOR (Deterministic MPEG-4 Decoding)")
    print("=" * 70)

    # Prepare the input file: transcode to MPEG-4 if needed
    processed_video_path = ensure_mpeg4_input(video_path)
    transcoded_video_path = None
    if processed_video_path != video_path:
        transcoded_video_path = processed_video_path

    # Open video capture
    cap = cv2.VideoCapture(processed_video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {processed_video_path}")
        return

    # Get video properties
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    metadata_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if video_fps <= 0:
        print("⚠ Warning: Could not detect video FPS, defaulting to 30")
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

    print("\n" + "-" * 70)
    print("Extracting frames...")
    print("-" * 70)

    frame_count = 0
    saved_count = 0

    while True:
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
    cap.release()
    if transcoded_video_path is not None and os.path.exists(transcoded_video_path):
        try:
            os.remove(transcoded_video_path)
        except Exception:
            pass

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"\nSummary:")
    print(f"  Metadata claimed: {metadata_frame_count}")
    print(f"  Actually decoded: {saved_count}")
    if saved_count != metadata_frame_count:
        discrepancy = saved_count / metadata_frame_count if metadata_frame_count > 0 else 0
        print(f"  Discrepancy: {discrepancy:.2f}x (codec/metadata issue)")
    print(f"  Output directory: {frames_dir}")
    print(f"  Frames per second in source: {video_fps:.2f}")
    print("=" * 70 + "\n")

    return frames_dir


def main():
    """Main function to run the frame extraction."""
    extract_frames(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        image_format=IMAGE_FORMAT,
        jpeg_quality=JPEG_QUALITY,
    )


if __name__ == "__main__":
    main()
