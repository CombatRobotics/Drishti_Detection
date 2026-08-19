#!/usr/bin/env python3
"""
Visualize ORB features and binary descriptors.

Shows:
1. Detected keypoints on the image
2. Keypoint positions and orientations
3. Binary descriptor visualization
4. Features nearest to image center
5. Feature statistics

========== HOW TO RUN ==========

1. Basic visualization:
   python3 visualize_orb_features.py /path/to/image.jpg

2. With custom number of features:
   python3 visualize_orb_features.py /path/to/image.jpg --n-features 500

3. Show top N features closest to center:
   python3 visualize_orb_features.py /path/to/image.jpg --top-n 10

4. Save visualization:
   python3 visualize_orb_features.py /path/to/image.jpg --output visualization.jpg

5. With region mask:
   python3 visualize_orb_features.py /path/to/image.jpg --mask /path/to/mask.png --top-n 5
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import json

# ==========================================================================
# CONFIGURATION
# ==========================================================================
N_FEATURES = 750 
MASK_PATH = r"/media/viraj/New Volume/Dhrishti/Detection_code/Mask.png"  # Set to mask image path if using region restriction


def detect_orb_features(image, mask=None, n_features=N_FEATURES):
    """Detect ORB keypoints and descriptors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if mask is not None:
        if mask.shape != gray.shape:
            mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]))
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]

    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, mask)

    return keypoints, descriptors, gray


def get_keypoint_distances_from_center(keypoints, image_h, image_w):
    """Calculate distance of each keypoint from image center."""
    center_x, center_y = image_w / 2, image_h / 2

    distances = []
    for kp in keypoints:
        x, y = kp.pt
        dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        distances.append({
            "keypoint": kp,
            "x": x,
            "y": y,
            "distance": dist
        })

    # Sort by distance
    distances.sort(key=lambda d: d["distance"])
    return distances


def draw_keypoints_with_orientation(image, keypoints):
    """Draw keypoints with orientation arrows."""
    img_with_kp = image.copy()

    for kp in keypoints:
        # Draw circle at keypoint
        pt = (int(kp.pt[0]), int(kp.pt[1]))
        cv2.circle(img_with_kp, pt, 5, (0, 255, 0), 2)

        # Draw orientation arrow
        angle = np.radians(kp.angle)
        arrow_len = 15
        arrow_end = (
            int(kp.pt[0] + arrow_len * np.cos(angle)),
            int(kp.pt[1] + arrow_len * np.sin(angle))
        )
        cv2.arrowedLine(img_with_kp, pt, arrow_end, (255, 0, 0), 2)

    return img_with_kp


def draw_center_point(image):
    """Draw image center point."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    cv2.circle(image, center, 10, (0, 0, 255), 3)
    cv2.putText(image, "CENTER", (center[0] + 15, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image


def visualize_descriptor_bits(descriptor, kp_index=0):
    """Visualize a single descriptor as binary bits."""
    if descriptor is None or len(descriptor) == 0:
        return None

    # Get a single descriptor (if multiple, take first)
    desc = descriptor[kp_index] if len(descriptor.shape) > 1 else descriptor

    # Convert to binary string
    binary_str = ''.join(format(byte, '08b') for byte in desc)

    # Create visualization (16x16 grid of bits)
    bits = [int(b) for b in binary_str]
    grid_size = min(16, len(bits))
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)

    for i, bit in enumerate(bits[:grid_size**2]):
        row = i // grid_size
        col = i % grid_size
        grid[row, col] = 255 if bit == 1 else 0

    # Scale up for visibility
    img_desc = cv2.resize(grid, (256, 256), interpolation=cv2.INTER_NEAREST)
    return img_desc, binary_str


def create_center_distance_viz(image, keypoints_info):
    """Create visualization showing features by distance from center."""
    h, w = image.shape[:2]
    center_x, center_y = w / 2, h / 2

    # Create image with center marked
    viz = image.copy()
    cv2.circle(viz, (int(center_x), int(center_y)), 10, (0, 0, 255), 3)

    # Draw distance circles and keypoints
    for i, kp_info in enumerate(keypoints_info[:10]):  # Top 10
        kp = kp_info["keypoint"]
        dist = kp_info["distance"]
        x, y = int(kp.pt[0]), int(kp.pt[1])

        # Color by rank (closer = greener)
        color = (0, 255 - i * 25, i * 25)

        # Draw keypoint
        cv2.circle(viz, (x, y), 6, color, 2)

        # Draw line to center
        cv2.line(viz, (int(center_x), int(center_y)), (x, y), color, 1)

        # Add distance label
        cv2.putText(viz, f"{i+1} ({dist:.0f}px)", (x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return viz


def visualize_orb_features(image_path, output_path=None, mask_path=None,
                          n_features=N_FEATURES, top_n=10):
    """Main visualization function."""

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        return

    h, w = image.shape[:2]
    print(f"\n📸 Image: {Path(image_path).name}")
    print(f"   Size: {w}x{h}")

    # Load mask if provided
    mask = None
    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️  Could not load mask: {mask_path}")
        else:
            print(f"✓ Mask loaded: {mask_path}")

    # Detect ORB features
    print(f"\n🔍 Detecting ORB features (n_features={n_features})...")
    keypoints, descriptors, gray = detect_orb_features(image, mask, n_features)

    print(f"✓ Found {len(keypoints)} keypoints")
    print(f"✓ Descriptors shape: {descriptors.shape if descriptors is not None else 'None'}")

    # Get keypoints sorted by distance from center
    print(f"\n📍 Calculating distances from center...")
    kp_distances = get_keypoint_distances_from_center(keypoints, h, w)

    print(f"\n🎯 Top {top_n} Features Closest to Center:")
    print("-" * 70)
    for i, kp_info in enumerate(kp_distances[:top_n]):
        kp = kp_info["keypoint"]
        print(f"  {i+1}. Position: ({kp_info['x']:.1f}, {kp_info['y']:.1f}) | "
              f"Distance: {kp_info['distance']:.1f}px | "
              f"Scale: {kp.size:.2f} | Angle: {kp.angle:.1f}°")

    # Create visualizations
    print(f"\n📊 Creating visualizations...")

    # 1. Keypoints with orientation
    viz_kp = draw_keypoints_with_orientation(image.copy(), keypoints)
    viz_kp = draw_center_point(viz_kp)

    # 2. Distance visualization
    viz_dist = create_center_distance_viz(image.copy(), kp_distances)

    # 3. Descriptor visualization (first keypoint)
    if descriptors is not None and len(keypoints) > 0:
        desc_viz, binary_str = visualize_descriptor_bits(descriptors, 0)
        print(f"\n📝 First Keypoint Descriptor:")
        print(f"   Binary ({len(binary_str)} bits): {binary_str[:64]}...")
        print(f"   (showing first 64 bits)")

    # Combine visualizations
    print(f"\n💾 Saving visualizations...")

    if output_path:
        # Save individual visualizations
        base_path = Path(output_path).stem
        output_dir = Path(output_path).parent

        kp_output = output_dir / f"{base_path}_keypoints.jpg"
        cv2.imwrite(str(kp_output), viz_kp)
        print(f"✓ Keypoints: {kp_output}")

        dist_output = output_dir / f"{base_path}_distance.jpg"
        cv2.imwrite(str(dist_output), viz_dist)
        print(f"✓ Distance visualization: {dist_output}")

        if descriptors is not None and len(keypoints) > 0:
            desc_output = output_dir / f"{base_path}_descriptor.jpg"
            cv2.imwrite(str(desc_output), desc_viz)
            print(f"✓ Descriptor visualization: {desc_output}")

        # Save statistics
        stats_output = output_dir / f"{base_path}_stats.json"
        stats = {
            "image_path": str(image_path),
            "image_size": {"width": w, "height": h},
            "n_keypoints": len(keypoints),
            "n_features_requested": n_features,
            "top_features_by_distance": [
                {
                    "rank": i + 1,
                    "x": kp_info["x"],
                    "y": kp_info["y"],
                    "distance_from_center": kp_info["distance"],
                    "scale": float(kp_info["keypoint"].size),
                    "angle_degrees": float(kp_info["keypoint"].angle)
                }
                for i, kp_info in enumerate(kp_distances[:top_n])
            ]
        }
        with open(stats_output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Statistics: {stats_output}")

    else:
        # Show visualizations
        print("\n📺 Displaying visualizations (press any key to close)...")
        cv2.imshow("ORB Keypoints with Orientation", viz_kp)
        cv2.imshow("Features by Distance from Center", viz_dist)
        if descriptors is not None and len(keypoints) > 0:
            cv2.imshow("Binary Descriptor Visualization", desc_viz)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print("✓ ORB Visualization Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize ORB features and binary descriptors",
        epilog="""
Examples:
  python3 visualize_orb_features.py /path/to/image.jpg
  python3 visualize_orb_features.py /path/to/image.jpg --output viz.jpg
  python3 visualize_orb_features.py /path/to/image.jpg --top-n 20
  python3 visualize_orb_features.py /path/to/image.jpg --mask /path/to/mask.png

Output files created:
  - {name}_keypoints.jpg: Detected keypoints with orientation arrows
  - {name}_distance.jpg: Features colored by distance from center
  - {name}_descriptor.jpg: Binary descriptor visualization (16x16 bit grid)
  - {name}_stats.json: Feature statistics
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", default=None, help="Output path for visualizations")
    parser.add_argument("--mask", default=MASK_PATH, help="Optional region mask image")
    parser.add_argument("--n-features", type=int, default=N_FEATURES,
                        help=f"Number of features to detect (default: {N_FEATURES})")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Show top N features closest to center (default: 10)")

    args = parser.parse_args()

    visualize_orb_features(
        args.image_path,
        output_path=args.output,
        mask_path=args.mask,
        n_features=args.n_features,
        top_n=args.top_n
    )
