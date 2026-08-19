"""
Visualize Stage 1 Robust pipeline — 12-stage debug output for one image.

Usage:
  python3 visualize_stage1_robust.py --image /path/to/frame.jpg
"""

import cv2
import numpy as np
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from stage1_deskew_robust import (
    _to_gray, preprocess_geometry, SCHARR_ALPHA, EDGE_PERCENTILE,
    MORPH_VKERNEL, detect_rail_robust, centreline_angle_and_width,
    rotate_and_deskew, resize_and_center, qc_check, process_frame,
    OUTPUT_HEIGHT_PX, OUTPUT_WIDTH_PX
)

OUTPUT_BASE = os.path.join(os.path.dirname(__file__), "output_robust")


def _draw_text(img, text, pos=(10, 30), scale=0.6, color=(0, 255, 0)):
    """Draw text on image."""
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)


def _draw_line_on_image(img, line, color=(0, 255, 0), thickness=2):
    """Draw a fitted line (from cv2.fitLine) on image."""
    if line is None:
        return
    vx, vy, x0, y0 = [float(v) for v in line.reshape(-1)]
    H, W = img.shape[:2]

    if abs(vy) > 1e-6:
        y1, y2 = 0, H
        x1 = x0 + (y1 - y0) / vy * vx
        x2 = x0 + (y2 - y0) / vy * vx
        x1, x2 = int(np.clip(x1, 0, W-1)), int(np.clip(x2, 0, W-1))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    else:
        x = int(np.clip(x0, 0, W-1))
        cv2.line(img, (x, 0), (x, H), color, thickness)


def visualize_pipeline(image_path):
    """Run debug pipeline and save 12-stage visualization."""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Cannot read image: {image_path}")
        sys.exit(1)

    H, W = frame.shape[:2]
    fname = os.path.basename(image_path)

    # Create output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = os.path.join(OUTPUT_BASE, f"debug_robust_{ts}")
    os.makedirs(debug_dir, exist_ok=True)

    print(f"\nUsing image: {image_path}\n")
    print(f"Running debug pipeline → {debug_dir}\n")

    stages = {}

    # Stage 1: Original color
    stages[1] = ("01_original_color.jpg", frame.copy())

    # Stage 2: Grayscale
    gray = _to_gray(frame)
    stages[2] = ("02_grayscale.jpg", cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

    # Stage 3: Bilateral denoised
    geom, glare_mask, glare_frac = preprocess_geometry(gray)
    stages[3] = ("03_bilateral_denoised.jpg", cv2.cvtColor(geom, cv2.COLOR_GRAY2BGR))

    # Stage 4: Glare mask
    stages[4] = ("04_glare_mask.jpg", cv2.cvtColor(glare_mask, cv2.COLOR_GRAY2BGR))

    # Stage 5: After inpaint
    stages[5] = ("05_after_inpaint.jpg", cv2.cvtColor(geom, cv2.COLOR_GRAY2BGR))

    # Stage 6: Scharr X energy
    gx = cv2.Scharr(geom, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(geom, cv2.CV_32F, 0, 1)
    e = np.abs(gx) - SCHARR_ALPHA * np.abs(gy)
    e[e < 0] = 0
    e = cv2.normalize(e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stages[6] = ("06_scharr_x_energy.jpg", cv2.cvtColor(e, cv2.COLOR_GRAY2BGR))

    # Stage 7: Binary edges
    p = np.percentile(e, EDGE_PERCENTILE)
    _, bw = cv2.threshold(e, int(p), 255, cv2.THRESH_BINARY)
    stages[7] = ("07_binary_edges.jpg", cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR))

    # Stage 8: Morphological close
    bw_morph = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_VKERNEL)
    )
    stages[8] = ("08_morph_close.jpg", cv2.cvtColor(bw_morph, cv2.COLOR_GRAY2BGR))

    # Stage 9: Edge points (split left/right)
    points = cv2.findNonZero(bw_morph)
    if points is not None:
        points = points.reshape(-1, 2).astype(np.float32)
        mid_x = W / 2.0
        left_points = points[points[:, 0] < mid_x]
        right_points = points[points[:, 0] >= mid_x]

        stage9_img = stages[8][1].copy()
        if len(left_points) > 0:
            for pt in left_points[::2]:
                cv2.circle(stage9_img, (int(pt[0]), int(pt[1])), 1, (255, 0, 0), -1)
        if len(right_points) > 0:
            for pt in right_points[::2]:
                cv2.circle(stage9_img, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)
        stages[9] = ("09_edge_points.jpg", stage9_img)
    else:
        stages[9] = ("09_edge_points.jpg", stages[8][1].copy())

    # Stage 10: Fitted lines
    stage10_img = stages[9][1].copy()
    result = detect_rail_robust(geom, H, W)
    if result is not None:
        line_l, line_r = result
        _draw_line_on_image(stage10_img, line_l, color=(255, 0, 0), thickness=2)  # Blue
        _draw_line_on_image(stage10_img, line_r, color=(0, 255, 0), thickness=2)  # Green
        _draw_text(stage10_img, "DETECTION OK", pos=(10, 30), color=(0, 255, 0))
    else:
        _draw_text(stage10_img, "DETECTION FAILED", pos=(10, 30), color=(0, 0, 255))
    stages[10] = ("10_fitted_lines.jpg", stage10_img)

    # Stage 11: Deskewed and resized
    deskewed, meta = process_frame(frame)
    if deskewed is not None:
        stage11_img = deskewed.copy()
        _draw_text(stage11_img, f"PASS (rot={meta['rotation_deg']:.1f}°)",
                   pos=(10, 30), color=(0, 255, 0))
        stages[11] = ("11_deskewed_final.jpg", stage11_img)
    else:
        stage11_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        _draw_text(stage11_img, f"FAILED: {meta['fail_reason']}",
                   pos=(10, 30), color=(0, 0, 255))
        stages[11] = ("11_deskewed_final.jpg", stage11_img)

    # Stage 12: Metadata
    stage12_img = np.ones((600, 800, 3), dtype=np.uint8) * 240
    y = 30
    for key, val in meta.items():
        text = f"{key}: {val}"
        _draw_text(stage12_img, text, pos=(20, y), scale=0.5, color=(0, 0, 0))
        y += 25
    stages[12] = ("12_metadata.jpg", stage12_img)

    # Save individual stages
    for idx in sorted(stages.keys()):
        fname_out, img = stages[idx]
        fpath = os.path.join(debug_dir, fname_out)
        cv2.imwrite(fpath, img)
        print(f"  saved: {fname_out}")

    # Create composite grid (4 rows × 3 cols = 12 stages)
    # Each cell: 400×300 (3:4 aspect ratio)
    cell_w, cell_h = 400, 300
    composite = np.ones((4 * cell_h, 3 * cell_w, 3), dtype=np.uint8) * 200

    for idx in sorted(stages.keys()):
        row = (idx - 1) // 3
        col = (idx - 1) % 3
        y_start = row * cell_h
        x_start = col * cell_w

        _, img = stages[idx]
        img_resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
        composite[y_start:y_start+cell_h, x_start:x_start+cell_w] = img_resized

    comp_path = os.path.join(debug_dir, "composite.jpg")
    cv2.imwrite(comp_path, composite)
    comp_h, comp_w = composite.shape[:2]
    print(f"\n  composite saved: {comp_path}")
    print(f"  composite size : {comp_w}x{comp_h} px\n")

    # Print metadata summary
    print("Metadata:")
    for k, v in meta.items():
        print(f"  {k:<30} {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Stage 1 Robust pipeline")
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()
    visualize_pipeline(args.image)
