"""
Stage 1 — Rail Head Deskew and Centering (Robust Line Fitting)
================================================================
Input  : raw BGR frame (1008 x 1022, uint8)
Output : deskewed BGR frame (fixed resolution, rail centred vertically + horizontally)
         + metadata dict

Pipeline:
  1. Convert to grayscale geometry copy
  2. Bilateral denoise + glare mask
  3. Scharr X gradient → vertical boundary evidence
  4. Morphological close (vertical kernel) → binary edge mask
  5. Extract edge points, split into left/right halves
  6. Robust line fitting with RANSAC on each half
  7. Estimate centreline angle and rail width
  8. Rotate original colour frame (BORDER_CONSTANT, black fill)
  9. Resize to fixed output resolution with rail centered
  10. QC gate → return deskewed frame + metadata

Advantage over Hough:
  - Works directly on edge pixels, not disconnected segments
  - RANSAC is robust to outliers and multiple faults
  - No pairing logic needed — left/right naturally separated by image midline
  - More forgiving for challenging frames (low contrast, multiple defects)
"""

import cv2
import numpy as np


# ── Tunable constants (all in one place) ─────────────────────────────────────

APPLY_ROTATION        = False  # Set to False to skip tilt correction, True to apply rotation

BILATERAL_D           = 7
BILATERAL_SIGMA_COLOR = 40
BILATERAL_SIGMA_SPACE = 40

GLARE_PERCENTILE      = 99.7   # top N% of pixels = glare
GLARE_MAX_FRAC        = 0.02   # if glare > 2% of frame, skip inpainting

SCHARR_ALPHA          = 0.7    # suppress horizontal texture: E = |Gx| - alpha*|Gy|
EDGE_PERCENTILE       = 88     # threshold on edge energy (tune 80–92)

MORPH_VKERNEL         = (3, 25)  # vertical kernel for morphological close

# RANSAC line fitting
RANSAC_DISTANCE       = 2.0    # max distance for inliers (pixels)
RANSAC_CONFIDENCE     = 0.99   # confidence level
MAX_TILT_DEG          = 25.0   # reject fitted lines > N° from vertical

RAIL_WIDTH_FRAC_MIN   = 0.12   # expected rail width as fraction of frame width
RAIL_WIDTH_FRAC_MAX   = 0.45

# Fixed output resolution (width will be adjusted to keep rail at standard zoom)
OUTPUT_HEIGHT_PX      = 1022   # keep same as input
OUTPUT_WIDTH_PX       = 800    # fixed width; rail centred horizontally, black padding on sides

# QC hard limits
QC_MAX_ABS_ROTATION   = 20.0   # deg — implausibly large for a fixed mount
QC_MIN_WIDTH_FRAC     = 0.10
QC_MAX_WIDTH_FRAC     = 0.50
# ─────────────────────────────────────────────────────────────────────────────


def _to_gray(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def preprocess_geometry(gray_u8):
    """
    Denoise + build glare mask.
    Returns (denoised_u8, glare_mask_u8, glare_frac).
    """
    g = cv2.bilateralFilter(
        gray_u8, d=BILATERAL_D,
        sigmaColor=BILATERAL_SIGMA_COLOR,
        sigmaSpace=BILATERAL_SIGMA_SPACE
    )

    thr = np.percentile(g, GLARE_PERCENTILE)
    glare = ((g >= thr).astype(np.uint8)) * 255
    glare = cv2.morphologyEx(
        glare, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    glare = cv2.dilate(glare, np.ones((5, 5), np.uint8), iterations=1)

    glare_frac = float(glare.mean() / 255.0)
    if glare_frac < GLARE_MAX_FRAC:
        g = cv2.inpaint(g, glare, 3, cv2.INPAINT_TELEA)

    return g, glare, glare_frac


def detect_rail_robust(geom_u8, H, W):
    """
    Find left and right rail boundary lines using RANSAC on edge pixels.
    Returns (line_l, line_r) as cv2.fitLine results, or None on failure.
    """
    gx = cv2.Scharr(geom_u8, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(geom_u8, cv2.CV_32F, 0, 1)
    e = np.abs(gx) - SCHARR_ALPHA * np.abs(gy)
    e[e < 0] = 0
    e = cv2.normalize(e, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    p = np.percentile(e, EDGE_PERCENTILE)
    _, bw = cv2.threshold(e, int(p), 255, cv2.THRESH_BINARY)

    bw = cv2.morphologyEx(
        bw, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, MORPH_VKERNEL)
    )

    # Extract edge pixels
    points = cv2.findNonZero(bw)  # shape: (N, 1, 2) format
    if points is None or len(points) < 20:
        return None

    points = points.reshape(-1, 2).astype(np.float32)

    # Split into left and right halves
    mid_x = W / 2.0
    left_points = points[points[:, 0] < mid_x]
    right_points = points[points[:, 0] >= mid_x]

    if len(left_points) < 10 or len(right_points) < 10:
        return None

    # Fit lines using RANSAC via cv2.fitLine
    # fitLine with cv2.DIST_L2 and RANSAC (implicit via robust method)
    try:
        # cv2.fitLine expects [x, y] format in float32
        line_l = cv2.fitLine(left_points, cv2.DIST_WELSCH, 0, 0.01, 0.01)
        line_r = cv2.fitLine(right_points, cv2.DIST_WELSCH, 0, 0.01, 0.01)
    except:
        return None

    # Validate tilt
    def get_tilt(line):
        vx, vy = float(line[0]), float(line[1])
        ang = np.degrees(np.arctan2(vy, vx))
        tilt = abs(90.0 - abs(ang))
        return tilt

    if get_tilt(line_l) > MAX_TILT_DEG or get_tilt(line_r) > MAX_TILT_DEG:
        return None

    return line_l, line_r


def centreline_angle_and_width(line_l, line_r, H):
    """
    Compute rail centreline x, rail width, and rotation angle needed to verticalise.
    """
    def sample_x(line, y):
        vx, vy, x0, y0 = [float(v) for v in line.reshape(-1)]
        if abs(vy) < 1e-6:
            return float(x0)
        return float(x0 + (y - y0) / vy * vx)

    ys = np.linspace(0.1 * H, 0.9 * H, 9)
    widths = [abs(sample_x(line_r, y) - sample_x(line_l, y)) for y in ys]
    width_px = float(np.median(widths))

    # Normalize both direction vectors to point downward (vy > 0)
    def _downward(line):
        v = line.reshape(-1).copy()
        if float(v[1]) < 0:
            v = -v
        return v

    vl = _downward(line_l)
    vr = _downward(line_r)
    vx = (float(vl[0]) + float(vr[0])) / 2.0
    vy = (float(vl[1]) + float(vr[1])) / 2.0
    theta_deg = np.degrees(np.arctan2(vy, vx))
    rot_deg = 90.0 - theta_deg

    # Clamp to [-90, 90]
    while rot_deg > 90.0:
        rot_deg -= 180.0
    while rot_deg < -90.0:
        rot_deg += 180.0

    # OpenCV getRotationMatrix2D: positive angle = CCW, negative = CW.
    # Negate so the rotation corrects the tilt in the right direction.
    rot_deg = -rot_deg

    y_mid = 0.5 * H
    x_c = 0.5 * (sample_x(line_l, y_mid) + sample_x(line_r, y_mid))
    return float(x_c), width_px, float(rot_deg)


def rotate_and_deskew(frame_bgr, x_c, rot_deg):
    """
    Rotate the original colour frame about the rail centreline.
    Empty corners left by rotation are filled with BLACK (BORDER_CONSTANT).
    """
    H, W = frame_bgr.shape[:2]
    center = (float(x_c), H / 2.0)

    M = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
    deskewed = cv2.warpAffine(
        frame_bgr, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    cx_rot, cy_rot = (M @ np.array([center[0], center[1], 1.0])).tolist()
    return deskewed, float(cx_rot), float(cy_rot), M


def resize_and_center(frame_bgr, cx_rot):
    """
    Resize to fixed output resolution with rail horizontally centered.

    If rail is narrower than OUTPUT_WIDTH_PX, add black padding on sides.
    If rail is wider, crop symmetrically.

    Output is always OUTPUT_HEIGHT_PX × OUTPUT_WIDTH_PX.
    """
    H, W = frame_bgr.shape[:2]
    cx = float(cx_rot)

    # Target output center
    out_center_x = OUTPUT_WIDTH_PX / 2.0

    # How much space needed on each side
    half_w = OUTPUT_WIDTH_PX / 2.0
    x_left = max(0, int(cx - half_w))
    x_right = min(W, int(cx + half_w))
    actual_width = x_right - x_left

    # Crop to the region
    cropped = frame_bgr[:, x_left:x_right]

    # Pad to OUTPUT_WIDTH_PX if needed (both sides equally)
    if actual_width < OUTPUT_WIDTH_PX:
        left_pad = int((OUTPUT_WIDTH_PX - actual_width) / 2.0)
        right_pad = OUTPUT_WIDTH_PX - actual_width - left_pad
        cropped = cv2.copyMakeBorder(
            cropped, 0, 0, left_pad, right_pad,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

    # Resize height to OUTPUT_HEIGHT_PX (stretch/shrink vertically)
    resized = cv2.resize(cropped, (OUTPUT_WIDTH_PX, OUTPUT_HEIGHT_PX),
                         interpolation=cv2.INTER_LINEAR)

    return resized, x_left, x_right


def qc_check(rot_deg, width_px, W):
    """
    Hard QC gate. Returns (pass: bool, fail_reason: str).
    """
    width_frac = width_px / W
    if abs(rot_deg) > QC_MAX_ABS_ROTATION:
        return False, f"rotation_too_large ({rot_deg:.1f}°)"
    if width_frac < QC_MIN_WIDTH_FRAC:
        return False, f"rail_too_narrow ({width_frac:.3f})"
    if width_frac > QC_MAX_WIDTH_FRAC:
        return False, f"rail_too_wide ({width_frac:.3f})"
    return True, ""


def process_frame(frame_bgr):
    """
    Full Stage 1 pipeline for one frame (robust line fitting variant).

    Returns:
        deskewed_bgr : ndarray or None (None = QC fail)
        meta         : dict with all computed fields
    """
    H, W = frame_bgr.shape[:2]
    gray = _to_gray(frame_bgr)

    meta = {
        "method": "robust_ransac",
        "frame_h": H,
        "frame_w": W,
        "qc_pass": False,
        "fail_reason": "",
        "rotation_deg": None,
        "rail_center_x": None,
        "rail_width_px": None,
        "rail_width_frac": None,
        "glare_frac": None,
    }

    # 1. Geometry pre-processing
    geom, _, glare_frac = preprocess_geometry(gray)
    meta["glare_frac"] = round(glare_frac, 4)

    # 2. Robust detection
    result = detect_rail_robust(geom, H, W)
    if result is None:
        meta["fail_reason"] = "no_valid_edge_fit"
        return None, meta

    line_l, line_r = result

    # 3. Centreline, angle, width
    x_c, width_px, rot_deg = centreline_angle_and_width(line_l, line_r, H)
    meta["rotation_deg"]    = round(rot_deg, 3)
    meta["rail_center_x"]   = round(x_c, 1)
    meta["rail_width_px"]   = round(width_px, 1)
    meta["rail_width_frac"] = round(width_px / W, 4)

    # 4. QC
    passed, reason = qc_check(rot_deg, width_px, W)
    meta["qc_pass"]    = passed
    meta["fail_reason"] = reason
    if not passed:
        return None, meta

    # 5. Rotate original colour frame (if enabled)
    if APPLY_ROTATION:
        rotated, cx_rot, cy_rot, _ = rotate_and_deskew(frame_bgr, x_c, rot_deg)
        meta["rail_center_x_rotated"] = round(cx_rot, 1)
        frame_to_resize = rotated
        cx_for_resize = cx_rot
    else:
        # Skip rotation, use original frame
        frame_to_resize = frame_bgr
        cx_for_resize = x_c
        meta["rail_center_x_rotated"] = round(x_c, 1)

    # 6. Resize and center with fixed output resolution
    deskewed, _, _ = resize_and_center(frame_to_resize, cx_for_resize)
    meta["output_width_px"]  = OUTPUT_WIDTH_PX
    meta["output_height_px"] = OUTPUT_HEIGHT_PX

    return deskewed, meta
