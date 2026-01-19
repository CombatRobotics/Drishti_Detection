#!/usr/bin/env python3
import cv2
import numpy as np
import math

# ================= USER CONFIG ================= #
IMAGE_PATH = "D:\\Dhrishti\\rosbags\\pun1_5km_300_0_20260106_025905_20260106_145734\\frame_00530.jpg"
# =============================================== #

# ---------- Utils ----------
def vertical_destripe(gray, strength, mask=None, smooth_sigma=0.0):
    if strength <= 0:
        return gray
    gray_f = gray.astype(np.float32)
    if mask is None:
        mask_f = np.ones_like(gray_f, dtype=np.float32)
    else:
        mask_f = (mask > 0).astype(np.float32)

    col_sum = (gray_f * mask_f).sum(axis=0, keepdims=True)
    col_count = mask_f.sum(axis=0, keepdims=True)
    col_mean = col_sum / np.maximum(col_count, 1.0)

    total_count = mask_f.sum()
    if total_count <= 0:
        global_mean = gray_f.mean()
    else:
        global_mean = (gray_f * mask_f).sum() / total_count

    correction = (global_mean - col_mean) * strength
    if smooth_sigma > 0:
        correction = cv2.GaussianBlur(correction, (0, 0), smooth_sigma, 0)
    out = gray_f + correction
    return np.clip(out, 0, 255).astype(np.uint8)

def flat_field(gray, illum_sigma, p_low, p_high, mask_for_stats=None):
    gray_f = gray.astype(np.float32)
    if illum_sigma > 0:
        illum = cv2.GaussianBlur(gray_f, (0, 0), illum_sigma)
    else:
        illum = gray_f.copy()
    flat = gray_f / (illum + 1e-6)

    if mask_for_stats is None:
        stats_vals = flat.reshape(-1)
    else:
        stats_vals = flat[mask_for_stats > 0]
        if stats_vals.size == 0:
            stats_vals = flat.reshape(-1)

    low = np.percentile(stats_vals, p_low)
    high = np.percentile(stats_vals, p_high)
    if high - low < 1e-6:
        low, high = stats_vals.min(), stats_vals.max()
        if high - low < 1e-6:
            return np.zeros_like(gray, dtype=np.uint8)

    out = (flat - low) * (255.0 / (high - low))
    return np.clip(out, 0, 255).astype(np.uint8)

def unsharp_mask(gray, radius, amount, threshold):
    if radius <= 0 or amount <= 0:
        return gray
    gray_f = gray.astype(np.float32)
    blur = cv2.GaussianBlur(gray_f, (0, 0), radius)
    detail = gray_f - blur
    if threshold > 0:
        detail[np.abs(detail) < threshold] = 0
    sharp = gray_f + amount * detail
    return np.clip(sharp, 0, 255).astype(np.uint8)

# ---------- Load image ----------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(IMAGE_PATH)

if img.ndim == 3:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
else:
    gray = img.copy()

H, W = gray.shape[:2]

cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controls", 420, H)
cv2.resizeWindow("Preview", W, H)
cv2.moveWindow("Controls", 20, 20)
cv2.moveWindow("Preview", 460, 20)

# ---------- Trackbars ----------
cv2.createTrackbar("LEFT_X", "Controls", int(W * 0.3), W - 2, lambda x: None)
cv2.createTrackbar("LEFT_ANGLE", "Controls", 180, 360, lambda x: None)  # -180 .. +180
cv2.createTrackbar("RIGHT_X", "Controls", int(W * 0.6), W - 1, lambda x: None)

cv2.createTrackbar("Destripe x100", "Controls", 40, 100, lambda x: None)
cv2.createTrackbar("StripeSmooth", "Controls", 7, 50, lambda x: None)

cv2.createTrackbar("IllumSigma", "Controls", 80, 200, lambda x: None)
cv2.createTrackbar("ROI pLow", "Controls", 1, 20, lambda x: None)
cv2.createTrackbar("ROI pHigh", "Controls", 99, 100, lambda x: None)

cv2.createTrackbar("ROI NLM h", "Controls", 5, 30, lambda x: None)

cv2.createTrackbar("BG CLAHE Clip x10", "Controls", 20, 100, lambda x: None)
cv2.createTrackbar("BG CLAHE Tile", "Controls", 8, 32, lambda x: None)

cv2.createTrackbar("SharpAmt x100", "Controls", 160, 300, lambda x: None)
cv2.createTrackbar("SharpRad x10", "Controls", 12, 50, lambda x: None)
cv2.createTrackbar("SharpThr", "Controls", 5, 50, lambda x: None)

cv2.createTrackbar("BG Blur x10", "Controls", 50, 200, lambda x: None)
cv2.createTrackbar("FeatherSigma", "Controls", 7, 50, lambda x: None)

# ---------- Main loop ----------
while True:
    left_x_base = cv2.getTrackbarPos("LEFT_X", "Controls")
    right_x = cv2.getTrackbarPos("RIGHT_X", "Controls")
    right_x = max(left_x_base + 10, right_x)

    angle_deg = cv2.getTrackbarPos("LEFT_ANGLE", "Controls") - 180
    angle_rad = math.radians(angle_deg)

    destripe_strength = cv2.getTrackbarPos("Destripe x100", "Controls") / 100.0
    stripe_smooth = cv2.getTrackbarPos("StripeSmooth", "Controls")

    illum_sigma = cv2.getTrackbarPos("IllumSigma", "Controls")
    p_low = cv2.getTrackbarPos("ROI pLow", "Controls")
    p_high = cv2.getTrackbarPos("ROI pHigh", "Controls")
    if p_high <= p_low + 1:
        p_high = min(100, p_low + 2)

    roi_nlm_h = cv2.getTrackbarPos("ROI NLM h", "Controls")

    bg_clahe_clip = cv2.getTrackbarPos("BG CLAHE Clip x10", "Controls") / 10.0
    bg_clahe_tile = max(2, cv2.getTrackbarPos("BG CLAHE Tile", "Controls"))

    sharp_amt = cv2.getTrackbarPos("SharpAmt x100", "Controls") / 100.0
    sharp_rad = cv2.getTrackbarPos("SharpRad x10", "Controls") / 10.0
    sharp_thr = cv2.getTrackbarPos("SharpThr", "Controls")

    bg_blur = cv2.getTrackbarPos("BG Blur x10", "Controls") / 10.0
    feather_sigma = cv2.getTrackbarPos("FeatherSigma", "Controls")

    # ---- Compute slanted LEFT boundary per row ----
    rows = np.arange(H)
    left_x_per_row = (
        left_x_base
        + (rows - H / 2) * math.tan(angle_rad)
    ).astype(np.int32)
    left_x_per_row = np.clip(left_x_per_row, 0, W - 2)

    # ---- Build ROI mask ----
    roi_mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        lx = left_x_per_row[y]
        roi_mask[y, lx:right_x] = 255

    # ---- Destripe before flat-field ----
    destriped = vertical_destripe(gray, destripe_strength, mask=cv2.bitwise_not(roi_mask), smooth_sigma=stripe_smooth)

    # ---- Illumination correction ----
    flat = flat_field(destriped, illum_sigma, p_low, p_high, mask_for_stats=roi_mask)

    # ---- ROI pipeline (keep detail) ----
    roi_proc = flat.copy()
    if roi_nlm_h > 0:
        roi_proc = cv2.fastNlMeansDenoising(roi_proc, None, roi_nlm_h, 7, 21)
    roi_proc = unsharp_mask(roi_proc, sharp_rad, sharp_amt, sharp_thr)

    # ---- Background pipeline (CLAHE then blur) ----
    bg_proc = flat.copy()
    if bg_clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=bg_clahe_clip, tileGridSize=(bg_clahe_tile, bg_clahe_tile))
        bg_proc = clahe.apply(bg_proc)
    if bg_blur > 0:
        bg_proc = cv2.GaussianBlur(bg_proc, (0, 0), bg_blur)

    # ---- Feathered blend ----
    alpha = roi_mask.astype(np.float32) / 255.0
    if feather_sigma > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather_sigma)
    out = roi_proc.astype(np.float32) * alpha + bg_proc.astype(np.float32) * (1.0 - alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)

    output = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Preview", output)

    key = cv2.waitKey(30)
    if key == 27 or key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("postprocess-bg-clahe.png", output)
        print("Saved postprocess-bg-clahe.png")

cv2.destroyAllWindows()
