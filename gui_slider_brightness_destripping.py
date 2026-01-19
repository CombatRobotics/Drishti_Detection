#!/usr/bin/env python3
import cv2
import numpy as np
import math

# ================= USER CONFIG ================= #
IMAGE_PATH = "D:\\Dhrishti\\rosbags\\pun1_6km_700_0_20260106_02355920260106_161121\\frame_00935.jpg"
# =============================================== #

# ---------- Utils ----------
def apply_gamma(img, gamma):
    gamma = max(gamma, 0.01)
    inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def vertical_destripe(img, strength):
    if strength <= 0:
        return img
    img_f = img.astype(np.float32)
    col_means = img_f.mean(axis=0, keepdims=True)
    global_mean = img_f.mean()
    correction = (global_mean - col_means) * strength
    out = img_f + correction
    return np.clip(out, 0, 255).astype(np.uint8)

# ---------- Load image ----------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(IMAGE_PATH)

H, W = img.shape[:2]

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

cv2.createTrackbar("Alpha x100", "Controls", 120, 300, lambda x: None)
cv2.createTrackbar("Gamma x100", "Controls", 100, 300, lambda x: None)

cv2.createTrackbar("NLM Strength", "Controls", 7, 30, lambda x: None)
cv2.createTrackbar("Destripe x100", "Controls", 40, 100, lambda x: None)

# ---------- Main loop ----------
while True:
    left_x_base = cv2.getTrackbarPos("LEFT_X", "Controls")
    right_x = cv2.getTrackbarPos("RIGHT_X", "Controls")
    right_x = max(left_x_base + 10, right_x)

    angle_deg = cv2.getTrackbarPos("LEFT_ANGLE", "Controls") - 180
    angle_rad = math.radians(angle_deg)

    alpha = cv2.getTrackbarPos("Alpha x100", "Controls") / 100.0
    gamma = cv2.getTrackbarPos("Gamma x100", "Controls") / 100.0

    nlm_h = cv2.getTrackbarPos("NLM Strength", "Controls")
    destripe_strength = cv2.getTrackbarPos("Destripe x100", "Controls") / 100.0

    # ---- Compute slanted LEFT boundary per row ----
    rows = np.arange(H)
    left_x_per_row = (
        left_x_base
        + (rows - H / 2) * math.tan(angle_rad)
    ).astype(np.int32)
    left_x_per_row = np.clip(left_x_per_row, 0, W - 2)

    # ---- Build NON-ROI mask (1 = process here) ----
    non_roi_mask = np.zeros((H, W), dtype=np.uint8)
    for y in range(H):
        lx = left_x_per_row[y]
        non_roi_mask[y, :lx] = 1
        non_roi_mask[y, right_x:] = 1

    # ---- Start from original image ----
    output = img.copy()

    # ---- Extract non-ROI pixels ----
    non_roi_img = img.copy()
    non_roi_img[non_roi_mask == 0] = 0

    # ---- Apply destriping ONLY to non-ROI ----
    if destripe_strength > 0:
        non_roi_img = vertical_destripe(non_roi_img, destripe_strength)

    # ---- Apply NLM ONLY to non-ROI ----
    if nlm_h > 0:
        non_roi_img = cv2.fastNlMeansDenoisingColored(
            non_roi_img, None, nlm_h, nlm_h, 7, 21
        )

    # ---- Write processed non-ROI back ----
    output[non_roi_mask == 1] = non_roi_img[non_roi_mask == 1]

    # ---- Apply alpha + gamma ONLY inside ROI ----
    for y in range(H):
        lx = left_x_per_row[y]
        rx = right_x
        roi_row = output[y, lx:rx]
        roi_row = cv2.convertScaleAbs(roi_row, alpha=alpha, beta=0)
        roi_row = apply_gamma(roi_row, gamma)
        output[y, lx:rx] = roi_row

    cv2.imshow("Preview", output)

    key = cv2.waitKey(30)
    if key == 27 or key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("postprocess-atharva.png", output)
        print("Saved postprocess-atharva.png")

cv2.destroyAllWindows()
