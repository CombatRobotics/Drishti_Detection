#!/usr/bin/env python3
import cv2
import numpy as np
import math

# ================= USER CONFIG ================= #
IMAGE_PATH = r"D:\Dhrishti\Detection_code\radialgradientmasking.png"
# =============================================== #

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
cv2.createTrackbar("CLAHE Clip x10", "Controls", 20, 100, lambda x: None)
cv2.createTrackbar("CLAHE Tile", "Controls", 8, 32, lambda x: None)

# ---------- Main loop ----------
while True:
    left_x_base = cv2.getTrackbarPos("LEFT_X", "Controls")
    right_x = cv2.getTrackbarPos("RIGHT_X", "Controls")
    right_x = max(left_x_base + 10, right_x)

    angle_deg = cv2.getTrackbarPos("LEFT_ANGLE", "Controls") - 180
    angle_rad = math.radians(angle_deg)

    clahe_clip = cv2.getTrackbarPos("CLAHE Clip x10", "Controls") / 10.0
    clahe_tile = max(2, cv2.getTrackbarPos("CLAHE Tile", "Controls"))

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
    non_roi_mask = cv2.bitwise_not(roi_mask)

    # ---- Apply CLAHE to non-ROI only ----
    out = gray.copy()
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_tile, clahe_tile))
        non_roi = clahe.apply(gray)
        out[non_roi_mask > 0] = non_roi[non_roi_mask > 0]

    output = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Preview", output)

    key = cv2.waitKey(30)
    if key == 27 or key == ord('q'):
        break
    if key == ord('s'):
        cv2.imwrite("clahe_nonroi_only.png", output)
        print("Saved clahe_nonroi_only.png")

cv2.destroyAllWindows()
