#!/usr/bin/env python3
import cv2
import numpy as np
import os

# ================= USER CONFIG ================= #
INPUT_PATH = r"D:\Dhrishti\Speed_exposure_gain\4_700_0\m3\frame_00350.jpg"
OUTPUT_PATH = r"D:\Dhrishti\Detection_code\hdr_toned.png"
# =============================================== #

def tonemap_hdr(gray, strength, bias, gamma, detail):
    g = gray.astype(np.float32) * (1.0 / 255.0)
    g = np.power(g, 1.0 / max(gamma, 1e-6)).astype(np.float32)

    # Log-like compression (HDR toning)
    s = max(strength, 1e-6)
    t = (np.log1p(s * g) / np.log1p(s)).astype(np.float32)

    # Midtone bias (shift)
    t = np.clip(t + bias, 0.0, 1.0)

    # Local detail (no blur/avg; use Laplacian)
    if detail > 0:
        t = t.astype(np.float32)
        lap = cv2.Laplacian(t, cv2.CV_32F, ksize=3)
        t = t + detail * lap

    t = np.clip(t, 0.0, 1.0)
    return (t * 255.0).astype(np.uint8)

def read_params():
    strength = cv2.getTrackbarPos("Strength x10", "Controls") / 10.0
    bias = (cv2.getTrackbarPos("Bias x100", "Controls") - 50) / 100.0
    gamma = cv2.getTrackbarPos("Gamma x100", "Controls") / 100.0
    detail = cv2.getTrackbarPos("Detail x100", "Controls") / 100.0
    return {
        "strength": max(strength, 0.1),
        "bias": np.clip(bias, -0.5, 0.5),
        "gamma": max(gamma, 0.01),
        "detail": np.clip(detail, 0.0, 2.0),
    }

def main():
    img = cv2.imread(INPUT_PATH, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(INPUT_PATH)

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 420, h)
    cv2.resizeWindow("Preview", w, h)
    cv2.moveWindow("Controls", 20, 20)
    cv2.moveWindow("Preview", 460, 20)

    cv2.createTrackbar("Strength x10", "Controls", 15, 100, lambda x: None)
    cv2.createTrackbar("Bias x100", "Controls", 50, 100, lambda x: None)
    cv2.createTrackbar("Gamma x100", "Controls", 100, 200, lambda x: None)
    cv2.createTrackbar("Detail x100", "Controls", 25, 200, lambda x: None)

    while True:
        params = read_params()
        out = tonemap_hdr(gray, params["strength"], params["bias"], params["gamma"], params["detail"])
        out_bgr = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Preview", out_bgr)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord("q"):
            break
        if key == ord("s"):
            os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
            cv2.imwrite(OUTPUT_PATH, out_bgr)
            print(f"Saved: {OUTPUT_PATH}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
