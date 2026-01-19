#!/usr/bin/env python3
import cv2
import numpy as np
import os

# ================= USER CONFIG ================= #
INPUT_PATH = r"D:\Dhrishti\Speed_exposure_gain\4_700_0\m3\frame_00350.jpg"
OUTPUT_PATH = r"D:\Dhrishti\Detection_code\retinex_msr.png"
# =============================================== #

def msr(gray, sigmas, weights):
    g = gray.astype(np.float32) + 1.0  # avoid log(0)
    log_g = np.log(g)
    ret = np.zeros_like(g, dtype=np.float32)
    wsum = max(sum(weights), 1e-6)
    for sigma, w in zip(sigmas, weights):
        blur = cv2.GaussianBlur(g, (0, 0), sigma)
        ret += (log_g - np.log(blur + 1.0)) * (w / wsum)
    return ret

def normalize_msr(msr_img, gain, offset):
    m = msr_img * gain + offset
    m = cv2.normalize(m, None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(m, 0, 255).astype(np.uint8)

def read_params():
    s1 = max(cv2.getTrackbarPos("Sigma 1", "Controls"), 1)
    s2 = max(cv2.getTrackbarPos("Sigma 2", "Controls"), 1)
    s3 = max(cv2.getTrackbarPos("Sigma 3", "Controls"), 1)
    w1 = cv2.getTrackbarPos("W1 x100", "Controls") / 100.0
    w2 = cv2.getTrackbarPos("W2 x100", "Controls") / 100.0
    w3 = cv2.getTrackbarPos("W3 x100", "Controls") / 100.0
    gain = cv2.getTrackbarPos("Gain x100", "Controls") / 100.0
    offset = (cv2.getTrackbarPos("Offset", "Controls") - 128) * 1.0
    gamma = cv2.getTrackbarPos("Gamma x100", "Controls") / 100.0
    return {
        "sigmas": [s1, s2, s3],
        "weights": [w1, w2, w3],
        "gain": max(gain, 0.01),
        "offset": offset,
        "gamma": max(gamma, 0.01),
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

    cv2.createTrackbar("Sigma 1", "Controls", 15, 200, lambda x: None)
    cv2.createTrackbar("Sigma 2", "Controls", 60, 300, lambda x: None)
    cv2.createTrackbar("Sigma 3", "Controls", 160, 500, lambda x: None)
    cv2.createTrackbar("W1 x100", "Controls", 33, 100, lambda x: None)
    cv2.createTrackbar("W2 x100", "Controls", 33, 100, lambda x: None)
    cv2.createTrackbar("W3 x100", "Controls", 34, 100, lambda x: None)
    cv2.createTrackbar("Gain x100", "Controls", 100, 300, lambda x: None)
    cv2.createTrackbar("Offset", "Controls", 128, 255, lambda x: None)
    cv2.createTrackbar("Gamma x100", "Controls", 100, 200, lambda x: None)

    while True:
        params = read_params()
        ret = msr(gray, params["sigmas"], params["weights"])
        out = normalize_msr(ret, params["gain"], params["offset"])
        if params["gamma"] != 1.0:
            g = out.astype(np.float32) * (1.0 / 255.0)
            g = np.power(g, 1.0 / max(params["gamma"], 1e-6)).astype(np.float32)
            out = np.clip(g * 255.0, 0, 255).astype(np.uint8)
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
