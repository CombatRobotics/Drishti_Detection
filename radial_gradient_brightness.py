#!/usr/bin/env python3
import cv2
import numpy as np
import os

# ================= USER CONFIG ================= #
INPUT_PATH = r"D:\rosbags_17thJan\ace_delhi_1_return_20260115_044439\Frames\ace_delhi_1_return_20260115_04443920260117_125830_20260119_122032\frame_2931.png"
OUTPUT_PATH = r"D:\Dhrishti\Detection_code\radialgradientmasking.png"
# =============================================== #

def build_radial_mask(h, w, cx, cy, radius, softness):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    dist = np.sqrt(dx * dx + dy * dy)
    r = max(radius, 1.0)
    s = max(softness, 1.0)
    t = (dist - r) / s
    t = np.clip(t, 0.0, 1.0)
    # smoothstep for smooth rolloff
    t = t * t * (3.0 - 2.0 * t)
    return t  # 0 inside radius, 1 outside

def apply_selective_brightness(gray, mask, strength, gamma):
    g = gray.astype(np.float32) * (1.0 / 255.0)
    if gamma != 1.0:
        g = np.power(g, 1.0 / max(gamma, 1e-6)).astype(np.float32)

    # Blend toward darker only in masked region
    dark = g * (1.0 - strength)
    out = g * (1.0 - mask) + dark * mask
    out = np.clip(out, 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)

def read_params(w, h):
    cx = cv2.getTrackbarPos("Center X", "Controls")
    cy = cv2.getTrackbarPos("Center Y", "Controls")
    radius = cv2.getTrackbarPos("Radius", "Controls")
    softness = cv2.getTrackbarPos("Softness", "Controls")
    strength = cv2.getTrackbarPos("Darken x100", "Controls") / 100.0
    gamma = cv2.getTrackbarPos("Gamma x100", "Controls") / 100.0
    return {
        "cx": np.clip(cx, 0, w - 1),
        "cy": np.clip(cy, 0, h - 1),
        "radius": max(radius, 1),
        "softness": max(softness, 1),
        "strength": np.clip(strength, 0.0, 1.0),
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

    cv2.createTrackbar("Center X", "Controls", int(w * 0.5), w - 1, lambda x: None)
    cv2.createTrackbar("Center Y", "Controls", int(h * 0.5), h - 1, lambda x: None)
    cv2.createTrackbar("Radius", "Controls", int(min(w, h) * 0.35), max(w, h), lambda x: None)
    cv2.createTrackbar("Softness", "Controls", int(min(w, h) * 0.15), max(w, h), lambda x: None)
    cv2.createTrackbar("Darken x100", "Controls", 35, 100, lambda x: None)
    cv2.createTrackbar("Gamma x100", "Controls", 100, 200, lambda x: None)

    while True:
        params = read_params(w, h)
        mask = build_radial_mask(
            h, w, params["cx"], params["cy"], params["radius"], params["softness"]
        )
        out = apply_selective_brightness(gray, mask, params["strength"], params["gamma"])
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
