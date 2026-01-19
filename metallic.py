#!/usr/bin/env python3
import cv2
import numpy as np
import os

# ================= USER CONFIG ================= #
INPUT_PATH = r"D:\Dhrishti\Detection_code\postprocess1.png"
OUTPUT_PATH = r"D:\Dhrishti\Detection_code\postprocess2.png"
# =============================================== #

def smoothstep(x):
    return x * x * (3.0 - 2.0 * x)

def metallic_curve(x, contrast):
    # S-curve with controlled roll-off to keep background from clipping
    denom = np.tanh(0.5 * contrast) + 1e-6
    return 0.5 + 0.5 * np.tanh((x - 0.5) * contrast) / denom

def metallic_postprocess(gray, params):
    g = gray.astype(np.float32) * np.float32(1.0 / 255.0)

    # Gentle gamma to lift exposure without boosting background too much
    gamma = params["gamma"]
    g = np.power(g, 1.0 / max(gamma, 1e-6)).astype(np.float32)

    # Texture mask from gradient magnitude (no smoothing)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    grad = grad / (grad.max() + 1e-6)

    tex_low = params["texture_low"]
    tex_high = params["texture_high"]
    t = (grad - tex_low) / max(tex_high - tex_low, 1e-6)
    t = np.clip(t, 0.0, 1.0)
    t = smoothstep(t)

    # Per-pixel contrast: compress background, keep texture/faults present
    mid = 0.5
    bg_contrast = params["bg_contrast"]
    tex_contrast = params["texture_contrast"]
    scale = bg_contrast + (tex_contrast - bg_contrast) * t
    g = mid + (g - mid) * scale

    # Add crisp texture detail without blurring
    g = g.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    detail_amount = params["detail_amount"]
    g = g + detail_amount * lap * t

    # Metallic tone curve (midtone sheen)
    metallic_strength = params["metallic_strength"]
    metallic_contrast = params["metallic_contrast"]
    if metallic_strength > 0:
        m = metallic_curve(g, metallic_contrast)
        g = (1.0 - metallic_strength) * g + metallic_strength * m

    g = np.clip(g, 0.0, 1.0)
    return (g * 255.0).astype(np.uint8)

def read_params():
    gamma = cv2.getTrackbarPos("Gamma x100", "Controls") / 100.0
    tex_low = cv2.getTrackbarPos("TexLow x100", "Controls") / 100.0
    tex_high = cv2.getTrackbarPos("TexHigh x100", "Controls") / 100.0
    if tex_high <= tex_low + 0.01:
        tex_high = min(1.0, tex_low + 0.02)
    bg_contrast = cv2.getTrackbarPos("BG Contrast x100", "Controls") / 100.0
    tex_contrast = cv2.getTrackbarPos("Tex Contrast x100", "Controls") / 100.0
    detail_amount = cv2.getTrackbarPos("Detail x100", "Controls") / 100.0
    metallic_strength = cv2.getTrackbarPos("Metallic x100", "Controls") / 100.0
    metallic_contrast = cv2.getTrackbarPos("Metal C x10", "Controls") / 10.0
    return {
        "gamma": max(gamma, 0.01),
        "texture_low": np.clip(tex_low, 0.0, 1.0),
        "texture_high": np.clip(tex_high, 0.0, 1.0),
        "bg_contrast": np.clip(bg_contrast, 0.0, 2.0),
        "texture_contrast": np.clip(tex_contrast, 0.0, 2.0),
        "detail_amount": np.clip(detail_amount, 0.0, 2.0),
        "metallic_strength": np.clip(metallic_strength, 0.0, 1.0),
        "metallic_contrast": max(metallic_contrast, 0.1),
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

    cv2.createTrackbar("Gamma x100", "Controls", 115, 200, lambda x: None)
    cv2.createTrackbar("TexLow x100", "Controls", 4, 100, lambda x: None)
    cv2.createTrackbar("TexHigh x100", "Controls", 20, 100, lambda x: None)
    cv2.createTrackbar("BG Contrast x100", "Controls", 75, 200, lambda x: None)
    cv2.createTrackbar("Tex Contrast x100", "Controls", 110, 200, lambda x: None)
    cv2.createTrackbar("Detail x100", "Controls", 55, 200, lambda x: None)
    cv2.createTrackbar("Metallic x100", "Controls", 85, 100, lambda x: None)
    cv2.createTrackbar("Metal C x10", "Controls", 30, 100, lambda x: None)

    while True:
        params = read_params()
        out = metallic_postprocess(gray, params)
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
