"""Reframe landscape video to 9:16 (1080x1920)."""
from __future__ import annotations

from pathlib import Path

OUT_W, OUT_H = 1080, 1920


def face_center_x(path: Path, start: float, end: float, src_w: int) -> float | None:
    """Median x of the largest face across sampled frames, or None if faces are rarely seen."""
    import cv2

    if not hasattr(cv2, "CascadeClassifier"):
        print("  face detection unavailable (needs opencv-python<5); using center crop")
        return None
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(path))
    samples = max(4, min(16, int((end - start) / 2)))
    xs = []
    try:
        for i in range(samples):
            t = start + (end - start) * (i + 0.5) / samples
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            scale = 640 / frame.shape[1]
            gray = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(30, 30))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                # back to full-frame pixels, then to the probed width (handles odd decoder sizes)
                xs.append((x + w / 2) / scale * src_w / frame.shape[1])
    finally:
        cap.release()
    if len(xs) < max(2, samples * 0.3):
        return None
    xs.sort()
    return xs[len(xs) // 2]


def video_filter(mode: str, src_w: int, src_h: int, center_x: float | None) -> str:
    """filter_complex chain from [0:v] to [rf], producing OUT_W x OUT_H."""
    if mode == "blur" or src_w / src_h <= OUT_W / OUT_H + 0.01:
        # Whole frame visible, blurred zoomed copy fills the background.
        return (
            "[0:v]split=2[bg][fg];"
            f"[bg]scale={OUT_W // 4}:{OUT_H // 4}:force_original_aspect_ratio=increase,"
            f"crop={OUT_W // 4}:{OUT_H // 4},gblur=sigma=12,eq=brightness=-0.1,scale={OUT_W}:{OUT_H}[bgb];"
            f"[fg]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease:flags=lanczos[fgs];"
            "[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1[rf]"
        )
    crop_w = int(src_h * OUT_W / OUT_H) // 2 * 2
    cx = center_x if center_x is not None else src_w / 2
    x = int(min(max(cx - crop_w / 2, 0), src_w - crop_w))
    return f"[0:v]crop={crop_w}:{src_h}:{x}:0,scale={OUT_W}:{OUT_H}:flags=lanczos,setsar=1[rf]"
