import cv2
import numpy as np
from pathlib import Path

def ensure_dirs(paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def read_image_bytes_as_bgr(byte_stream):
    """Read uploaded bytes into BGR cv2 image"""
    arr = np.frombuffer(byte_stream, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to decode image bytes")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img

def save_image(path, img):
    cv2.imwrite(str(path), img)

def to_bgr_rgb(bgr):
    """Convert BGR to RGB for display (Streamlit expects RGB)"""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def draw_boxes_on_image(image_bgr, boxes, scores=None):
    """Draw green boxes and labels like your notebook"""
    out = image_bgr.copy()
    for i, b in enumerate(boxes):
        x1,y1,x2,y2 = map(int, b)
        cv2.rectangle(out, (x1,y1), (x2,y2), (0,255,0), 2)
        label = f"stone {i+1}"
        if scores is not None and len(scores) > i:
            label = f"{label} {scores[i]:.2f}"
        cv2.putText(out, label, (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)
    return out
