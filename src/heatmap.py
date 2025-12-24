import numpy as np
import cv2
from src.detection import run_yolo

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1: return 0.0
    inter = (x2-x1)*(y2-y1)
    area_a = max(1,(a[2]-a[0]))*max(1,(a[3]-a[1]))
    area_b = max(1,(b[2]-b[0]))*max(1,(b[3]-b[1]))
    return inter / (area_a + area_b - inter + 1e-12)

def compute_occlusion_heatmap(orig_bgr, orig_box, yolo_model, patch_size=20, stride=10, bg_fill=(0,0,0)):
    """
    Recreates occlusion sensitivity heatmap logic from notebook.
    orig_box: [x1,y1,x2,y2] numpy or list
    returns an overlay BGR image with heatmap applied.
    """
    H, W = orig_bgr.shape[:2]
    orig_box = list(map(int, orig_box))
    baseline_out = run_yolo(yolo_model, orig_bgr, conf=0.05)
    baseline_conf = 0.0
    if hasattr(baseline_out, "boxes") and baseline_out.boxes is not None and len(baseline_out.boxes) > 0:
        baseline_conf = float(baseline_out.boxes.conf.cpu().numpy()[0])
    x1,y1,x2,y2 = orig_box
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    bw = x2 - x1; bh = y2 - y1
    if bw <= 0 or bh <= 0:
        raise RuntimeError("Invalid bounding box for heatmap")
    heatmap = np.zeros((bh, bw), dtype=np.float32)
    counts = np.zeros_like(heatmap)
    xs = list(range(x1, x2, stride))
    ys = list(range(y1, y2, stride))
    if len(xs) == 0: xs = [x1]
    if len(ys) == 0: ys = [y1]
    if xs[-1] + patch_size < x2:
        xs.append(max(x2 - patch_size, x1))
    if ys[-1] + patch_size < y2:
        ys.append(max(y2 - patch_size, y1))
    # imgsz detection (ultralytics model arg)
    try:
        imgsz_val = yolo_model.model.args.get('imgsz', 640)
        if isinstance(imgsz_val, (tuple, list)):
            imgsz = int(imgsz_val[0])
        else:
            imgsz = int(imgsz_val)
    except Exception:
        imgsz = 640

    for px in xs:
        for py in ys:
            sx = int(px); sy = int(py)
            ex = int(min(px + patch_size, W)); ey = int(min(py + patch_size, H))
            masked = orig_bgr.copy()
            masked[sy:ey, sx:ex] = bg_fill
            out = yolo_model.predict(source=masked, conf=0.05, verbose=False, imgsz=imgsz)
            r = out[0]
            found_conf = 0.0
            if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
                cand_boxes = r.boxes.xyxy.cpu().numpy()
                cand_confs = r.boxes.conf.cpu().numpy()
                best_iou = 0.0; best_conf = 0.0
                for cb, cc in zip(cand_boxes, cand_confs):
                    cbf = cb.astype(float)
                    iouv = iou_xyxy(orig_box, cbf)
                    if iouv > best_iou:
                        best_iou = iouv; best_conf = float(cc)
                if best_iou >= 0.2:
                    found_conf = best_conf
                else:
                    found_conf = 0.0
            else:
                found_conf = 0.0
            drop = baseline_conf - found_conf
            rx1 = sx - x1; ry1 = sy - y1
            rx2 = min(ex, x2) - x1; ry2 = min(ey, y2) - y1
            if rx2 > rx1 and ry2 > ry1:
                heatmap[ry1:ry2, rx1:rx2] += drop
                counts[ry1:ry2, rx1:rx2] += 1

    mask_nonzero = counts > 0
    heatmap[mask_nonzero] = heatmap[mask_nonzero] / (counts[mask_nonzero] + 1e-12)
    if heatmap.max() > 0:
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-12)
    else:
        hm_norm = heatmap
    hm_full = np.zeros((H, W), dtype=np.float32)
    if bw > 0 and bh > 0:
        hm_full[y1:y2, x1:x2] = cv2.resize(hm_norm, (bw, bh), interpolation=cv2.INTER_LINEAR)
    hm_full = hm_full - hm_full.min()
    if hm_full.max() > 0:
        hm_full = hm_full / (hm_full.max() + 1e-8)
    # color map and overlay
    hm_color_bgr = cv2.applyColorMap((hm_full * 255).astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)
    orig_float_bgr = orig_bgr.astype(np.float32)
    alpha = 0.45
    overlay_heat_bgr = ( (1 - alpha) * orig_float_bgr + (alpha) * hm_color_bgr ).astype(np.uint8)
    # draw original box and conf label
    cv2.rectangle(overlay_heat_bgr, (x1,y1), (x2,y2), (0,255,0), 2)
    cv2.putText(overlay_heat_bgr, f"orig_conf={baseline_conf:.2f}", (x1, max(12,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return overlay_heat_bgr
