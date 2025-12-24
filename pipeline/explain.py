import numpy as np
import cv2

def occlusion_heatmap(yolo, img, box, base_conf, patch=20, stride=10):
    H, W = img.shape[:2]
    x1,y1,x2,y2 = map(int, box)
    heat = np.zeros((H,W), np.float32)

    for y in range(y1, y2, stride):
        for x in range(x1, x2, stride):
            masked = img.copy()
            masked[y:y+patch, x:x+patch] = 0
            out = yolo.predict(masked, conf=0.05, verbose=False)[0]
            conf = out.boxes.conf[0].item() if out.boxes else 0
            heat[y:y+patch, x:x+patch] += base_conf - conf

    heat = heat / (heat.max() + 1e-8)
    return heat
