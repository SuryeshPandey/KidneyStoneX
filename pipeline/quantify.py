import numpy as np
import cv2

def quantify(img, boxes):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    areas, intensities, sides = [], [], []

    H, W = gray.shape
    for x1,y1,x2,y2 in boxes:
        area = (x2-x1)*(y2-y1)
        crop = gray[int(y1):int(y2), int(x1):int(x2)]
        areas.append(area)
        intensities.append(float(np.mean(crop)))
        sides.append("Left" if (x1+x2)/2 < W/2 else "Right")

    return {
        "count": len(boxes),
        "avg_area": float(np.mean(areas)) if areas else 0,
        "mean_intensity": float(np.mean(intensities)) if intensities else 0,
        "side": sides[0] if sides else "Unknown"
    }
