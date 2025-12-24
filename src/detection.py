from ultralytics import YOLO
import numpy as np
import cv2

def disable_inplace(model):
    import torch.nn as nn
    for child in model.children():
        if hasattr(child, 'inplace'):
            try:
                child.inplace = False
            except:
                pass
        disable_inplace(child)

def load_yolo(model_path):
    model = YOLO(str(model_path))
    try:
        disable_inplace(model.model)
    except Exception:
        pass
    model.model.eval()
    return model

def run_yolo(model, image_bgr, conf=0.15):
    """
    image_bgr: numpy BGR image (H,W,3)
    returns ultralytics results[0]
    """
    out = model.predict(source=image_bgr, conf=conf, verbose=False)
    return out[0]

def extract_detections(result):
    """
    returns (boxes_np, confs_list)
    boxes_np shape: (N,4) xyxy
    """
    if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy().tolist()
    else:
        boxes = np.array([])
        confs = []
    return boxes, confs