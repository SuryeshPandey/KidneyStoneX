def detect_stones(yolo, img_path, conf=0.2):
    res = yolo.predict(img_path, conf=conf, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy() if res.boxes else []
    scores = res.boxes.conf.cpu().numpy() if res.boxes else []
    return boxes, scores
