from ultralytics import YOLO
import numpy as np
import cv2

# Load YOLOv5s model (auto-downloads)
model = YOLO("yolov5s.pt")  # You can change to yolov5m.pt or yolov5l.pt

def detect_objects(image):
    """
    Detect objects in an image using YOLOv5.

    Args:
        image (np.ndarray): RGB image.

    Returns:
        List[Dict]: List of detections with bbox, confidence, class_id, and class_name.
    """
    results = model.predict(source=image, verbose=False)[0]  # returns one result

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        detections.append({
            'bbox': (x1, y1, x2, y2),
            'confidence': conf,
            'class_id': cls_id,
            'class_name': class_name
        })

    return detections
