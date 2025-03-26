import matplotlib.pyplot as plt
import cv2
import numpy as np


def draw_projected_lidar(image, projected_pts):
    """
    Show LiDAR point projection only.

    Args:
        image (np.ndarray): RGB image
        projected_pts (np.ndarray): (N, 2) image-plane coordinates from LiDAR
    """
    vis_img = image.copy()

    for pt in projected_pts.astype(int):
        x, y = pt
        if 0 <= x < vis_img.shape[1] and 0 <= y < vis_img.shape[0]:
            vis_img[y, x] = (255, 0, 0)  # Red dot

    plt.figure(figsize=(12, 6))
    plt.imshow(vis_img)
    plt.title("LiDAR Projection Only")
    plt.axis('off')
    plt.show()


def draw_yolo_detections(image, detections):
    """
    Show YOLOv5 2D detections only.

    Args:
        image (np.ndarray): RGB image
        detections (List[Dict]): Output from YOLOv5
    """
    vis_img = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        cls = det['class_name']
        conf = det['confidence']
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, f"{cls} ({conf:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    plt.figure(figsize=(12, 6))
    plt.imshow(vis_img)
    plt.title("YOLOv5 2D Detections Only")
    plt.axis('off')
    plt.show()
