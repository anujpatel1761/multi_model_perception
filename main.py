import sys
import os

# Add 'src' directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.kitti_loader import KITTILoader
from utils.projection_utils import project_lidar_to_image
from detection.yolo_detector import detect_objects
from visualization.visualizer import draw_projected_lidar, draw_yolo_detections


def main():
    dataset = KITTILoader("C:/Users/anujp/Desktop/multi_model_perception/data")
    idx = "000049"

    print(f"\n🔹 Loading KITTI data for frame {idx}...")
    image = dataset.get_image(idx)
    lidar = dataset.get_lidar(idx)
    calib = dataset.get_calib(idx)

    print(f"   Image shape: {image.shape}")
    print(f"   LiDAR points: {lidar.shape[0]}")
    print(f"   Calibration keys: {list(calib.keys())}")

    # Project LiDAR
    projected_pts, _ = project_lidar_to_image(lidar, calib, image.shape)
    print(f"   Projected LiDAR points: {projected_pts.shape[0]}")

    # Detect objects
    detections = detect_objects(image)
    print(f"\n🔹 YOLOv5 detected {len(detections)} objects:")
    for det in detections:
        print(f"   - {det['class_name']} ({det['confidence']:.2f}) → {det['bbox']}")

    # Show LiDAR projection only
    print("\n📸 Showing only LiDAR projections...")
    draw_projected_lidar(image, projected_pts)

    # Show YOLO detections only
    print("📸 Showing only YOLO detections...")
    draw_yolo_detections(image, detections)


if __name__ == "__main__":
    main()
