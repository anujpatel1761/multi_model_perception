# test_calibration.py
import numpy as np
import matplotlib.pyplot as plt
from src.dataset import get_kitti_dataset
from src.calibration_utils import lidar_to_image, filter_points_in_image, create_depth_map
from src.visualization import visualize_image, visualize_lidar_on_image, visualize_depth_map
import os

def test_calibration_and_visualization():
    # Set the path to your KITTI dataset
    data_path = os.path.join(os.getcwd(), 'data')
    
    # Create dataset instance
    dataset = get_kitti_dataset(data_path)
    
    # Load a frame
    frame_id = dataset.frame_ids[0]  # First frame
    print(f"Loading frame: {frame_id}")
    
    # Load data
    image = dataset.load_image(frame_id)
    lidar_points = dataset.load_lidar(frame_id)
    calib = dataset.load_calibration(frame_id)
    
    # Use only 3D coordinates from LiDAR (drop intensity)
    lidar_points_3d = lidar_points[:, :3]
    
    # Project LiDAR points to image plane
    image_points, camera_points = lidar_to_image(lidar_points_3d, calib)
    
    # Filter points that are in front of the camera (positive depth)
    mask = camera_points[:, 2] > 0
    lidar_points_3d_filtered = lidar_points_3d[mask]
    image_points_filtered = image_points[mask]
    
    # Visualize image
    visualize_image(image, title=f"Camera Image - Frame {frame_id}")
    plt.savefig('test_camera_image.png')
    
    # Visualize LiDAR points on image
    visualize_lidar_on_image(image, image_points_filtered, lidar_points_3d_filtered)
    plt.savefig('test_lidar_projection.png')
    
    # Create and visualize depth map
    depth_map = create_depth_map(lidar_points_3d_filtered, image_points_filtered, image.shape)
    visualize_depth_map(depth_map, title=f"Depth Map - Frame {frame_id}")
    plt.savefig('test_depth_map.png')
    
    print("Test completed successfully!")
    
    return image, lidar_points_3d, calib, image_points_filtered

if __name__ == "__main__":
    image, lidar_points, calib, image_points = test_calibration_and_visualization()
    
    # Keep figures open until closed by user
    plt.show()