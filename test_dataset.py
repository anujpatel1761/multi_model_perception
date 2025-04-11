import numpy as np
import matplotlib.pyplot as plt
from src.dataset import get_kitti_dataset
import os

def test_dataset_loading():
    # Set the path to your KITTI dataset
    data_path = os.path.join(os.getcwd(), 'data')
    
    # Create dataset instance
    dataset = get_kitti_dataset(data_path)
    
    # Print basic information
    print(f"Dataset contains {len(dataset)} frames")
    print(f"First few frame IDs: {dataset.frame_ids[:5]}")
    
    # Test loading a single frame (using the first frame)
    frame_id = dataset.frame_ids[0]
    print(f"\nLoading frame: {frame_id}")
    
    # Load individual components and check dimensions
    image = dataset.load_image(frame_id)
    lidar = dataset.load_lidar(frame_id)
    calib = dataset.load_calibration(frame_id)
    
    print(f"Image shape: {image.shape}")
    print(f"LiDAR point cloud shape: {lidar.shape}")
    print(f"Number of points in LiDAR scan: {lidar.shape[0]}")
    
    # Print calibration matrices
    print("\nCalibration Matrices:")
    for key, value in calib.items():
        print(f"{key}: Shape {value.shape}")
    
    # Display the image
    plt.figure(figsize=(10, 6))
    plt.imshow(image)
    plt.title(f"Camera Image - Frame {frame_id}")
    plt.axis('off')
    plt.savefig('test_camera_image.png')
    
    # Display LiDAR points (top-down view)
    plt.figure(figsize=(10, 10))
    plt.scatter(lidar[:, 0], lidar[:, 1], s=0.5, c=lidar[:, 2], cmap='viridis')
    plt.title(f"LiDAR Point Cloud (Top-Down) - Frame {frame_id}")
    plt.xlabel('X (meters)')
    plt.ylabel('Y (meters)')
    plt.axis('equal')
    plt.savefig('test_lidar_topdown.png')
    
    # Try to load labels if in training set
    try:
        labels = dataset.load_labels(frame_id)
        print(f"\nNumber of labeled objects: {len(labels)}")
        if len(labels) > 0:
            print(f"First object: {labels[0]}")
    except Exception as e:
        print(f"Error loading labels: {e}")
    
    print("\nTest completed successfully!")
    
    return dataset, frame_id, image, lidar, calib

if __name__ == "__main__":
    dataset, frame_id, image, lidar, calib = test_dataset_loading()
    
    # Keep figures open until closed by user
    plt.show()