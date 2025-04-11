import numpy as np
import matplotlib.pyplot as plt
from src.dataset import get_kitti_dataset
from src.calibration_utils import lidar_to_image
from src.feature_extraction import create_fusion_module
import os
import torch
import time

def visualize_features(image, camera_features, lidar_features, title="Feature Visualization"):
    """Visualize the extracted features"""
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # Plot camera features (take mean across channels for visualization)
    if len(camera_features.shape) > 2:
        camera_feat_vis = np.mean(camera_features, axis=2)
    else:
        camera_feat_vis = camera_features
    axes[1].imshow(camera_feat_vis, cmap='viridis')
    axes[1].set_title(f"Camera Features (shape: {camera_features.shape})")
    axes[1].axis("off")
    
    # Plot LiDAR BEV features (take mean across channels for visualization)
    if len(lidar_features.shape) > 2:
        lidar_feat_vis = np.mean(lidar_features, axis=2)
    else:
        lidar_feat_vis = lidar_features
    axes[2].imshow(lidar_feat_vis, cmap='plasma')
    axes[2].set_title(f"LiDAR BEV Features (shape: {lidar_features.shape})")
    axes[2].axis("off")
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig

def test_feature_extraction():
    # Set the path to your KITTI dataset
    data_path = os.path.join(os.getcwd(), 'data')
    
    # Create dataset instance
    print("Loading dataset...")
    dataset = get_kitti_dataset(data_path)
    
    # Load a frame
    frame_id = dataset.frame_ids[0]  # First frame
    print(f"Processing frame: {frame_id}")
    
    # Load data
    image = dataset.load_image(frame_id)
    lidar_points = dataset.load_lidar(frame_id)
    calib = dataset.load_calibration(frame_id)
    
    # Create feature fusion module
    print("Creating feature extractors...")
    try:
        fusion_module = create_fusion_module(camera_model='resnet18', voxel_size=(0.2, 0.2, 0.2))
        print("Feature extractors created successfully.")
    except Exception as e:
        print(f"Error creating feature extractors: {e}")
        return None, None, None
    
    # Extract and fuse features
    print("Extracting features...")
    start_time = time.time()
    try:
        features = fusion_module.fuse_features(image, lidar_points, calib)
        processing_time = time.time() - start_time
        print(f"Feature extraction completed in {processing_time:.2f} seconds.")
    except Exception as e:
        print(f"Error extracting features: {e}")
        return image, None, None
    
    # Get individual features
    camera_features = features['camera_features']
    lidar_features = features['lidar_features']
    
    # Print feature shapes
    print(f"Camera feature shape: {camera_features.shape}")
    print(f"LiDAR BEV feature shape: {lidar_features.shape}")
    
    # Visualize features
    print("Visualizing features...")
    fig = visualize_features(image, camera_features, lidar_features)
    plt.savefig('test_features.png')
    
    print("Feature extraction test completed successfully!")
    return image, camera_features, lidar_features

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Run the test
    image, camera_features, lidar_features = test_feature_extraction()
    
    # Keep figures open until closed by user
    plt.show()