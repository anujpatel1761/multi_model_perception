import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
from src.dataset import get_kitti_dataset
from src.calibration_utils import lidar_to_image
from src.feature_extraction import create_fusion_module
from src.fusion import FeatureFusion, GridFusion  # Direct class imports
import os
import time

def visualize_fusion_results(image, camera_features, lidar_features, fused_features, method):
    """Visualize original inputs and fusion results"""
    plt.figure(figsize=(16, 10))
    
    # Create a 2x2 grid of subplots
    plt.subplot(2, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    # Show camera features (average across channels)
    plt.subplot(2, 2, 2)
    cam_feat_vis = np.mean(camera_features, axis=2)
    plt.imshow(cam_feat_vis, cmap='viridis')
    plt.title(f"Camera Features (shape: {camera_features.shape})")
    plt.axis("off")
    
    # Show LiDAR features (average across channels)
    plt.subplot(2, 2, 3)
    lidar_feat_vis = np.mean(lidar_features, axis=2)
    plt.imshow(lidar_feat_vis, cmap='plasma')
    plt.title(f"LiDAR Features (shape: {lidar_features.shape})")
    plt.axis("off")
    
    # Show fused features
    plt.subplot(2, 2, 4)
    fused_feat_vis = np.mean(fused_features, axis=2)
    plt.imshow(fused_feat_vis, cmap='jet')
    plt.title(f"Fused Features - {method} (shape: {fused_features.shape})")
    plt.axis("off")
    
    plt.tight_layout()
    return plt.gcf()

def test_fusion_methods():
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
    
    # Create feature extraction module
    print("Creating feature extractors...")
    fusion_module = create_fusion_module(camera_model='resnet18', voxel_size=(0.2, 0.2, 0.2))
    
    # Extract features
    print("Extracting features...")
    start_time = time.time()
    features = fusion_module.fuse_features(image, lidar_points, calib)
    processing_time = time.time() - start_time
    print(f"Feature extraction completed in {processing_time:.2f} seconds.")
    
    # Get individual features
    camera_features = features['camera_features']
    lidar_features = features['lidar_features']
    
    # Print feature shapes
    print(f"Camera feature shape: {camera_features.shape}")
    print(f"LiDAR BEV feature shape: {lidar_features.shape}")
    
    # Test different fusion methods
    fusion_methods = ['concat', 'sum', 'attention']
    if torch.cuda.is_available():
        fusion_methods.append('learned')
    
    for method in fusion_methods:
        print(f"\nTesting {method} fusion...")
        
        # Create fusion module - UPDATED
        fusion = FeatureFusion(fusion_method=method, use_gpu=torch.cuda.is_available())
        
        # Apply fusion
        start_time = time.time()
        fused_features = fusion.fuse_features(camera_features, lidar_features)
        fusion_time = time.time() - start_time
        
        print(f"Fusion completed in {fusion_time:.4f} seconds.")
        print(f"Fused feature shape: {fused_features.shape}")
        
        # Visualize results
        fig = visualize_fusion_results(image, camera_features, lidar_features, fused_features, method)
        plt.savefig(f'test_fusion_{method}.png')
        plt.close(fig)
    
    # Test grid fusion
    print("\nTesting grid fusion...")
    # UPDATED
    grid_fusion = GridFusion(grid_size=(50, 50), cell_size=(1.0, 1.0), fusion_method='max')
    
    # Need original points for grid fusion
    lidar_points_3d = lidar_points[:, :3]
    
    # Project LiDAR points to image for visualization
    image_points, _ = lidar_to_image(lidar_points_3d, calib)
    
    # Apply grid fusion
    start_time = time.time()
    grid_features = grid_fusion.create_grid(camera_features, lidar_features, calib, image.shape, lidar_points_3d)
    grid_fusion_time = time.time() - start_time
    
    print(f"Grid fusion completed in {grid_fusion_time:.4f} seconds.")
    print(f"Grid feature shape: {grid_features.shape}")
    
    # Visualize grid fusion
    plt.figure(figsize=(10, 8))
    grid_vis = np.mean(grid_features, axis=2)
    plt.imshow(grid_vis, cmap='viridis')
    plt.title(f"Grid Fusion Features (shape: {grid_features.shape})")
    plt.colorbar(label='Feature Magnitude')
    plt.axis('equal')
    plt.savefig('test_grid_fusion.png')
    
    print("\nFusion testing completed successfully!")
    return camera_features, lidar_features, fused_features, grid_features

if __name__ == "__main__":
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Run the test
    camera_features, lidar_features, fused_features, grid_features = test_fusion_methods()
    
    # Show interactive plots
    plt.show()