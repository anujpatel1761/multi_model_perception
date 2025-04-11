"""
visualization.py
This file provides visual feedback for debugging and results:
- 2D image visualization with overlays
- 3D point cloud visualization 
- Combined visualizations (projected points)
- Feature map visualizations
- Before/after fusion comparisons
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

def visualize_image(image, title='Image', figsize=(10, 6)):
    """
    Visualize a single image
    
    Args:
        image: Image to visualize (RGB)
        title: Title of the plot
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_lidar_on_image(image, points_image, points_lidar=None, color_by_depth=True, 
                             point_size=1, alpha=0.5, figsize=(16, 10)):
    """
    Visualize LiDAR points projected onto an image
    
    Args:
        image: RGB image
        points_image: Nx2 array of 2D points in image
        points_lidar: Nx3 array of 3D points in LiDAR coordinate (used for coloring by depth)
        color_by_depth: Whether to color points by depth
        point_size: Size of points in visualization
        alpha: Transparency of points
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display the image
    ax.imshow(image)
    
    # Filter points within image bounds
    height, width = image.shape[:2]
    mask = (points_image[:, 0] >= 0) & (points_image[:, 0] < width) & \
           (points_image[:, 1] >= 0) & (points_image[:, 1] < height)
    
    visible_points = points_image[mask]
    
    # Color by depth if requested
    if color_by_depth and points_lidar is not None:
        depths = np.sqrt(np.sum(points_lidar[mask, :3] ** 2, axis=1))
        scatter = ax.scatter(visible_points[:, 0], visible_points[:, 1], 
                             c=depths, cmap='viridis', s=point_size, alpha=alpha)
        plt.colorbar(scatter, ax=ax, label='Depth (m)')
    else:
        ax.scatter(visible_points[:, 0], visible_points[:, 1], 
                   c='red', s=point_size, alpha=alpha)
    
    ax.set_title('LiDAR Points Projected on Image')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_depth_map(depth_map, title='Depth Map', cmap='viridis', figsize=(10, 6)):
    """
    Visualize a depth map
    
    Args:
        depth_map: Depth map image
        title: Title of the plot
        cmap: Colormap to use
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.imshow(depth_map, cmap=cmap)
    plt.title(title)
    plt.colorbar(label='Normalized Depth')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_feature_fusion(image, lidar_features, camera_features, figsize=(15, 10)):
    """
    Visualize feature fusion results
    
    Args:
        image: Original RGB image
        lidar_features: LiDAR features to visualize
        camera_features: Camera features to visualize
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # LiDAR features
    if len(lidar_features.shape) == 2:
        axes[1].imshow(lidar_features, cmap='viridis')
    else:
        axes[1].imshow(lidar_features)
    axes[1].set_title('LiDAR Features')
    axes[1].axis('off')
    
    # Camera features
    if len(camera_features.shape) == 2:
        axes[2].imshow(camera_features, cmap='viridis')
    else:
        axes[2].imshow(camera_features)
    axes[2].set_title('Camera Features')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
