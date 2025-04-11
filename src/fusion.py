"""
fusion.py
This file implements your feature-level fusion strategies:
- Grid-based or region-based fusion methods
- Feature concatenation/combination algorithms
- Weighted fusion implementations
- Post-fusion processing
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class FeatureFusion:
    def __init__(self, fusion_method='concat', use_gpu=False):
        """
        Initialize the feature fusion module
        
        Args:
            fusion_method: Method to fuse features ('concat', 'sum', 'attention', or 'learned')
            use_gpu: Whether to use GPU for learned fusion
        """
        self.fusion_method = fusion_method
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        # Initialize fusion network if using learned fusion
        if fusion_method == 'learned':
            self.fusion_network = self._create_fusion_network()
            if self.use_gpu:
                self.fusion_network.cuda()
            self.fusion_network.eval()
    
    def _create_fusion_network(self):
        """
        Create a simple fusion network
        
        Returns:
            PyTorch fusion network
        """
        # Simple fusion network with conv layers
        network = nn.Sequential(
            nn.Conv2d(519, 256, kernel_size=3, padding=1),  # 512 (camera) + 7 (lidar) = 519
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1)
        )
        return network
    
    def resize_and_align(self, camera_features, lidar_features):
        """
        Resize and align camera and LiDAR features to a common spatial resolution
        
        Args:
            camera_features: Camera features from CNN (H_cam, W_cam, C_cam)
            lidar_features: LiDAR BEV features (H_lidar, W_lidar, C_lidar)
            
        Returns:
            Resized and aligned features with the same spatial dimensions
        """
        # Determine target size (use camera feature map size)
        target_height, target_width = camera_features.shape[:2]
        
        # Resize LiDAR features to match camera feature map size
        lidar_resized = cv2.resize(lidar_features, (target_width, target_height))
        
        return camera_features, lidar_resized
    
    def concatenate_fusion(self, camera_features, lidar_features):
        """
        Fuse features by concatenation along channel dimension
        
        Args:
            camera_features: Camera features (H, W, C_cam)
            lidar_features: LiDAR features (H, W, C_lidar)
            
        Returns:
            Concatenated features (H, W, C_cam + C_lidar)
        """
        # Ensure features are aligned
        camera_aligned, lidar_aligned = self.resize_and_align(camera_features, lidar_features)
        
        # Concatenate along channel dimension
        fused_features = np.concatenate([camera_aligned, lidar_aligned], axis=2)
        
        return fused_features
    
    def weighted_sum_fusion(self, camera_features, lidar_features, cam_weight=0.5, lidar_weight=0.5):
        """
        Fuse features by weighted sum
        
        Args:
            camera_features: Camera features (H, W, C_cam)
            lidar_features: LiDAR features (H, W, C_lidar)
            cam_weight: Weight for camera features
            lidar_weight: Weight for LiDAR features
            
        Returns:
            Fused features by weighted average across channels
        """
        # Ensure features are aligned
        camera_aligned, lidar_aligned = self.resize_and_align(camera_features, lidar_features)
        
        # Normalize features to have similar scale
        cam_norm = camera_aligned / np.max(camera_aligned)
        lidar_norm = lidar_aligned / np.max(lidar_aligned)
        
        # Calculate weighted sum for each channel separately
        cam_channels = cam_norm.shape[2]
        lidar_channels = lidar_norm.shape[2]
        
        # For simplicity, we'll average across channels and then combine
        cam_avg = np.mean(cam_norm, axis=2, keepdims=True)
        lidar_avg = np.mean(lidar_norm, axis=2, keepdims=True)
        
        # Weighted sum
        fused_features = cam_weight * cam_avg + lidar_weight * lidar_avg
        
        return fused_features
    
    def attention_fusion(self, camera_features, lidar_features):
        """
        Fuse features using attention mechanism
        
        Args:
            camera_features: Camera features (H, W, C_cam)
            lidar_features: LiDAR features (H, W, C_lidar)
            
        Returns:
            Fused features with attention weighting
        """
        # Ensure features are aligned
        camera_aligned, lidar_aligned = self.resize_and_align(camera_features, lidar_features)
        
        # Calculate attention weights based on feature magnitudes
        cam_magnitude = np.sum(np.abs(camera_aligned), axis=2, keepdims=True)
        lidar_magnitude = np.sum(np.abs(lidar_aligned), axis=2, keepdims=True)
        
        # Normalize attention weights
        total_magnitude = cam_magnitude + lidar_magnitude + 1e-10  # avoid division by zero
        cam_attention = cam_magnitude / total_magnitude
        lidar_attention = lidar_magnitude / total_magnitude
        
        # Average across channels for simplicity
        cam_avg = np.mean(camera_aligned, axis=2, keepdims=True)
        lidar_avg = np.mean(lidar_aligned, axis=2, keepdims=True)
        
        # Apply attention
        fused_features = cam_attention * cam_avg + lidar_attention * lidar_avg
        
        return fused_features
    
    def learned_fusion(self, camera_features, lidar_features):
        """
        Fuse features using a learned fusion network
        
        Args:
            camera_features: Camera features (H, W, C_cam)
            lidar_features: LiDAR features (H, W, C_lidar)
            
        Returns:
            Fused features from the fusion network
        """
        # Ensure features are aligned
        camera_aligned, lidar_aligned = self.resize_and_align(camera_features, lidar_features)
        
        # Convert to PyTorch tensors
        # Reshape to (B, C, H, W) format for PyTorch
        camera_tensor = torch.from_numpy(np.transpose(camera_aligned, (2, 0, 1))).float().unsqueeze(0)
        lidar_tensor = torch.from_numpy(np.transpose(lidar_aligned, (2, 0, 1))).float().unsqueeze(0)
        
        if self.use_gpu:
            camera_tensor = camera_tensor.cuda()
            lidar_tensor = lidar_tensor.cuda()
        
        # Concatenate features along channel dimension
        combined_features = torch.cat([camera_tensor, lidar_tensor], dim=1)
        
        # Apply fusion network
        with torch.no_grad():
            fused_features = self.fusion_network(combined_features)
        
        # Convert back to numpy
        if self.use_gpu:
            fused_features = fused_features.cpu()
        
        fused_features = fused_features.squeeze(0).numpy()
        
        # Transpose back to (H, W, C) format
        fused_features = np.transpose(fused_features, (1, 2, 0))
        
        return fused_features
    
    def fuse_features(self, camera_features, lidar_features):
        """
        Fuse camera and LiDAR features based on the selected fusion method
        
        Args:
            camera_features: Camera features (H_cam, W_cam, C_cam)
            lidar_features: LiDAR BEV features (H_lidar, W_lidar, C_lidar)
            
        Returns:
            Fused features
        """
        if self.fusion_method == 'concat':
            return self.concatenate_fusion(camera_features, lidar_features)
        elif self.fusion_method == 'sum':
            return self.weighted_sum_fusion(camera_features, lidar_features)
        elif self.fusion_method == 'attention':
            return self.attention_fusion(camera_features, lidar_features)
        elif self.fusion_method == 'learned':
            return self.learned_fusion(camera_features, lidar_features)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")
    
    def create_fusion_visualization(self, camera_features, lidar_features, fused_features):
        """
        Create a visualization of the fusion process
        
        Args:
            camera_features: Camera features
            lidar_features: LiDAR features
            fused_features: Fused features
            
        Returns:
            Visualization image
        """
        # Ensure features are aligned
        camera_aligned, lidar_aligned = self.resize_and_align(camera_features, lidar_features)
        
        # Convert to visualizable format (mean across channels)
        if len(camera_aligned.shape) > 2:
            camera_vis = np.mean(camera_aligned, axis=2)
        else:
            camera_vis = camera_aligned
            
        if len(lidar_aligned.shape) > 2:
            lidar_vis = np.mean(lidar_aligned, axis=2)
        else:
            lidar_vis = lidar_aligned
            
        if len(fused_features.shape) > 2:
            fused_vis = np.mean(fused_features, axis=2)
        else:
            fused_vis = fused_features
        
        # Normalize for visualization
        def normalize(x):
            min_val = np.min(x)
            max_val = np.max(x)
            if max_val > min_val:
                return (x - min_val) / (max_val - min_val)
            return x
        
        camera_vis = normalize(camera_vis)
        lidar_vis = normalize(lidar_vis)
        fused_vis = normalize(fused_vis)
        
        # Convert to color images
        camera_colored = cv2.applyColorMap((camera_vis * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        lidar_colored = cv2.applyColorMap((lidar_vis * 255).astype(np.uint8), cv2.COLORMAP_PLASMA)
        fused_colored = cv2.applyColorMap((fused_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Stack horizontally
        visualization = np.hstack([camera_colored, lidar_colored, fused_colored])
        
        return visualization


class GridFusion:
    def __init__(self, grid_size=(50, 50), cell_size=(1.0, 1.0), fusion_method='max'):
        """
        Initialize the grid-based fusion module
        
        Args:
            grid_size: Size of grid in cells (height, width)
            cell_size: Size of each cell in meters (height, width)
            fusion_method: Method to fuse features within each cell ('max', 'mean', or 'sum')
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.fusion_method = fusion_method
    
    def create_grid(self, camera_features, lidar_features, calib, image_shape, lidar_points_3d):
        """
        Create a grid representation for fusion
        
        Args:
            camera_features: Camera features (H_cam, W_cam, C_cam)
            lidar_features: LiDAR features (H_lidar, W_lidar, C_lidar)
            calib: Calibration matrices
            image_shape: Shape of the original image (H, W, C)
            lidar_points_3d: Original LiDAR points (N, 3)
            
        Returns:
            Grid representation with fused features
        """
        grid_height, grid_width = self.grid_size
        cell_height, cell_width = self.cell_size
        
        # Initialize grid
        grid = np.zeros((grid_height, grid_width, camera_features.shape[2] + lidar_features.shape[2]), dtype=np.float32)
        
        # Define grid boundaries in world coordinates
        grid_x_min = -grid_width * cell_width / 2
        grid_x_max = grid_width * cell_width / 2
        grid_y_min = 0
        grid_y_max = grid_height * cell_height
        
        # Project LiDAR points to grid
        # For each LiDAR point, find its grid cell and add its features
        grid_points = np.zeros((grid_height, grid_width), dtype=np.int32)
        
        for i, point in enumerate(lidar_points_3d):
            x, y, z = point
            
            # Check if point is within grid boundaries
            if (grid_x_min <= x <= grid_x_max) and (grid_y_min <= y <= grid_y_max):
                # Convert to grid coordinates
                grid_x = int((x - grid_x_min) / cell_width)
                grid_y = int((y - grid_y_min) / cell_height)
                
                # Ensure within bounds
                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    grid_points[grid_y, grid_x] += 1
                    
                    # Get LiDAR features for this point
                    lidar_feat_x = int(((x - grid_x_min) / (grid_x_max - grid_x_min)) * lidar_features.shape[1])
                    lidar_feat_y = int(((y - grid_y_min) / (grid_y_max - grid_y_min)) * lidar_features.shape[0])
                    
                    # Ensure within feature bounds
                    if 0 <= lidar_feat_x < lidar_features.shape[1] and 0 <= lidar_feat_y < lidar_features.shape[0]:
                        lidar_feat = lidar_features[lidar_feat_y, lidar_feat_x]
                        
                        # Project point to image to get camera features
                        point_homogeneous = np.append(point, 1)
                        P2 = calib['P2']
                        image_point = np.dot(P2, point_homogeneous)
                        image_point = image_point[:2] / image_point[2]
                        
                        img_x, img_y = int(image_point[0]), int(image_point[1])
                        
                        # Check if point projects to image
                        if 0 <= img_x < image_shape[1] and 0 <= img_y < image_shape[0]:
                            # Get camera feature coordinates
                            cam_feat_x = int((img_x / image_shape[1]) * camera_features.shape[1])
                            cam_feat_y = int((img_y / image_shape[0]) * camera_features.shape[0])
                            
                            # Ensure within feature bounds
                            if 0 <= cam_feat_x < camera_features.shape[1] and 0 <= cam_feat_y < camera_features.shape[0]:
                                cam_feat = camera_features[cam_feat_y, cam_feat_x]
                                
                                # Combine features
                                combined_feat = np.concatenate([cam_feat, lidar_feat])
                                
                                # Update grid based on fusion method
                                if self.fusion_method == 'max':
                                    grid[grid_y, grid_x] = np.maximum(grid[grid_y, grid_x], combined_feat)
                                elif self.fusion_method == 'mean':
                                    grid[grid_y, grid_x] += combined_feat
                                elif self.fusion_method == 'sum':
                                    grid[grid_y, grid_x] += combined_feat
        
        # Normalize grid if using mean fusion
        if self.fusion_method == 'mean':
            # Avoid division by zero
            mask = grid_points > 0
            grid[mask] = grid[mask] / grid_points[mask, np.newaxis]
        
        return grid


def create_feature_fusion(method='concat', use_gpu=False):
    """
    Create a feature fusion module
    
    Args:
        method: Fusion method ('concat', 'sum', 'attention', or 'learned')
        use_gpu: Whether to use GPU for learned fusion
        
    Returns:
        Feature fusion module
    """
    return FeatureFusion(fusion_method=method, use_gpu=use_gpu)

def create_grid_fusion(grid_size=(50, 50), cell_size=(1.0, 1.0), method='max'):
    """
    Create a grid-based fusion module
    
    Args:
        grid_size: Size of grid in cells (height, width)
        cell_size: Size of each cell in meters (height, width)
        method: Method to fuse features within each cell ('max', 'mean', or 'sum')
        
    Returns:
        Grid fusion module
    """
    return GridFusion(grid_size=grid_size, cell_size=cell_size, fusion_method=method)