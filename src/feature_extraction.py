"""
feature_extraction.py
This file handles extracting features from both sensors:
- Camera feature extraction (edges, textures, CNN features)
- LiDAR feature extraction (geometric features, height maps)
- Feature normalization functions
- Spatial alignment of features
"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from scipy.spatial import cKDTree

class LidarFeatureExtractor:
    def __init__(self, voxel_size=(0.2, 0.2, 0.2), point_range=[-30, -30, -3, 30, 30, 1], 
                 max_points_per_voxel=35):
        """
        Initialize the LiDAR feature extractor with voxelization
        
        Args:
            voxel_size: Size of each voxel (x, y, z) in meters
            point_range: Range of points to consider [x_min, y_min, z_min, x_max, y_max, z_max]
            max_points_per_voxel: Maximum number of points to consider per voxel
        """
        self.voxel_size = np.array(voxel_size)
        self.point_range = np.array(point_range)
        self.max_points_per_voxel = max_points_per_voxel
        
        # Calculate grid size
        self.grid_size = np.ceil((self.point_range[3:6] - self.point_range[0:3]) / self.voxel_size).astype(np.int32)
        
    def voxelize_points(self, points):
        """
        Voxelize point cloud data
        
        Args:
            points: Nx4 array of points (x, y, z, intensity)
            
        Returns:
            Dictionary containing voxel information
        """
        # Filter points that are in the specified range
        mask = np.all(
            (points[:, :3] >= self.point_range[:3]) & (points[:, :3] <= self.point_range[3:6]),
            axis=1
        )
        points = points[mask]
        
        # Calculate voxel coordinates for each point
        voxel_coords = ((points[:, :3] - self.point_range[:3]) / self.voxel_size).astype(np.int32)
        
        # Create a unique ID for each voxel
        voxel_ids = voxel_coords[:, 0] + voxel_coords[:, 1] * self.grid_size[0] + voxel_coords[:, 2] * self.grid_size[0] * self.grid_size[1]
        
        # Get unique voxels
        unique_voxel_ids, inverse_indices = np.unique(voxel_ids, return_inverse=True)
        
        # Create voxel to points mapping
        voxel_to_points = {}
        for i, voxel_id in enumerate(voxel_ids):
            if voxel_id not in voxel_to_points:
                voxel_to_points[voxel_id] = []
            voxel_to_points[voxel_id].append(i)
            
        # Create voxels
        num_voxels = len(unique_voxel_ids)
        voxels = np.zeros((num_voxels, self.max_points_per_voxel, points.shape[1]), dtype=np.float32)
        voxel_point_counts = np.zeros(num_voxels, dtype=np.int32)
        voxel_coords_out = np.zeros((num_voxels, 3), dtype=np.int32)
        
        # Fill voxels with points
        for i, voxel_id in enumerate(unique_voxel_ids):
            point_indices = voxel_to_points[voxel_id]
            num_points = min(len(point_indices), self.max_points_per_voxel)
            
            voxels[i, :num_points] = points[point_indices[:num_points]]
            voxel_point_counts[i] = num_points
            
            # Get voxel coordinates from any point in this voxel
            voxel_coords_out[i] = voxel_coords[point_indices[0]]
            
        return {
            'voxels': voxels,
            'voxel_coords': voxel_coords_out,
            'voxel_point_counts': voxel_point_counts
        }
        
    def extract_voxel_features(self, voxel_data):
        """
        Extract features from voxelized point cloud
        
        Args:
            voxel_data: Dictionary containing voxel information from voxelize_points()
            
        Returns:
            Dictionary containing voxel features
        """
        voxels = voxel_data['voxels']
        voxel_coords = voxel_data['voxel_coords']
        voxel_point_counts = voxel_data['voxel_point_counts']
        
        # Initialize features
        num_voxels = voxels.shape[0]
        features = np.zeros((num_voxels, 7), dtype=np.float32)  # 7 features per voxel
        
        for i in range(num_voxels):
            # Get points in this voxel
            points = voxels[i, :voxel_point_counts[i]]
            
            if voxel_point_counts[i] > 0:
                # Calculate mean position
                mean_xyz = np.mean(points[:, :3], axis=0)
                
                # Calculate variance of position
                var_xyz = np.var(points[:, :3], axis=0) if voxel_point_counts[i] > 1 else np.zeros(3)
                
                # Mean intensity
                mean_intensity = np.mean(points[:, 3]) if points.shape[1] > 3 else 0
                
                # Features: mean_x, mean_y, mean_z, var_x, var_y, var_z, mean_intensity
                features[i] = np.hstack([mean_xyz, var_xyz, mean_intensity])
        
        return {
            'voxel_features': features,
            'voxel_coords': voxel_coords,
            'voxel_point_counts': voxel_point_counts
        }
    
    def create_feature_map(self, feature_data):
        """
        Create a dense 3D feature map from sparse voxel features
        
        Args:
            feature_data: Dictionary containing feature information from extract_voxel_features()
            
        Returns:
            3D feature map (grid_size_z, grid_size_y, grid_size_x, num_features)
        """
        features = feature_data['voxel_features']
        coords = feature_data['voxel_coords']
        
        # Create empty feature map
        feature_map = np.zeros((self.grid_size[2], self.grid_size[1], self.grid_size[0], features.shape[1]), dtype=np.float32)
        
        # Fill feature map with voxel features
        for i in range(coords.shape[0]):
            x, y, z = coords[i]
            if 0 <= x < self.grid_size[0] and 0 <= y < self.grid_size[1] and 0 <= z < self.grid_size[2]:
                feature_map[z, y, x] = features[i]
                
        return feature_map
    
    def extract_features(self, points):
        """
        Extract features from LiDAR point cloud
        
        Args:
            points: Nx4 array of points (x, y, z, intensity)
            
        Returns:
            Features extracted from the point cloud
        """
        voxel_data = self.voxelize_points(points)
        feature_data = self.extract_voxel_features(voxel_data)
        feature_map = self.create_feature_map(feature_data)
        
        # Create a more compact representation - Bird's Eye View
        # We'll take max along z-axis to get a 2D representation
        bev_feature_map = np.max(feature_map, axis=0)
        
        return {
            '3d_feature_map': feature_map,
            'bev_feature_map': bev_feature_map
        }


class CameraFeatureExtractor:
    def __init__(self, model_name='resnet18', pretrained=True):
        """
        Initialize the camera feature extractor with CNN
        
        Args:
            model_name: Name of the pre-trained model to use
            pretrained: Whether to use pre-trained weights
        """
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Initialize transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize model
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        """
        Initialize the pre-trained CNN model
        
        Returns:
            PyTorch model with hooks for feature extraction
        """
        if self.model_name == 'resnet18':
            model = models.resnet18(pretrained=self.pretrained)
        elif self.model_name == 'resnet34':
            model = models.resnet34(pretrained=self.pretrained)
        elif self.model_name == 'resnet50':
            model = models.resnet50(pretrained=self.pretrained)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
        
        # Remove the fully connected layer
        model = nn.Sequential(*list(model.children())[:-2])
        
        # Set to evaluation mode
        model.eval()
        
        return model
    
    def preprocess_image(self, image):
        """
        Preprocess the image for the CNN
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Preprocessed image tensor
        """
        # Convert image to PyTorch tensor and normalize
        if isinstance(image, np.ndarray):
            # Convert to PIL Image
            image_tensor = self.transform(image)
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
        else:
            raise ValueError("Image must be a numpy array")
            
        return image_tensor
        
    def extract_features(self, image):
        """
        Extract features from camera image
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            
        Returns:
            Features extracted from the image
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_tensor)
        
        # Convert to numpy
        features_np = features.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        return features_np


class FeatureFusion:
    def __init__(self, camera_feat_extractor, lidar_feat_extractor):
        """
        Initialize the feature fusion module
        
        Args:
            camera_feat_extractor: Camera feature extractor
            lidar_feat_extractor: LiDAR feature extractor
        """
        self.camera_feat_extractor = camera_feat_extractor
        self.lidar_feat_extractor = lidar_feat_extractor
        
    def fuse_features(self, image, points_lidar, calib):
        """
        Fuse features from camera and LiDAR
        
        Args:
            image: RGB image as numpy array (H, W, 3)
            points_lidar: Nx4 array of points (x, y, z, intensity)
            calib: Calibration matrices
            
        Returns:
            Fused features
        """
        # Extract features from each modality
        camera_features = self.camera_feat_extractor.extract_features(image)
        lidar_features = self.lidar_feat_extractor.extract_features(points_lidar)
        
        # Get BEV feature map from LiDAR
        bev_features = lidar_features['bev_feature_map']
        
        # For simplicity, we'll just concatenate the camera features and LiDAR BEV features
        # In a real application, you'd want to project the features into a common space
        
        # We'll return both individual and fused features for visualization
        return {
            'camera_features': camera_features,
            'lidar_features': bev_features,
            'fused_features': {
                'camera': camera_features,
                'lidar_bev': bev_features
            }
        }


# Helper functions to create extractors
def create_feature_extractors(camera_model='resnet18', voxel_size=(0.2, 0.2, 0.2)):
    """
    Create feature extractors for camera and LiDAR
    
    Args:
        camera_model: Name of the pre-trained CNN model to use
        voxel_size: Size of voxels for LiDAR
        
    Returns:
        Camera and LiDAR feature extractors
    """
    camera_extractor = CameraFeatureExtractor(model_name=camera_model)
    lidar_extractor = LidarFeatureExtractor(voxel_size=voxel_size)
    
    return camera_extractor, lidar_extractor

def create_fusion_module(camera_model='resnet18', voxel_size=(0.2, 0.2, 0.2)):
    """
    Create a feature fusion module
    
    Args:
        camera_model: Name of the pre-trained CNN model to use
        voxel_size: Size of voxels for LiDAR
        
    Returns:
        Feature fusion module
    """
    camera_extractor, lidar_extractor = create_feature_extractors(camera_model, voxel_size)
    fusion_module = FeatureFusion(camera_extractor, lidar_extractor)
    
    return fusion_module