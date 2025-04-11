"""
dataset.py
This file handles loading the synchronized data from KITTI:
- Functions to find corresponding files across modalities
- Camera image loading and basic preprocessing
- LiDAR point cloud loading from binary files
- Calibration file parsing
- Batch processing for multiple frames
"""
import os
import numpy as np
import cv2
from pathlib import Path

class KittiDataset:
    def __init__(self, base_path, split='training'):
        """
        Initialize the KITTI dataset loader
        
        Args:
            base_path: Path to the KITTI dataset directory
            split: Either 'training' or 'testing'
        """
        self.base_path = Path(base_path)
        self.split = split
        
        # Define paths to different data types
        self.camera_path = self.base_path / 'camera' / 'data_object_image_3' / split / 'image_3'
        self.lidar_path = self.base_path / 'lidar' / 'data_object_velodyne' / split / 'velodyne'
        self.calib_path = self.base_path / 'camera_calibration' / 'data_object_calib' / split / 'calib'
        
        # If using training data, also get label path
        if split == 'training':
            self.label_path = self.base_path / 'bounding_box' / 'data_object_label_2' / split / 'label_2'
        
        # Verify paths exist
        self._check_paths()
        
        # Get list of available frame IDs
        self.frame_ids = self._get_frame_ids()
        
    def _check_paths(self):
        """Verify that all data paths exist"""
        if not self.camera_path.exists():
            raise FileNotFoundError(f"Camera path does not exist: {self.camera_path}")
        if not self.lidar_path.exists():
            raise FileNotFoundError(f"LiDAR path does not exist: {self.lidar_path}")
        if not self.calib_path.exists():
            raise FileNotFoundError(f"Calibration path does not exist: {self.calib_path}")
        
        if self.split == 'training' and not self.label_path.exists():
            raise FileNotFoundError(f"Label path does not exist: {self.label_path}")
    
    def _get_frame_ids(self):
        """Get list of available frame IDs based on available image files"""
        frame_ids = []
        for file_path in sorted(self.camera_path.glob('*.png')):
            frame_id = file_path.stem  # Get filename without extension
            frame_ids.append(frame_id)
        return frame_ids
    
    def __len__(self):
        """Return the number of frames in the dataset"""
        return len(self.frame_ids)
    
    def get_frame_path(self, frame_id):
        """
        Get paths to all data for a specific frame
        
        Args:
            frame_id: Frame ID (as string, e.g., '000123')
            
        Returns:
            Dictionary with paths to each data type
        """
        paths = {
            'image': self.camera_path / f"{frame_id}.png",
            'lidar': self.lidar_path / f"{frame_id}.bin",
            'calib': self.calib_path / f"{frame_id}.txt"
        }
        
        if self.split == 'training':
            paths['label'] = self.label_path / f"{frame_id}.txt"
            
        return paths
    
    def load_image(self, frame_id):
        """
        Load camera image for a specific frame
        
        Args:
            frame_id: Frame ID (as string, e.g., '000123')
            
        Returns:
            RGB image as numpy array (H, W, 3)
        """
        image_path = self.get_frame_path(frame_id)['image']
        # OpenCV loads as BGR, convert to RGB
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def load_lidar(self, frame_id):
        """
        Load LiDAR point cloud for a specific frame
        
        Args:
            frame_id: Frame ID (as string, e.g., '000123')
            
        Returns:
            Point cloud as numpy array (N, 4) - x, y, z, intensity
        """
        lidar_path = self.get_frame_path(frame_id)['lidar']
        # KITTI stores LiDAR as binary float32 arrays with 4 values per point
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)
        return points
    
    def load_calibration(self, frame_id):
        """
        Load calibration data for a specific frame
        
        Args:
            frame_id: Frame ID (as string, e.g., '000123')
            
        Returns:
            Dictionary with calibration matrices
        """
        calib_path = self.get_frame_path(frame_id)['calib']
        
        # Read the calibration file
        with open(str(calib_path), 'r') as f:
            lines = f.readlines()
        
        # Parse the calibration data
        calib_data = {}
        for line in lines:
            line = line.strip()
            if line == '' or ':' not in line:
                continue
                
            try:
                key, value = line.split(':', 1)
                # Convert matrix strings to numpy arrays
                calib_data[key] = np.array([float(x) for x in value.split()])
            except Exception as e:
                print(f"Warning: Could not parse calibration line: {line}")
                continue
        
        # Reshape matrices
        if 'P0' in calib_data:  # Camera projection matrices
            calib_data['P0'] = calib_data['P0'].reshape(3, 4)
        if 'P1' in calib_data:
            calib_data['P1'] = calib_data['P1'].reshape(3, 4)
        if 'P2' in calib_data:
            calib_data['P2'] = calib_data['P2'].reshape(3, 4)
        if 'P3' in calib_data:
            calib_data['P3'] = calib_data['P3'].reshape(3, 4)
        if 'R0_rect' in calib_data:  # Rectification matrix
            calib_data['R0_rect'] = calib_data['R0_rect'].reshape(3, 3)
        if 'Tr_velo_to_cam' in calib_data:  # Velodyne to camera transform
            calib_data['Tr_velo_to_cam'] = calib_data['Tr_velo_to_cam'].reshape(3, 4)
        if 'Tr_imu_to_velo' in calib_data:  # IMU to Velodyne transform
            calib_data['Tr_imu_to_velo'] = calib_data['Tr_imu_to_velo'].reshape(3, 4)
            
        return calib_data
        
    def load_labels(self, frame_id):
        """
        Load object labels for a specific frame (only for training split)
        
        Args:
            frame_id: Frame ID (as string, e.g., '000123')
            
        Returns:
            List of dictionaries, each containing object information
        """
        if self.split != 'training':
            raise ValueError("Labels are only available for training split")
            
        label_path = self.get_frame_path(frame_id)['label']
        
        objects = []
        with open(str(label_path), 'r') as f:
            for line in f.readlines():
                values = line.strip().split()
                obj = {
                    'type': values[0],
                    'truncation': float(values[1]),
                    'occlusion': int(values[2]),
                    'alpha': float(values[3]),
                    'bbox': [float(values[4]), float(values[5]), float(values[6]), float(values[7])],
                    'dimensions': [float(values[8]), float(values[9]), float(values[10])],
                    'location': [float(values[11]), float(values[12]), float(values[13])],
                    'rotation_y': float(values[14])
                }
                objects.append(obj)
                
        return objects
    
    def get_frame_data(self, frame_id):
        """
        Load all data for a specific frame
        
        Args:
            frame_id: Frame ID (as string, e.g., '000123')
            
        Returns:
            Dictionary with all data for the frame
        """
        data = {
            'image': self.load_image(frame_id),
            'lidar': self.load_lidar(frame_id),
            'calib': self.load_calibration(frame_id)
        }
        
        if self.split == 'training':
            data['labels'] = self.load_labels(frame_id)
            
        return data


# Helper function to get a dataset instance
def get_kitti_dataset(base_path, split='training'):
    """
    Create and return a KITTI dataset instance
    
    Args:
        base_path: Path to the KITTI dataset directory
        split: Either 'training' or 'testing'
        
    Returns:
        KittiDataset instance
    """
    return KittiDataset(base_path, split)