"""
KITTI Dataset Loader
--------------------
This module provides functions and a class to load images, LiDAR point clouds,
calibration matrices, and 3D labels from the KITTI dataset.
"""

import os
import cv2
import numpy as np


def load_image(image_path):
    """
    Load a KITTI RGB image.

    Args:
        image_path (str): Path to the .png image.

    Returns:
        np.ndarray: Image in RGB format (H, W, 3).
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def load_pointcloud(bin_path):
    """
    Load a LiDAR point cloud from a .bin file.

    Args:
        bin_path (str): Path to the .bin file.

    Returns:
        np.ndarray: Array of shape (N, 4) — [x, y, z, reflectance].
    """
    pointcloud = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    return pointcloud


def load_labels(label_path):
    """
    Load 3D bounding box labels from KITTI label file.

    Args:
        label_path (str): Path to the label .txt file.

    Returns:
        List[Dict]: List of label dictionaries.
    """
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            obj = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox': np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])]),
                'dimensions': np.array([float(parts[8]), float(parts[9]), float(parts[10])]),  # h, w, l
                'location': np.array([float(parts[11]), float(parts[12]), float(parts[13])]),  # x, y, z
                'rotation_y': float(parts[14])
            }
            labels.append(obj)
    return labels


def load_calibration(calib_path):
    """
    Load calibration matrices from KITTI calib file.

    Args:
        calib_path (str): Path to the calib .txt file.

    Returns:
        Dict[str, np.ndarray]: Dictionary of matrices like 'P2', 'Tr_velo_to_cam', etc.
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            key, *values = line.strip().split()
            key = key.rstrip(':')  # remove trailing colon
            values = np.array(values, dtype=np.float32)

            if key.startswith('P') or key.startswith('Tr'):
                calib[key] = values.reshape(3, 4)
            elif key.startswith('R'):
                calib[key] = values.reshape(3, 3)
            else:
                calib[key] = values
    return calib


class KITTILoader:
    """
    Object-oriented wrapper to load KITTI data easily.
    """
    def __init__(self, base_path, split='training'):
        self.base_path = base_path
        self.split = split

    def get_image(self, idx):
        path = os.path.join(self.base_path, "camera/data_object_image_3", self.split, "image_3", f"{idx}.png")
        return load_image(path)

    def get_lidar(self, idx):
        path = os.path.join(self.base_path, "lidar/data_object_velodyne", self.split, "velodyne", f"{idx}.bin")
        return load_pointcloud(path)

    def get_labels(self, idx):
        path = os.path.join(self.base_path, "bounding_box/data_object_label_2", self.split, "label_2", f"{idx}.txt")
        return load_labels(path)

    def get_calib(self, idx):
        path = os.path.join(self.base_path, "camera_calibration/data_object_calib", self.split, "calib", f"{idx}.txt")
        return load_calibration(path)


# 🔍 Test this module standalone
if __name__ == "__main__":
    dataset = KITTILoader(base_path="C:/Users/anujp/Desktop/multi_model_perception/data", split="training")
    idx = "000123"

    image = dataset.get_image(idx)
    lidar = dataset.get_lidar(idx)
    labels = dataset.get_labels(idx)
    calib = dataset.get_calib(idx)

    print(f"✅ Image shape: {image.shape}")
    print(f"✅ LiDAR shape: {lidar.shape}")
    print(f"✅ Labels loaded: {len(labels)}")
    print(f"✅ Calibration keys: {list(calib.keys())}")
