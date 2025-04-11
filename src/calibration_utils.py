"""
calibration_utils.py
This file contains all the transformation logic:
- Parse KITTI calibration files (extracting matrices)
- Functions to convert between coordinate systems:
  - LiDAR to camera transformations
  - 3D to 2D projections
  - Rectification calculations
"""
import numpy as np

def read_calib_file(calib_path):
    """
    Read KITTI calibration file
    (This is a helper function for cases where you directly want to read a calib file)
    
    Args:
        calib_path: Path to the calibration file
        
    Returns:
        dict: Calibration matrices
    """
    calib = {}
    with open(calib_path, 'r') as f:
        for line in f.readlines():
            if not line or line == '\n':
                continue
            key, value = line.split(':', 1)
            calib[key] = np.array([float(x) for x in value.split()])

    # Reshape matrices
    calib_matrices = {}
    calib_matrices['P0'] = calib['P0'].reshape(3, 4)
    calib_matrices['P1'] = calib['P1'].reshape(3, 4)
    calib_matrices['P2'] = calib['P2'].reshape(3, 4)
    calib_matrices['P3'] = calib['P3'].reshape(3, 4)
    calib_matrices['R0_rect'] = calib['R0_rect'].reshape(3, 3)
    calib_matrices['Tr_velo_to_cam'] = calib['Tr_velo_to_cam'].reshape(3, 4)
    calib_matrices['Tr_imu_to_velo'] = calib['Tr_imu_to_velo'].reshape(3, 4)
    
    return calib_matrices

def lidar_to_camera_point(point, calib):
    """
    Convert a point from LiDAR coordinate to camera coordinate
    
    Args:
        point: 3D point in LiDAR coordinate [x, y, z]
        calib: Calibration dictionary with the required matrices
        
    Returns:
        np.array: 3D point in camera coordinate [x, y, z]
    """
    # Extract required matrices
    R0_rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    
    # Convert point to homogeneous coordinate
    point_h = np.hstack((point, 1))
    
    # Apply transformation: first lidar to camera, then rectification
    point_cam = np.dot(Tr_velo_to_cam, point_h)
    point_rect = np.dot(R0_rect, point_cam[:3])
    
    return point_rect

def lidar_to_camera_points(points, calib):
    """
    Convert multiple points from LiDAR coordinate to camera coordinate
    
    Args:
        points: Nx3 array of 3D points in LiDAR coordinate
        calib: Calibration dictionary with the required matrices
        
    Returns:
        np.array: Nx3 array of 3D points in camera coordinate
    """
    # Extract required matrices
    R0_rect = calib['R0_rect']
    Tr_velo_to_cam = calib['Tr_velo_to_cam']
    
    # Convert points to homogeneous coordinates (Nx4)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    
    # Apply transformation: first lidar to camera
    points_cam = np.dot(points_h, Tr_velo_to_cam.T)
    
    # Then apply rectification
    points_rect = np.dot(points_cam, R0_rect.T)
    
    return points_rect

def project_to_image(points_3d, calib, camera_id=2):
    """
    Project 3D points in camera coordinate to 2D image plane
    
    Args:
        points_3d: Nx3 array of 3D points in camera coordinate
        calib: Calibration dictionary with projection matrices
        camera_id: Which camera to use (0-3, default is 2 for left color camera)
        
    Returns:
        np.array: Nx2 array of 2D points in image plane
    """
    # Choose the appropriate projection matrix
    P = calib[f'P{camera_id}']
    
    # Convert to homogeneous coordinates
    points_3d_h = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))
    
    # Project to image plane
    points_2d_h = np.dot(points_3d_h, P.T)
    
    # Convert from homogeneous to image coordinates
    points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]
    
    return points_2d

def lidar_to_image(points, calib, camera_id=2):
    """
    Project LiDAR points to image plane in one step
    
    Args:
        points: Nx3 array of 3D points in LiDAR coordinate
        calib: Calibration dictionary with the required matrices
        camera_id: Which camera to use (0-3, default is 2 for left color camera)
        
    Returns:
        tuple: (Nx2 array of 2D points in image, Nx3 array of 3D points in camera coordinate)
    """
    # First convert to camera coordinate
    points_cam = lidar_to_camera_points(points, calib)
    
    # Then project to image plane
    points_img = project_to_image(points_cam, calib, camera_id)
    
    return points_img, points_cam

def filter_points_in_image(points_lidar, points_image, image_shape):
    """
    Filter LiDAR points that are within the image boundaries
    
    Args:
        points_lidar: Nx3 array of 3D points in LiDAR coordinate
        points_image: Nx2 array of 2D points in image
        image_shape: Tuple of (height, width) of the image
        
    Returns:
        tuple: (filtered LiDAR points, filtered image points)
    """
    height, width = image_shape[:2]
    
    # Create mask for points within image bounds
    mask = (points_image[:, 0] >= 0) & (points_image[:, 0] < width) & \
           (points_image[:, 1] >= 0) & (points_image[:, 1] < height)
    
    # Apply mask
    filtered_lidar = points_lidar[mask]
    filtered_image = points_image[mask]
    
    return filtered_lidar, filtered_image

def get_point_depth(points_lidar):
    """
    Calculate depth (distance from camera) for each LiDAR point
    
    Args:
        points_lidar: Nx3 array of 3D points in LiDAR coordinate
        
    Returns:
        np.array: Depth of each point
    """
    # In LiDAR coordinate, the depth is the x value
    return np.sqrt(np.sum(points_lidar[:, :3] ** 2, axis=1))

def get_colored_point_cloud_in_image(points_lidar, points_image, image):
    """
    Extract colors from image for each LiDAR point that projects onto the image
    
    Args:
        points_lidar: Nx3 array of 3D points in LiDAR coordinate
        points_image: Nx2 array of 2D points in image
        image: RGB image
        
    Returns:
        np.array: Nx6 array with [x, y, z, r, g, b] for each point
    """
    # Filter points that fall within image
    filtered_lidar, filtered_image = filter_points_in_image(points_lidar, points_image, image.shape)
    
    # Extract colors
    colors = np.zeros((filtered_lidar.shape[0], 3), dtype=np.uint8)
    for i, (u, v) in enumerate(filtered_image.astype(int)):
        colors[i] = image[v, u]
    
    # Combine position and color
    colored_points = np.hstack((filtered_lidar, colors))
    
    return colored_points

def create_depth_map(points_lidar, points_image, image_shape, max_depth=100):
    """
    Create a depth map image from projected LiDAR points
    
    Args:
        points_lidar: Nx3 array of 3D points in LiDAR coordinate
        points_image: Nx2 array of 2D points in image
        image_shape: Tuple of (height, width) of the image
        max_depth: Maximum depth value for normalization
        
    Returns:
        np.array: Depth map image
    """
    height, width = image_shape[:2]
    depth_map = np.zeros((height, width), dtype=np.float32)
    
    # Calculate depth for each point
    depths = get_point_depth(points_lidar)
    
    # Filter points within image
    filtered_depths = depths[
        (points_image[:, 0] >= 0) & (points_image[:, 0] < width) & 
        (points_image[:, 1] >= 0) & (points_image[:, 1] < height)
    ]
    filtered_image_points = points_image[
        (points_image[:, 0] >= 0) & (points_image[:, 0] < width) & 
        (points_image[:, 1] >= 0) & (points_image[:, 1] < height)
    ]
    
    # Assign depth values to depth map
    for i, (u, v) in enumerate(filtered_image_points.astype(int)):
        # Keep the closest point if multiple points project to the same pixel
        if depth_map[v, u] == 0 or filtered_depths[i] < depth_map[v, u]:
            depth_map[v, u] = filtered_depths[i]
    
    # Normalize for visualization
    depth_map = np.clip(depth_map, 0, max_depth)
    depth_map = depth_map / max_depth
    
    return depth_map