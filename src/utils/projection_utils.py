import numpy as np


def cartesian_to_homogeneous(points):
    """
    Convert (N, 3) Cartesian coordinates to (N, 4) homogeneous by adding 1.

    Args:
        points (np.ndarray): shape (N, 3)

    Returns:
        np.ndarray: shape (N, 4)
    """
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    return np.hstack((points, ones))


def project_lidar_to_image(points_lidar, calib, image_shape):
    """
    Project LiDAR points onto the image plane using KITTI calibration.

    Args:
        points_lidar (np.ndarray): LiDAR points, shape (N, 4) — [x, y, z, reflectance]
        calib (dict): Calibration dictionary with keys 'Tr_velo_to_cam', 'R0_rect', 'P2'
        image_shape (tuple): Image dimensions (H, W, C)

    Returns:
        projected_points (np.ndarray): Valid projected points in 2D image space (u, v)
        valid_indices (np.ndarray): Indices of LiDAR points that were projected onto the image
    """
    # Unpack calibration
    Tr = calib['Tr_velo_to_cam']  # (3, 4)
    R0 = calib['R0_rect']         # (3, 3)
    P2 = calib['P2']              # (3, 4)

    # (1) Convert to homogeneous and transform to camera coordinates
    pts_lidar_hom = cartesian_to_homogeneous(points_lidar[:, :3])  # (N, 4)
    pts_cam = (Tr @ pts_lidar_hom.T).T                             # (N, 3)

    # (2) Rectify
    pts_cam_rect = (R0 @ pts_cam.T).T  # (N, 3)

    # Filter out points behind the camera (z <= 0)
    in_front = pts_cam_rect[:, 2] > 0
    pts_cam_rect = pts_cam_rect[in_front]

    # (3) Project to image plane
    pts_cam_rect_hom = cartesian_to_homogeneous(pts_cam_rect)  # (N, 4)
    pts_2d_hom = (P2 @ pts_cam_rect_hom.T).T  # (N, 3)

    # Normalize to get pixel coordinates
    pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2, np.newaxis]  # (u, v)

    # Filter points that are within image bounds
    H, W = image_shape[:2]
    valid_mask = (
        (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < W) &
        (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < H)
    )

    return pts_2d[valid_mask], in_front.nonzero()[0][valid_mask]
