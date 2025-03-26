import matplotlib.pyplot as plt
import sys
import os

# Add 'src' directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.kitti_loader import KITTILoader
from utils.projection_utils import project_lidar_to_image
import matplotlib.pyplot as plt

# Load data
dataset = KITTILoader("C:/Users/anujp/Desktop/multi_model_perception/data")
idx = "000405"
image = dataset.get_image(idx)
lidar = dataset.get_lidar(idx)
calib = dataset.get_calib(idx)

# Project LiDAR points into image plane
projected_pts, _ = project_lidar_to_image(lidar, calib, image.shape)

# Plot
plt.figure(figsize=(12, 6))
plt.imshow(image)
plt.scatter(projected_pts[:, 0], projected_pts[:, 1], s=0.5, c='red')  # Small red dots
plt.title(f"Projected LiDAR Points on Image ({idx})")
plt.axis('off')
plt.show()
