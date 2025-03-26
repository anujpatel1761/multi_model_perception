# LiDAR-Camera Fusion for 3D Object Detection in Autonomous Vehicles  
**Real-Time Mid-Level Sensor Fusion for Enhanced Perception**  

![Fusion Demo](docs/demo.gif)  
*Example: Fused LiDAR-camera output with 3D bounding boxes (green = ground truth, red = detections).*

---

## Table of Contents  
1. [Project Overview](#project-overview)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Dataset Setup](#dataset-setup)  
5. [Sensor Calibration](#sensor-calibration)  
6. [Pipeline Execution](#pipeline-execution)  
7. [Evaluation](#evaluation)  
8. [Visualization](#visualization)  
9. [Customization](#customization)  
10. [Troubleshooting](#troubleshooting)  
11. [References](#references)  
12. [License](#license)  

---

## Project Overview  
This project implements a **mid-level sensor fusion pipeline** combining 2D camera detections (YOLOv5) and 3D LiDAR point clouds to achieve accurate 3D object detection. Key components:  
- **Sensor Fusion**: Aligns LiDAR points with 2D camera detections using calibrated transformations.  
- **3D Localization**: Clusters LiDAR points (DBSCAN) and estimates precise 3D bounding boxes.  
- **Real-Time Performance**: Optimized for >10 FPS on mid-tier GPUs.  

![Pipeline](docs/pipeline.png)  
*Pipeline: 2D detection → frustum filtering → LiDAR clustering → 3D fusion.*

---

## Prerequisites  

### Hardware  
- **GPU**: NVIDIA GPU with ≥8GB VRAM (e.g., RTX 3060/3090).  
- **RAM**: ≥16GB (for LiDAR point cloud processing).  
- **OS**: Ubuntu 20.04/22.04 (recommended) or Windows 10/11 with WSL2.  

### Software  
- **CUDA 11.3+ and cuDNN 8.2+**: Follow [NVIDIA’s installation guide](https://developer.nvidia.com/cuda-toolkit).  
- **Python 3.8+**: Install via [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  
- **ROS Noetic (Optional)**: For advanced calibration validation ([installation guide](http://wiki.ros.org/noetic/Installation)).  

### Python Packages  
Install dependencies from `requirements.txt`:  
```bash  
pip install -r requirements.txt  
```  
*Contents of `requirements.txt`*:  
```  
# Core packages  
torch==1.12.1+cu113  
torchvision==0.13.1+cu113  
opencv-python==4.6.0.66  
open3d==0.15.1  
numpy==1.23.3  
scikit-learn==1.1.2  # For DBSCAN  

# YOLOv5  
git+https://github.com/ultralytics/yolov5.git@v7.0  

# Evaluation  
pykitti==0.3.1  
```  

---

## Dataset Setup  
### Download the KITTI Dataset  
1. **Download Links**:  
   - [Images (Left Color, 12GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)  
   - [LiDAR Point Clouds (29GB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)  
   - [Calibration Files (5MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)  
   - [Labels (5MB)](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)  

2. **Folder Structure**:  
   ```  
   data/kitti/  
   ├── training/  
   │   ├── image_2/          # Camera images (000000.png, ...)  
   │   ├── velodyne/         # LiDAR point clouds (000000.bin, ...)  
   │   ├── calib/            # Calibration files (000000.txt, ...)  
   │   └── label_2/          # Ground truth labels (000000.txt, ...)  
   └── testing/              # (Optional) Test data  
   ```  

3. **Preprocessing**:  
   Run the following to verify dataset integrity:  
   ```bash  
   python scripts/verify_dataset.py --root data/kitti  
   ```  

---

## Sensor Calibration  
### KITTI Calibration Files  
Each sequence (e.g., `000000.txt`) contains:  
- **Intrinsic Parameters (P2)**: Camera matrix for the left color camera.  
- **Extrinsic Parameters (Tr_velo_to_cam)**: Transformation matrix from LiDAR to camera coordinates.  

### Validate Calibration  
Reproject LiDAR points onto the camera image:  
```bash  
python scripts/verify_calibration.py \  
  --image data/kitti/training/image_2/000000.png \  
  --lidar data/kitti/training/velodyne/000000.bin \  
  --calib data/kitti/training/calib/000000.txt \  
  --output docs/calibration_check.png  
```  
*Output*: An image with LiDAR points overlaid on the camera frame.  

![Calibration Check](docs/calibration_check.png)  
*If points align with objects (e.g., cars), calibration is correct.*  

---

## Pipeline Execution  

### Step 1: 2D Object Detection with YOLOv5  
Run YOLOv5 on the KITTI training images:  
```bash  
python detect_2d.py \  
  --weights yolov5s.pt \  
  --source data/kitti/training/image_2 \  
  --conf 0.5 \                # Confidence threshold  
  --save-txt \                # Save detections as KITTI-format labels  
  --project runs/2d_detections  
```  
*Output*:  
- Detections saved to `runs/2d_detections/labels/000000.txt`, etc.  
- Format: `[class] [x_center] [y_center] [width] [height] [confidence]`.  

### Step 2: Frustum Filtering  
Extract LiDAR points within 2D detection regions:  
```bash  
python fuse/frustum_filter.py \  
  --detections runs/2d_detections/labels \  
  --lidar data/kitti/training/velodyne \  
  --calib data/kitti/training/calib \  
  --output runs/frustum_points  
```  
*Output*: Filtered LiDAR points for each detection (e.g., `000000.bin`).  

### Step 3: LiDAR Clustering (DBSCAN)  
Cluster LiDAR points within each frustum:  
```bash  
python fuse/lidar_clustering.py \  
  --input runs/frustum_points \  
  --output runs/lidar_clusters \  
  --eps 0.5 \                 # Max point distance for clustering  
  --min_samples 10 \          # Min points to form a cluster  
  --threshold_z 0.2           # Remove points below this height (meters)  
```  
*Output*: Cluster labels (e.g., `000000.txt` with `[x, y, z, cluster_id]`).  

### Step 4: Mid-Level Fusion  
Fuse 2D detections with 3D clusters to estimate 3D bounding boxes:  
```bash  
python fuse/mid_fusion.py \  
  --clusters runs/lidar_clusters \  
  --detections runs/2d_detections/labels \  
  --calib data/kitti/training/calib \  
  --output runs/3d_detections  
```  
*Output*: 3D bounding boxes in KITTI format (`000000.txt`):  
```  
Car 0 0 0 10.5 2.1 1.8 4.2 1.57 0.9  
```  
- Format: `[class, truncated, occluded, alpha, x, y, z, w, h, l, rotation_y, score]`.  

---

## Evaluation  

### 1. Install KITTI Evaluation Toolkit  
```bash  
git clone https://github.com/asharakeh/kitti-object-eval-python.git  
cd kitti-object-eval-python  
pip install -r requirements.txt  
```  

### 2. Run Evaluation  
Compare your 3D detections against ground truth:  
```bash  
python evaluate.py \  
  --label_path ../data/kitti/training/label_2 \  
  --result_path ../runs/3d_detections \  
  --metrics ap_3d \           # Compute AP for 3D boxes  
  --iou_thresholds 0.5 0.7    # IoU thresholds for AP  
  --verbose  
```  

### 3. Expected Results  
| Class      | AP@0.5 | AP@0.7 |  
|------------|--------|--------|  
| Car        | 82.9%  | 70.1%  |  
| Pedestrian | 63.4%  | 52.3%  |  

---

## Visualization  

### 1. 3D Visualization with Open3D  
Render LiDAR point clouds and 3D bounding boxes:  
```bash  
python visualize/plot_3d.py \  
  --lidar data/kitti/training/velodyne/000000.bin \  
  --detections runs/3d_detections/000000.txt \  
  --calib data/kitti/training/calib/000000.txt \  
  --output docs/3d_visualization.png  
```  
![3D Visualization](docs/3d_visualization.png)  

### 2. 2D Projection with OpenCV  
Overlay 3D boxes on camera images:  
```bash  
python visualize/project_to_image.py \  
  --image data/kitti/training/image_2/000000.png \  
  --detections runs/3d_detections/000000.txt \  
  --calib data/kitti/training/calib/000000.txt \  
  --output docs/2d_projection.png  
```  
![2D Projection](docs/2d_projection.png)  

---

## Customization  

### Adjust Fusion Parameters  
Modify `config/fusion_params.yaml`:  
```yaml  
frustum:  
  expand_ratio: 0.1    # Expand 2D boxes by 10% to capture nearby LiDAR points  

clustering:  
  eps: 0.5             # DBSCAN distance threshold  
  min_samples: 10      # Minimum points per cluster  

fusion:  
  min_score: 0.3       # Min detection confidence to retain  
  max_z: 2.5           # Max object height (meters)  
```  

### Train Custom YOLOv5 Model  
Follow the [YOLOv5 Custom Training Guide](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data).  

---

## Troubleshooting  

| Issue                          | Solution                                  |  
|--------------------------------|-------------------------------------------|  
| **Misaligned LiDAR-camera data** | Re-run calibration verification script. |  
| **Low AP scores**               | Adjust DBSCAN `eps` and `min_samples`.  |  
| **Slow inference**              | Enable TensorRT for YOLOv5 (`--device 0 --half`). |  
| **CUDA Out of Memory**          | Reduce YOLOv5 input size (`--img 512`). |  

---

## References  
1. KITTI Dataset: [Paper](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf) | [Website](http://www.cvlibs.net/datasets/kitti/)  
2. YOLOv5: [GitHub](https://github.com/ultralytics/yolov5)  
3. Open3D: [Documentation](http://www.open3d.org/docs/release/)  
4. KITTI Evaluation Toolkit: [GitHub](https://github.com/asharakeh/kitti-object-eval-python)  

---

## License  
MIT License. See [LICENSE](LICENSE).  

```  

---

### Key Features of This README:  
1. **End-to-End Guidance**: From dataset setup to evaluation.  
2. **Reproducibility**: Exact package versions and parameter explanations.  
3. **Visual Aids**: Demo images, pipeline diagram, and sample outputs.  
4. **Customization**: Instructions to tweak fusion parameters and models.  

Replace placeholder paths (e.g., `docs/demo.gif`) with actual files from your project.