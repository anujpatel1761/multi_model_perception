import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import time
import argparse
from tqdm import tqdm

from src.dataset import get_kitti_dataset
from src.calibration_utils import lidar_to_image
from src.feature_extraction import create_fusion_module
from src.fusion import FeatureFusion, GridFusion

# Function to download YOLOv5 if not present
def download_yolov5(save_dir='yolov5'):
    if not os.path.exists(save_dir):
        print("Downloading YOLOv5...")
        os.system(f"git clone https://github.com/ultralytics/yolov5 {save_dir}")
        # Install requirements
        os.system(f"pip install -r {save_dir}/requirements.txt")
    return save_dir

def process_frame(frame_id, dataset, fusion_module, fusion_method, yolo_model=None, visualize=True):
    """
    Process a single frame through the entire pipeline
    
    Args:
        frame_id: Frame ID to process
        dataset: KITTI dataset instance
        fusion_module: Feature extraction module
        fusion_method: Fusion method to use
        yolo_model: Pre-trained YOLOv5 model
        visualize: Whether to create visualizations
        
    Returns:
        Dictionary with results
    """
    # Load data
    image = dataset.load_image(frame_id)
    lidar_points = dataset.load_lidar(frame_id)
    calib = dataset.load_calibration(frame_id)
    
    # Extract features
    features = fusion_module.fuse_features(image, lidar_points, calib)
    camera_features = features['camera_features']
    lidar_features = features['lidar_features']
    
    # Create fusion module
    fusion = FeatureFusion(fusion_method=fusion_method, use_gpu=torch.cuda.is_available())
    
    # Apply fusion
    fused_features = fusion.fuse_features(camera_features, lidar_features)
    
    # Run YOLO object detection on the original image
    detections = None
    if yolo_model is not None:
        # Convert image to format expected by YOLO
        img_for_detection = image.copy()  # RGB format
        
        # Run detection
        results = yolo_model(img_for_detection)
        
        # Get detections
        detections = results.pandas().xyxy[0]  # Get detection results as pandas DataFrame
    
    # Create visualization if requested
    if visualize:
        # Create visualization of features and fusion
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")
        
        # Camera features
        cam_feat_vis = np.mean(camera_features, axis=2)
        im1 = axes[0, 1].imshow(cam_feat_vis, cmap='viridis')
        axes[0, 1].set_title(f"Camera Features")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # LiDAR features
        lidar_feat_vis = np.mean(lidar_features, axis=2)
        im2 = axes[1, 0].imshow(lidar_feat_vis, cmap='plasma')
        axes[1, 0].set_title(f"LiDAR Features")
        axes[1, 0].axis("off")
        plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
        
        # Fused features
        fused_feat_vis = np.mean(fused_features, axis=2)
        im3 = axes[1, 1].imshow(fused_feat_vis, cmap='jet')
        axes[1, 1].set_title(f"Fused Features - {fusion_method}")
        axes[1, 1].axis("off")
        plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(f"features_{frame_id}.png")
        
        # If we have detections, show them on a separate figure
        if detections is not None and len(detections) > 0:
            # Create a copy of the image for drawing
            det_img = image.copy()
            
            plt.figure(figsize=(12, 8))
            
            # Draw bounding boxes
            for i, det in detections.iterrows():
                x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                conf = det['confidence']
                cls = det['name']
                
                # Choose color based on class
                if cls == 'person':
                    color = (0, 255, 0)  # Green for pedestrians
                elif cls == 'car' or cls == 'truck':
                    color = (255, 0, 0)  # Red for vehicles
                elif cls == 'bicycle':
                    color = (0, 0, 255)  # Blue for cyclists
                else:
                    color = (255, 255, 0)  # Yellow for others
                
                # Draw bounding box
                cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{cls}: {conf:.2f}"
                cv2.putText(det_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Convert BGR to RGB for matplotlib
            plt.imshow(det_img)
            plt.title("YOLO Object Detections")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(f"detections_{frame_id}.png")
            
        plt.show()
    
    return {
        'frame_id': frame_id,
        'camera_features': camera_features,
        'lidar_features': lidar_features,
        'fused_features': fused_features,
        'detections': detections
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Run LiDAR-Camera fusion pipeline with YOLOv5')
    parser.add_argument('--data_path', type=str, default='data', help='Path to KITTI dataset')
    parser.add_argument('--fusion_method', type=str, default='concat', 
                        choices=['concat', 'sum', 'attention', 'learned'],
                        help='Fusion method to use')
    parser.add_argument('--frame_id', type=str, default=None, help='Specific frame ID to process')
    parser.add_argument('--batch_mode', action='store_true', help='Process multiple frames')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of frames to process in batch mode')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--detect', action='store_true', help='Run object detection with YOLOv5')
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--yolo_model', type=str, default='yolov5s', 
                        choices=['yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x'],
                        help='YOLOv5 model to use')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load dataset
    print("Loading dataset...")
    dataset = get_kitti_dataset(args.data_path)
    
    # Create feature fusion module
    print("Creating feature extractors...")
    fusion_module = create_fusion_module(camera_model='resnet18', voxel_size=(0.2, 0.2, 0.2))
    
    # Load YOLOv5 if requested
    yolo_model = None
    if args.detect:
        print(f"Loading YOLOv5 {args.yolo_model} model...")
        # Make sure YOLOv5 is available
        yolo_dir = download_yolov5()
        
        # Need to add YOLOv5 directory to Python path
        import sys
        sys.path.append(yolo_dir)
        
        # Now import YOLOv5 modules
        from models.common import DetectMultiBackend
        from utils.general import non_max_suppression
        from utils.torch_utils import select_device
        
        # Load YOLOv5 model
        yolo_model = torch.hub.load('ultralytics/yolov5', args.yolo_model, pretrained=True)
        
        # Set model parameters
        yolo_model.conf = 0.25  # Confidence threshold
        yolo_model.iou = 0.45   # IoU threshold
        yolo_model.classes = None  # All classes
        
    # Process frames
    if args.frame_id is not None:
        # Process a specific frame
        print(f"Processing frame {args.frame_id}...")
        result = process_frame(args.frame_id, dataset, fusion_module, args.fusion_method, 
                               yolo_model, args.visualize)
        
        # Save result (excluding large tensors to save space)
        result_slim = {
            'frame_id': result['frame_id'],
            'detections': result['detections']
        }
        np.save(os.path.join(args.output_dir, f"frame_{args.frame_id}_result.npy"), result_slim)
        
    elif args.batch_mode:
        # Process multiple frames
        print(f"Processing {args.num_frames} frames in batch mode...")
        results = []
        
        # Get frame IDs to process
        frame_ids = dataset.frame_ids[:args.num_frames]
        
        # Process each frame
        for frame_id in tqdm(frame_ids):
            result = process_frame(frame_id, dataset, fusion_module, args.fusion_method, 
                                  yolo_model, visualize=False)
            
            # Store only detection results to save memory
            results.append({
                'frame_id': result['frame_id'],
                'detections': result['detections']
            })
            
        # Save batch results
        np.save(os.path.join(args.output_dir, f"batch_results_{args.num_frames}frames.npy"), results)
        
        # Create summary visualization for detections
        if args.visualize and yolo_model is not None:
            print("Creating detection summary...")
            
            # Count detections by class
            class_counts = {}
            
            for result in results:
                if result['detections'] is not None and len(result['detections']) > 0:
                    for _, det in result['detections'].iterrows():
                        cls = det['name']
                        if cls not in class_counts:
                            class_counts[cls] = 0
                        class_counts[cls] += 1
            
            # Plot detection counts
            plt.figure(figsize=(12, 6))
            classes = list(class_counts.keys())
            counts = [class_counts[cls] for cls in classes]
            
            plt.bar(classes, counts)
            plt.title(f"Object Detections Across {args.num_frames} Frames")
            plt.ylabel("Number of Detections")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"detection_summary_{args.num_frames}frames.png"))
            plt.show()
    
    else:
        # Process a single random frame
        frame_id = dataset.frame_ids[0]
        print(f"Processing frame {frame_id}...")
        result = process_frame(frame_id, dataset, fusion_module, args.fusion_method, 
                              yolo_model, args.visualize)
        
        # Save result (excluding large tensors to save space)
        result_slim = {
            'frame_id': result['frame_id'],
            'detections': result['detections']
        }
        np.save(os.path.join(args.output_dir, f"frame_{frame_id}_result.npy"), result_slim)
    
    print("Processing complete!")

if __name__ == "__main__":
    # Print CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is not available. Using CPU.")
    
    # Run main function
    main()