import json, sys
import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
import open3d as o3d
from typing import Tuple, Dict, Optional, Union # Ensure typing imports

# Assuming base_processor.py is in the same directory or a discoverable path
from .base_processor import BaseProcessor, ProcessRequest # ProcessRequest might not be needed here directly

# Path to the directory containing the 'depth_pro' package (if not installed)
# depth_pro_parent_dir = r'/path/to/parent/of/depth_pro'
# if depth_pro_parent_dir not in sys.path:
# sys.path.insert(0, depth_pro_parent_dir)
from depth_pro import create_model_and_transforms #, load_rgb # load_rgb might not be needed if image comes as np.ndarray

class DepthProcessor(BaseProcessor):
    def __init__(self,  
                 checkpoint_uri="/home/znasif/vision-depth-pro/checkpoints/depth_pro.pt", # Ensure path is correct
                 use_gpu=True,
                 voxel_size: float = 0.01,
                 estimate_normals: bool = True):
        super().__init__()
        self.checkpoint_uri = checkpoint_uri
        self.voxel_size = voxel_size
        self.estimate_normals = estimate_normals
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self._load_model()
        
    def _load_model(self):
        self.model, self.transform = create_model_and_transforms(
            device=self.device,
            precision=torch.half if self.device == 'cuda' else torch.float32 # half precision only for cuda
        )
        self.model.eval()
        print(f"DepthPro model loaded successfully on {self.device}")
    
    def preprocess_image(self, frame: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4: # BGRA or RGBA
            # Check actual order if issues arise, OpenCV imdecode usually gives BGR/BGRA
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB) 
        elif frame.shape[2] == 3: # BGR or RGB
             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Assuming BGR from cv2.imdecode
        else:
            raise ValueError(f"Unsupported number of channels: {frame.shape[2]}")
        
        pil_image = Image.fromarray(frame_rgb)
        return pil_image, frame_rgb
    
    def pointcloud_to_json(self, pcd: o3d.geometry.PointCloud) -> Dict:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.array([])
        normals = np.asarray(pcd.normals) if pcd.has_normals() else np.array([])
        
        return {
            'points': points.tolist(),
            'colors': colors.tolist(), # Ensure colors are in [0,1] range if applicable later
            'normals': normals.tolist()
        }
    
    def create_point_cloud(self, depth: np.ndarray, rgb_image: np.ndarray, focallength_px_tensor: torch.Tensor) -> o3d.geometry.PointCloud:
        height, width = depth.shape
        cx, cy = width / 2.0, height / 2.0
        
        focallength_px = focallength_px_tensor.item() # Convert tensor to scalar

        # Create Open3D intrinsic object (optional, for compatibility if other o3d functions use it)
        # intrinsic = o3d.camera.PinholeCameraIntrinsic(width, height, focallength_px, focallength_px, cx, cy)

        y_indices, x_indices = np.indices((height, width), dtype=np.float32)
        
        # Prevent division by zero or very small depth values
        z_values = depth.astype(np.float32)
        z_values[z_values < 1e-6] = 1e-6 # Or handle as invalid points

        X = (x_indices - cx) * z_values / focallength_px
        Y = (y_indices - cy) * z_values / focallength_px
        Z = z_values
        
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        colors_normalized = rgb_image.reshape(-1, 3) / 255.0 # Normalize colors to [0, 1]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
        
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        if self.estimate_normals and pcd.has_points(): # Check if points exist before estimating normals
            if len(pcd.points) >= 30 : # KDTreeSearchParamHybrid needs sufficient points
                 pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=30) # Adjusted radius
                )
            else:
                print("Not enough points to estimate normals after downsampling.")
        return pcd
        
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        output_image_for_client = frame.copy() # Original frame to be returned as visual
        
        pil_image, rgb_image_np = self.preprocess_image(frame)
        
        try:
            image_tensor = self.transform(pil_image).to(self.device)
            if image_tensor.dtype == torch.float16 and self.device == 'cpu': # half not supported on CPU
                image_tensor = image_tensor.float()

            with torch.no_grad():
                prediction = self.model.infer(image_tensor.unsqueeze(0)) # Add batch dimension
            
            depth_tensor = prediction["depth"].squeeze() # Remove batch dim
            focallength_px_tensor = prediction["focallength_px"].squeeze()
            
            depth_np = depth_tensor.detach().cpu().numpy()
            
            pcd = self.create_point_cloud(depth_np, rgb_image_np, focallength_px_tensor)
            pointcloud_json = self.pointcloud_to_json(pcd)
            
            return output_image_for_client, pointcloud_json # Result is the point cloud
            
        except Exception as e:
            import traceback
            print(f"DepthProcessor processing error: {e}\n{traceback.format_exc()}")
            return output_image_for_client, {'error': str(e), 'points': [], 'colors': [], 'normals': []}

    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        # This processor generates point clouds, doesn't typically process existing ones.
        # Return the input point cloud and an informative message, or raise error.
        # raise NotImplementedError("DepthProcessor does not process existing point clouds.")
        return point_cloud_data, {"message": "DepthProcessor received point cloud data but does not process it further."}

# To run this processor standalone (example):
processor = DepthProcessor()
app = processor.app