import json, sys
import numpy as np
import cv2
from PIL import Image
import torch
from pathlib import Path
import open3d as o3d
from typing import Tuple, Dict
from .base_processor import BaseProcessor

# Path to the directory containing the 'llava' package
llava_parent_dir = r'/home/znasif/vision-depth-pro'  # Use raw string for paths

# Add this directory to sys.path if it's not already there
if llava_parent_dir not in sys.path:
    sys.path.insert(0, llava_parent_dir)

from depth_pro import create_model_and_transforms, load_rgb

class DepthProcessor(BaseProcessor):
    def __init__(self, 
                 checkpoint_uri="/home/znasif/vision-depth-pro/checkpoints/depth_pro.pt",
                 use_gpu=True,
                 voxel_size: float = 0.01,
                 estimate_normals: bool = True):
        """
        Initialize DepthPro processor
        
        Args:
            checkpoint_uri (str): Path to the DepthPro model checkpoint
            use_gpu (bool): Whether to use GPU acceleration
            voxel_size (float): Voxel size for point cloud downsampling
            estimate_normals (bool): Whether to estimate surface normals
        """
        super().__init__()
        self.checkpoint_uri = checkpoint_uri
        self.voxel_size = voxel_size
        self.estimate_normals = estimate_normals
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Load the model and transforms
        self._load_model()
        
    def _load_model(self):
        """
        Load the DepthPro model and transforms
        """
        self.model, self.transform = create_model_and_transforms(
            device=self.device,
            precision=torch.half
        )
        self.model.eval()
        print(f"DepthPro model loaded successfully on {self.device}")
    
    def preprocess_image(self, frame: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """
        Preprocess the image for DepthPro
        
        Args:
            frame (np.ndarray): Input OpenCV frame (BGR)
            
        Returns:
            Tuple[PIL.Image, np.ndarray]: Processed PIL image and RGB numpy array
        """
        # Convert from BGR to RGB
        if len(frame.shape) == 2:
            # Grayscale to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            # BGRA to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        return pil_image, frame_rgb
    
    def pointcloud_to_json(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """
        Convert Open3D point cloud to JSON format
        
        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud
            
        Returns:
            Dict: JSON-serializable dictionary containing point cloud data
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals) if pcd.has_normals() else None
        
        json_data = {
            'points': points.tolist(),
            'colors': colors.tolist() if colors is not None else [],
            'normals': normals.tolist() if normals is not None else []
        }
        return json_data
    
    def create_point_cloud(self, depth: np.ndarray, rgb_image: np.ndarray, focallength_px: torch.Tensor) -> o3d.geometry.PointCloud:
        """
        Create a point cloud from depth map, RGB image, and focal length
        
        Args:
            depth (np.ndarray): Depth map
            rgb_image (np.ndarray): RGB image
            focallength_px (torch.Tensor): Focal length in pixels
            
        Returns:
            o3d.geometry.PointCloud: Generated point cloud
        """
        height, width = depth.shape
        # Create intrinsic matrix (assuming principal point at image center)
        cx, cy = width / 2, height / 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width=width,
            height=height,
            fx=focallength_px,
            fy=focallength_px,
            cx=cx,
            cy=cy
        )
        
        # Convert focal length to NumPy scalar
        focallength_px = focallength_px.detach().cpu().numpy().item()
        
        # Create coordinate grid
        y, x = np.indices((height, width))
        z = depth
        # Avoid division by zero
        z[z == 0] = 1e-6
        
        # Convert to 3D coordinates
        X = (x - cx) * z / focallength_px
        Y = (y - cy) * z / focallength_px
        Z = z
        
        # Stack into point cloud
        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        
        # Normalize RGB colors to [0, 1]
        colors = rgb_image.reshape(-1, 3) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Downsample point cloud
        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        
        # Estimate normals if requested
        if self.estimate_normals:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
        
        return pcd
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process frame using DepthPro and return original frame and point cloud JSON
        
        Args:
            frame (np.ndarray): Input frame to process (BGR)
            
        Returns:
            Tuple[np.ndarray, Dict]: Original frame and point cloud JSON data
        """
        # Create output frame as copy of input
        output = frame.copy()
        
        # Preprocess frame
        pil_image, rgb_image = self.preprocess_image(frame)
        
        try:
            # Transform image for model
            image_tensor = self.transform(pil_image)
            
            # Run inference
            with torch.no_grad():
                prediction = self.model.infer(image_tensor)
            
            depth = prediction["depth"]
            focallength_px = prediction["focallength_px"]
            
            # Convert depth to numpy
            depth_np = depth.detach().cpu().numpy().squeeze()
            
            # Generate point cloud
            pcd = self.create_point_cloud(depth_np, rgb_image, focallength_px)
            
            # Convert point cloud to JSON
            pointcloud_json = self.pointcloud_to_json(pcd)
            
            return output, pointcloud_json
            
        except Exception as e:
            print(f"Processing error: {e}")
            return output, {'error': str(e), 'points': [], 'colors': [], 'normals': []}

processor = DepthProcessor()
app = processor.app