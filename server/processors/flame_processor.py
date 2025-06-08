import torch
import torchvision.transforms as TTR
import numpy as np
import cv2
import math
import json
from PIL import Image
import open3d as o3d
from typing import Tuple, Dict, Optional, Union
from .base_processor import BaseProcessor
from depth_pro import create_model_and_transforms

class FlameProcessor(BaseProcessor):
    def __init__(self, 
                 jit_path: str = None, 
                 taxonomy_path: str = None,
                 depth_checkpoint_uri: str = None,
                 use_gpu: bool = True,
                 voxel_size: float = 0.01,
                 estimate_normals: bool = True):
        """
        Initialize the FlameProcessor with material segmentation and depth estimation.

        Args:
            jit_path: Path to the pre-trained material segmentation model (e.g., DMS46_v1.pt).
            taxonomy_path: Path to the taxonomy JSON file for material names and colors.
            depth_checkpoint_uri: Path to the depth estimation model checkpoint.
            use_gpu: Whether to use GPU if available.
            voxel_size: Voxel size for point cloud downsampling.
            estimate_normals: Whether to estimate normals for the point cloud.
        """
        super().__init__()

        # Material Segmentation Initialization
        if jit_path is None:
            jit_path = '/home/znasif/vidServer/server/models/DMS46_v1.pt'
        if taxonomy_path is None:
            taxonomy_path = '/home/znasif/vidServer/server/models/taxonomy.json'

        print(f"DEBUG: Loading material model from: {jit_path}")
        print(f"DEBUG: Loading taxonomy from: {taxonomy_path}")

        # Load taxonomy
        try:
            with open(taxonomy_path, 'rb') as f:
                t = json.load(f)
            print(f"DEBUG: Taxonomy loaded successfully. Keys: {list(t.keys())}")
        except Exception as e:
            print(f"ERROR: Failed to load taxonomy: {e}")
            raise

        # Define DMS46 indices
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        print(f"DEBUG: DMS46 indices: {len(dms46)} materials")

        # Material names
        self.material_names = [t['names'][i] for i in dms46]
        print(f"DEBUG: Material names: {self.material_names}")

        # Original colormap
        original_colormap = np.array([t['srgb_colormap'][i] for i in dms46], dtype=np.uint8)
        print(f"DEBUG: Original colormap shape: {original_colormap.shape}")

        # Flammability mapping
        self.flammability_map = {
            1: 'A1', 2: 'B/C', 3: 'A1', 4: 'D/E', 5: 'D/E', 6: 'B/C', 7: 'A1', 8: 'B/C',
            9: 'D/E', 10: 'A1', 11: 'B/C', 12: 'A1', 13: 'D/E', 15: 'F', 16: 'D/E', 17: 'D/E',
            18: 'D/E', 19: 'D/E', 20: 'A2', 21: 'D/E', 23: 'D/E', 24: 'D/E', 26: 'A1',
            27: 'A2', 29: 'B/C', 30: 'D/E', 32: 'D/E', 33: 'D/E', 34: 'D/E', 35: 'D/E',
            36: 'D/E', 37: 'D/E', 38: 'D/E', 39: 'D/E', 41: 'D/E', 43: 'A1', 44: 'A1',
            46: 'A1', 47: 'B/C', 48: 'D/E', 49: 'D/E', 50: 'B/C', 51: 'D/E', 52: 'D/E',
            53: 'D/E', 56: 'D/E'
        }
        print(f"DEBUG: Flammability mapping created for {len(self.flammability_map)} materials")

        # Flammability class colors
        try:
            self.class_colors = {
                'A1': original_colormap[dms46.index(26)],  # Yellow [225, 225, 0]
                'A2': original_colormap[dms46.index(20)],  # Cyan [0, 137, 188]
                'B/C': original_colormap[dms46.index(29)],  # Pink [225, 137, 188]
                'D/E': original_colormap[dms46.index(52)],  # Dark red [137, 0, 0]
                'F': original_colormap[dms46.index(15)]    # Light red [225, 188, 188]
            }
            print(f"DEBUG: Class colors: {self.class_colors}")
        except Exception as e:
            print(f"ERROR: Failed to create class colors: {e}")
            raise

        # SRGB colormap for flammability classes
        self.srgb_colormap = np.array([self.class_colors[self.flammability_map[i]] for i in dms46], dtype=np.uint8)
        print(f"DEBUG: SRGB colormap shape: {self.srgb_colormap.shape}")

        # Load material segmentation model
        self.device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
        print(f"DEBUG: Using device: {self.device}")
        try:
            self.material_model = torch.jit.load(jit_path, map_location=self.device)
            self.material_model.eval()
            print("DEBUG: Material model loaded successfully")
        except Exception as e:
            print(f"ERROR: Failed to load material model: {e}")
            raise

        # Depth Estimation Initialization
        if depth_checkpoint_uri is None:
            depth_checkpoint_uri = '/home/znasif/vision-depth-pro/checkpoints/depth_pro.pt'
        self.depth_checkpoint_uri = depth_checkpoint_uri
        self.voxel_size = voxel_size
        self.estimate_normals = estimate_normals
        self._load_depth_model()

    def _load_depth_model(self):
        """Load the depth estimation model."""
        self.depth_model, self.depth_transform = create_model_and_transforms(
            device=self.device,
            precision=torch.half if self.device == 'cuda' else torch.float32
        )
        self.depth_model.eval()
        print(f"DEBUG: Depth model loaded successfully on {self.device}")

    def preprocess_image(self, frame: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """Preprocess the input frame for depth estimation."""
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported number of channels: {frame.shape[2]}")
        
        pil_image = Image.fromarray(frame_rgb)
        return pil_image, frame_rgb

    def pointcloud_to_json(self, pcd: o3d.geometry.PointCloud) -> Dict:
        """Convert an Open3D point cloud to a JSON-serializable dictionary."""
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.array([])
        normals = np.asarray(pcd.normals) if pcd.has_normals() else np.array([])
        
        return {
            'points': points.tolist(),
            'colors': colors.tolist(),
            'normals': normals.tolist()
        }

    def create_point_cloud(self, depth: np.ndarray, vis_bgr: np.ndarray, focallength_px_tensor: torch.Tensor) -> o3d.geometry.PointCloud:
        """Create a point cloud using depth map and flammability-based colors from vis_bgr."""
        height, width = depth.shape
        cx, cy = width / 2.0, height / 2.0
        focallength_px = focallength_px_tensor.item()

        y_indices, x_indices = np.indices((height, width), dtype=np.float32)
        z_values = depth.astype(np.float32)
        z_values[z_values < 1e-6] = 1e-6  # Prevent division by zero

        X = (x_indices - cx) * z_values / focallength_px
        Y = (y_indices - cy) * z_values / focallength_px
        Z = z_values

        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        
        # Convert vis_bgr (BGR) to RGB for point cloud colors
        vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
        colors_normalized = vis_rgb.reshape(-1, 3) / 255.0  # Normalize to [0, 1]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)

        if self.voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        if self.estimate_normals and pcd.has_points():
            if len(pcd.points) >= 30:
                pcd.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=30)
                )
            else:
                print("DEBUG: Not enough points to estimate normals after downsampling.")

        return pcd

    def apply_color(self, label_mask: np.ndarray) -> np.ndarray:
        """Apply the flammability-based color map to the label mask."""
        print(f"DEBUG: apply_color - label_mask shape: {label_mask.shape}")
        print(f"DEBUG: apply_color - label_mask unique values: {np.unique(label_mask)}")
        vis = np.take(self.srgb_colormap, label_mask, axis=0)  # Map labels to RGB
        return vis[..., ::-1]  # Convert RGB to BGR

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Process the input frame with material segmentation and depth estimation.

        Args:
            frame: Input image as a numpy array in BGR format.

        Returns:
            Tuple containing:
            - processed_frame: Concatenated original and segmented image in BGR.
            - result_dict: Dictionary with flammability classes and point cloud data.
        """
        print(f"DEBUG: process_frame - Input frame shape: {frame.shape}")
        h, w = frame.shape[:2]
        print(f"DEBUG: Original dimensions: {h}x{w}")

        # Resize frame (smallest side to 512, maintain aspect ratio)
        scale = 512 / max(h, w)
        new_h = math.ceil(scale * h)
        new_w = math.ceil(scale * w)
        print(f"DEBUG: Resize scale: {scale}, new dimensions: {new_h}x{new_w}")

        resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        print(f"DEBUG: Resized frame shape: {resized_frame.shape}")

        # Material Segmentation
        img_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(img_rgb.transpose((2, 0, 1))).float()
        
        value_scale = 255
        mean = [0.485, 0.456, 0.406]
        mean = [item * value_scale for item in mean]
        std = [0.229, 0.224, 0.225]
        std = [item * value_scale for item in std]
        normalize = TTR.Normalize(mean, std)
        image = normalize(image)

        image = image.unsqueeze(0).to(self.device)
        print(f"DEBUG: Material input tensor shape: {image.shape}")

        try:
            with torch.no_grad():
                model_output = self.material_model(image)
                prediction = model_output[0].detach().cpu().numpy()[0, 0].astype(np.int64)
                print(f"DEBUG: Material prediction shape: {prediction.shape}")
        except Exception as e:
            print(f"ERROR: Material inference failed: {e}")
            prediction = np.zeros((new_h, new_w), dtype=np.int64)

        vis_bgr = self.apply_color(prediction)
        processed_frame = np.concatenate((resized_frame, vis_bgr), axis=1)

        # Detect flammability classes
        unique_labels = np.unique(prediction)
        dms46 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21,
                 23, 24, 26, 27, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 43, 44,
                 46, 47, 48, 49, 50, 51, 52, 53, 56]
        
        detected_classes = []
        for label in unique_labels:
            if 0 <= label < len(dms46):
                material_id = dms46[label]
                if material_id in self.flammability_map:
                    detected_classes.append(self.flammability_map[material_id])
        detected_classes = sorted(set(detected_classes))
        result_text = "Detected flammability classes: " + ", ".join(detected_classes) if detected_classes else "No materials detected"
        print(result_text)
        # Depth Estimation
        pil_image, _ = self.preprocess_image(resized_frame)
        try:
            image_tensor = self.depth_transform(pil_image).to(self.device)
            if image_tensor.dtype == torch.float16 and self.device == 'cpu':
                image_tensor = image_tensor.float()

            with torch.no_grad():
                prediction_depth = self.depth_model.infer(image_tensor.unsqueeze(0))
            
            depth_np = prediction_depth["depth"].squeeze().detach().cpu().numpy()
            focallength_px = prediction_depth["focallength_px"].squeeze()

            pcd = self.create_point_cloud(depth_np, vis_bgr, focallength_px)
            pointcloud_json = self.pointcloud_to_json(pcd)
            
        except Exception as e:
            print(f"ERROR: Depth processing failed: {e}")
            pointcloud_json = {'error': str(e), 'points': [], 'colors': [], 'normals': []}

        return processed_frame, pointcloud_json
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        # This processor generates point clouds, doesn't typically process existing ones.
        # Return the input point cloud and an informative message, or raise error.
        # raise NotImplementedError("DepthProcessor does not process existing point clouds.")
        return point_cloud_data, {"message": "DepthProcessor received point cloud data but does not process it further."}


processor = FlameProcessor()
app = processor.app