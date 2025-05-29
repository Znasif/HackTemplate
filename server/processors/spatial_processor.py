import os, sys
import torch
import numpy as np
from typing import Dict, Tuple, Optional # Ensure typing imports
# Assuming base_processor.py is in the same directory or a discoverable path
from .base_processor import BaseProcessor 

from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

# Path to the directory containing the 'spatiallm' package
llava_parent_dir = r'/home/znasif/SpatialLm' # Use raw string for paths, ensure this path is correct
if llava_parent_dir not in sys.path:
    sys.path.insert(0, llava_parent_dir)

try:
    from spatiallm import Layout, SpatialLMLlamaForCausalLM, SpatialLMQwenForCausalLM
    from spatiallm.pcd import get_points_and_colors, cleanup_pcd, Compose
except ImportError as e:
    print(f"Failed to import from spatiallm: {e}. Ensure SpatialLm is installed and path is correct.")
    raise

import open3d as o3d
from threading import Thread
import json # For error stringification if needed

class SpatialProcessor(BaseProcessor):
    def __init__(self,  
                 model_path: str = "manycore-research/SpatialLM-Llama-1B", # Ensure model exists
                 code_template_file: str = "code_template.txt", # Ensure this file exists and is accessible
                 top_k: int = 10,
                 top_p: float = 0.95,
                 temperature: float = 0.6,
                 num_beams: int = 1,
                 use_gpu: bool = True):
        super().__init__()
        self.model_path = model_path
        self.code_template_file = code_template_file
        if not os.path.exists(self.code_template_file):
             print(f"Warning: Code template file not found at {self.code_template_file}. Using default prompt.")
             # Fallback or error, for now, just a warning, generation might fail.
             self.code_template_for_prompt = "Layout(" # Minimal fallback
        else:
            with open(self.code_template_file, "r") as f:
                self.code_template_for_prompt = f.read()

        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.num_beams = num_beams
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        self._load_model()

    def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True) # Added trust_remote_code
        self.model.to(self.device)
        # Set point_backbone_dtype based on device to avoid potential issues
        pt_dtype = torch.float16 if self.device == 'cuda' else torch.float32
        self.model.set_point_backbone_dtype(pt_dtype)
        self.model.eval()
        print(f"SpatialLM model loaded successfully on {self.device}")

    def preprocess_point_cloud_for_model(self, points: np.ndarray, colors: np.ndarray, grid_size: float, num_bins: int) -> torch.Tensor:
        # Renamed from preprocess_point_cloud to avoid conflict with abstract method name
        transform = Compose(
            [
                dict(type="PositiveShift"),
                dict(type="NormalizeColor"), # Assumes colors are 0-255
                dict(
                    type="GridSample",
                    grid_size=grid_size,
                    hash_type="fnv",
                    mode="test",
                    keys=("coord", "color"),
                    return_grid_coord=True,
                    max_grid_coord=num_bins,
                ),
            ]
        )
        point_cloud_data = transform( # Corrected variable name
            {
                "name": "pcd",
                "coord": points.copy(), # GridSample might modify inplace
                "color": colors.copy(), # GridSample might modify inplace
            }
        )
        coord = point_cloud_data["grid_coord"]
        xyz = point_cloud_data["coord"]
        rgb = point_cloud_data["color"] # Colors should be normalized by NormalizeColor transform
        
        # Concatenate features: grid_coord (integer), xyz (float), rgb (float, normalized)
        processed_pcd_features = np.concatenate([coord, xyz, rgb], axis=1)
        return torch.as_tensor(np.stack([processed_pcd_features], axis=0)).to(self.device) # Add batch dim

    def pointcloud_to_json(self, pcd: o3d.geometry.PointCloud) -> Dict:
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else np.array([])
        normals = np.asarray(pcd.normals) if pcd.has_normals() else np.array([])
        return {
            'points': points.tolist(),
            'colors': colors.tolist(),
            'normals': normals.tolist()
        }

    def layout_to_json(self, layout: Layout) -> Dict:
        detections = []
        if layout and layout.elements: # Check if layout and elements exist
            for element in layout.elements:
                bbox = {
                    'type': element.type if hasattr(element, 'type') else 'unknown',
                    'center': element.center.tolist() if hasattr(element, 'center') and element.center is not None else [0,0,0],
                    'size': element.size.tolist() if hasattr(element, 'size') and element.size is not None else [1,1,1],
                    'orientation': element.orientation.tolist() if hasattr(element, 'orientation') and element.orientation is not None else [0,0,0,1], # Assuming quaternion
                }
                detections.append(bbox)
        
        return {
            'detections': detections,
            'timestamp': np.datetime64('now').astype(str) # Consider UTC: datetime.now(timezone.utc).isoformat()
        }

    def _internal_process_pcd_object(self, pcd_object: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, Dict]:
        try:
            pcd_object = cleanup_pcd(pcd_object) # cleanup_pcd should handle empty point clouds
            if not pcd_object.has_points():
                 return pcd_object, {"error": "Point cloud is empty after cleanup", "detections": []}

            points_np, colors_np = get_points_and_colors(pcd_object) # colors_np will be 0-1 if pcd.colors was 0-1
                                                                # if pcd.colors was 0-255, this needs adjustment or ensure NormalizeColor handles it.
                                                                # Assuming colors_np from get_points_and_colors are 0-255 for NormalizeColor transform

            min_extent = np.min(points_np, axis=0) if points_np.size > 0 else np.array([0,0,0])


            grid_size = Layout.get_grid_size() # Class method
            num_bins = Layout.get_num_bins()   # Class method
            
            input_pcd_tensor = self.preprocess_point_cloud_for_model(points_np, colors_np * 255, grid_size, num_bins) # Scale colors to 0-255 for NormalizeColor


            prompt = f"<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {self.code_template_for_prompt}"

            if self.model.config.model_type == SpatialLMLlamaForCausalLM.config_class.model_type:
                conversation = [{"role": "user", "content": prompt}]
            elif self.model.config.model_type == SpatialLMQwenForCausalLM.config_class.model_type:
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            else:
                raise ValueError(f"Unsupported model type: {self.model.config.model_type}")

            input_ids = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True # Increased timeout
            )

            generate_kwargs = dict(
                input_ids=input_ids, # Removed extra {}
                point_clouds=input_pcd_tensor,
                streamer=streamer,
                max_new_tokens=4096,
                do_sample=True,
                use_cache=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                num_beams=self.num_beams,
                pad_token_id=self.tokenizer.eos_token_id # Often needed for robust generation
            )
            
            generation_thread = Thread(target=self.model.generate, kwargs=generate_kwargs)
            generation_thread.start()

            generated_texts = [text for text in streamer]
            generation_thread.join(timeout=60.0) # Wait for thread with timeout
            if generation_thread.is_alive():
                print("Warning: SpatialLM generation thread timed out.")
                # Consider how to handle this; for now, it might result in partial text

            layout_str = "".join(generated_texts).strip()
            if not layout_str:
                print("Warning: SpatialLM generated an empty layout string.")
                return pcd_object, {'error': 'Empty layout generated by SpatialLM', 'detections': []}
            
            try:
                layout = Layout(layout_str) # This can fail if layout_str is malformed
                layout.undiscretize_and_unnormalize()
                layout.translate(min_extent)
                layout_json_output = self.layout_to_json(layout)
            except Exception as layout_exc:
                print(f"Error parsing SpatialLM layout: {layout_exc}. Generated string: '{layout_str}'")
                return pcd_object, {'error': f'Failed to parse SpatialLM layout: {str(layout_exc)}', 'detections': []}

            return pcd_object, layout_json_output

        except Exception as e:
            import traceback
            print(f"SpatialProcessor _internal_process_pcd_object error: {e}\n{traceback.format_exc()}")
            return pcd_object, {'error': f'SpatialLM processing error: {str(e)}', 'detections': []}

    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Dict]:
        try:
            points = np.array(point_cloud_data.get('points', []))
            colors_input = np.array(point_cloud_data.get('colors', [])) # Expected to be 0-1 from DepthProc

            if points.size == 0:
                empty_pcd_json = {'points': [], 'colors': [], 'normals': []}
                return empty_pcd_json, {"error": "Empty point cloud received by SpatialProcessor", "detections": []}

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors_input.size > 0 and colors_input.shape[0] == points.shape[0]:
                # Ensure colors are float and in [0,1] for o3d.geometry.PointCloud
                colors_float = colors_input.astype(np.float64)
                if colors_float.max() > 1.0: # If accidentally 0-255
                     colors_float /= 255.0
                pcd.colors = o3d.utility.Vector3dVector(np.clip(colors_float,0,1))


            processed_pcd_object, layout_json = self._internal_process_pcd_object(pcd)
            
            # Convert the (potentially modified by cleanup_pcd) pcd_object back to JSON
            processed_pcd_json_output = self.pointcloud_to_json(processed_pcd_object)
            
            return processed_pcd_json_output, layout_json

        except Exception as e:
            import traceback
            error_detail = f"SpatialProcessor process_pointcloud error: {e}\n{traceback.format_exc()}"
            print(error_detail)
            # Fallback: return a minimal representation of an empty point cloud and the error
            fallback_pcd_json = {'points': [], 'colors': [], 'normals': []}
            if point_cloud_data and isinstance(point_cloud_data, dict): # Try to preserve keys if possible
                fallback_pcd_json = {k: point_cloud_data.get(k,[]) for k in ['points', 'colors', 'normals']}

            return fallback_pcd_json, {'error': str(e), 'detections': []}

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        # SpatialProcessor expects point cloud data. Return original frame and info.
        return frame, "SpatialProcessor expects point cloud data. Please use a DepthProcessor or provide point cloud directly."

# To run this processor standalone (example):
processor = SpatialProcessor()
app = processor.app