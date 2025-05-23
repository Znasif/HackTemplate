import os, sys
import torch
import numpy as np
from typing import Dict, Tuple
from .base_processor import BaseProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

# Path to the directory containing the 'llava' package
llava_parent_dir = r'/home/znasif/SpatialLm' # Use raw string for paths

# Add this directory to sys.path if it's not already there
if llava_parent_dir not in sys.path:
    sys.path.insert(0, llava_parent_dir)

from spatiallm import Layout, SpatialLMLlamaForCausalLM, SpatialLMQwenForCausalLM
from spatiallm.pcd import get_points_and_colors, cleanup_pcd, Compose
import open3d as o3d
from threading import Thread

class SpatialProcessor(BaseProcessor):
    def __init__(self, 
                 model_path: str = "manycore-research/SpatialLM-Llama-1B",
                 code_template_file: str = "code_template.txt",
                 top_k: int = 10,
                 top_p: float = 0.95,
                 temperature: float = 0.6,
                 num_beams: int = 1,
                 use_gpu: bool = True):
        """
        Initialize SpatialProcessor with SpatialLM model for 3D bounding box detection.

        Args:
            model_path (str): Path to the SpatialLM model checkpoint.
            code_template_file (str): Path to the code template file for prompting.
            top_k (int): Number of highest probability tokens for top-k filtering.
            top_p (float): Cumulative probability for top-p filtering.
            temperature (float): Temperature for token sampling.
            num_beams (int): Number of beams for beam search.
            use_gpu (bool): Whether to use GPU acceleration.
        """
        super().__init__()
        self.model_path = model_path
        self.code_template_file = code_template_file
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.num_beams = num_beams
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Load the model and tokenizer
        self._load_model()

    def _load_model(self):
        """
        Load the SpatialLM model and tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.set_point_backbone_dtype(torch.float32)
        self.model.eval()
        print(f"SpatialLM model loaded successfully on {self.device}")

    def preprocess_point_cloud(self, points: np.ndarray, colors: np.ndarray, grid_size: float, num_bins: int) -> torch.Tensor:
        """
        Preprocess point cloud for SpatialLM model.

        Args:
            points (np.ndarray): Point cloud coordinates (N, 3).
            colors (np.ndarray): Point cloud colors (N, 3).
            grid_size (float): Grid size for sampling.
            num_bins (int): Number of bins for grid sampling.

        Returns:
            torch.Tensor: Preprocessed point cloud tensor.
        """
        transform = Compose(
            [
                dict(type="PositiveShift"),
                dict(type="NormalizeColor"),
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
        point_cloud = transform(
            {
                "name": "pcd",
                "coord": points.copy(),
                "color": colors.copy(),
            }
        )
        coord = point_cloud["grid_coord"]
        xyz = point_cloud["coord"]
        rgb = point_cloud["color"]
        point_cloud = np.concatenate([coord, xyz, rgb], axis=1)
        return torch.as_tensor(np.stack([point_cloud], axis=0)).to(self.device)

    def layout_to_json(self, layout: Layout) -> Dict:
        """
        Convert SpatialLM Layout object to JSON format with 3D bounding boxes.

        Args:
            layout (Layout): SpatialLM layout object containing detected elements.

        Returns:
            Dict: JSON-serializable dictionary containing 3D bounding box data.
        """
        detections = []
        for element in layout.elements:
            # Extract bounding box information
            # Assuming element has attributes like type, center, size, etc.
            bbox = {
                'type': element.type,  # e.g., 'wall', 'door', 'window', 'box'
                'center': element.center.tolist() if hasattr(element, 'center') else [0, 0, 0],  # Center of the bounding box
                'size': element.size.tolist() if hasattr(element, 'size') else [1, 1, 1],  # Dimensions (width, height, depth)
                'orientation': element.orientation.tolist() if hasattr(element, 'orientation') else [0, 0, 0, 1],  # Quaternion
            }
            detections.append(bbox)
        
        return {
            'detections': detections,
            'timestamp': np.datetime64('now').astype(str)
        }

    def process_point_cloud(self, pcd: o3d.geometry.PointCloud) -> Tuple[o3d.geometry.PointCloud, Dict]:
        """
        Process point cloud using SpatialLM to generate 3D bounding boxes in JSON format.

        Args:
            pcd (o3d.geometry.PointCloud): Input point cloud.

        Returns:
            Tuple[o3d.geometry.PointCloud, Dict]: Original point cloud and JSON with 3D bounding box data.
        """
        try:
            # Clean up point cloud
            pcd = cleanup_pcd(pcd)
            points, colors = get_points_and_colors(pcd)
            min_extent = np.min(points, axis=0)

            # Preprocess point cloud
            grid_size = Layout.get_grid_size()
            num_bins = Layout.get_num_bins()
            input_pcd = self.preprocess_point_cloud(points, colors, grid_size, num_bins)

            # Load code template
            with open(self.code_template_file, "r") as f:
                code_template = f.read()

            # Prepare prompt
            prompt = f"<|point_start|><|point_pad|><|point_end|>Detect walls, doors, windows, boxes. The reference code is as followed: {code_template}"

            # Prepare conversation data
            if self.model.config.model_type == SpatialLMLlamaForCausalLM.config_class.model_type:
                conversation = [{"role": "user", "content": prompt}]
            elif self.model.config.model_type == SpatialLMQwenForCausalLM.config_class.model_type:
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ]
            else:
                raise ValueError(f"Unsupported model type: {self.model.config.model_type}")

            # Tokenize input
            input_ids = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, return_tensors="pt"
            ).to(self.device)

            # Set up streamer for generation
            streamer = TextIteratorStreamer(
                self.tokenizer, timeout=20.0, skip_prompt=True, skip_special_tokens=True
            )

            # Generate layout in a separate thread
            generate_kwargs = dict(
                {"input_ids": input_ids, "point_clouds": input_pcd},
                streamer=streamer,
                max_new_tokens=4096,
                do_sample=True,
                use_cache=True,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                num_beams=self.num_beams,
            )
            t = Thread(target=self.model.generate, kwargs=generate_kwargs)
            t.start()

            # Collect generated text
            generate_texts = []
            for text in streamer:
                generate_texts.append(text)
            
            # Process layout
            layout_str = "".join(generate_texts)
            layout = Layout(layout_str)
            layout.undiscretize_and_unnormalize()
            layout.translate(min_extent)

            # Convert layout to JSON
            layout_json = self.layout_to_json(layout)

            return pcd, layout_json

        except Exception as e:
            print(f"Processing error: {e}")
            return pcd, {'error': str(e), 'detections': []}

processor = SpatialProcessor()
app = processor.app