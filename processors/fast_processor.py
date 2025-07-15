from .base_processor import BaseProcessor
import numpy as np
import cv2
from PIL import Image
import torch
import os, sys
import time
from typing import Tuple, Dict, Optional, Union # Ensure typing imports

# Path to the directory containing the 'llava' package
llava_parent_dir = r'/home/znasif/ml-fastvlm' # Use raw string for paths

# Add this directory to sys.path if it's not already there
if llava_parent_dir not in sys.path:
    sys.path.insert(0, llava_parent_dir)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.utils import disable_torch_init

class FastVLMProcessor(BaseProcessor):
    def __init__(self, 
                 model_path="/home/znasif/ml-fastvlm/checkpoints/llava-fastvithd_1.5b_stage3",
                 prompt="Describe what you see in the image.",
                 use_gpu=True,
                 model_size="1.5b"):
        """
        Initialize FastVLM processor
        
        Args:
            model_path (str): Path to the FastVLM model
            prompt (str): Default prompt for the model
            use_gpu (bool): Whether to use GPU acceleration
            model_size (str): Model size variant ('0.5b', '1.5b', or '7b')
        """
        super().__init__()
        self.prompt = prompt
        self.model_path = model_path
        self.model_size = model_size
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        self.font = cv2.freetype.createFreeType2()
        font_path = os.path.join(os.path.dirname(__file__), "..", "models", "AtkinsonHyperlegible-Regular.ttf")
        self.font.loadFontData(font_path, 0)
        
        # Load the model and tokenizer
        self._load_model()
        
    def _load_model(self):
        """
        Load the FastVLM model and tokenizer
        """
        # Disable torch default initialization for faster loading
        disable_torch_init()
        
        # Load model components
        model_name = get_model_name_from_path(self.model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, 
            None, 
            model_name, 
            device=self.device
        )
        
        # Store constants for inference
        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.DEFAULT_IMAGE_TOKEN = DEFAULT_IMAGE_TOKEN
        self.DEFAULT_IM_START_TOKEN = DEFAULT_IM_START_TOKEN
        self.DEFAULT_IM_END_TOKEN = DEFAULT_IM_END_TOKEN
        
        print(f"FastVLM model (size: {self.model_size}) loaded successfully")
    
    def preprocess_image(self, frame):
        """
        Preprocess the image for FastVLM
        
        Args:
            frame (numpy.ndarray): Input OpenCV frame
            
        Returns:
            PIL.Image: Processed PIL image
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
        return pil_image
    
    def run_inference(self, pil_image):
        """
        Run FastVLM inference on the image
        
        Args:
            pil_image (PIL.Image): Input image
            
        Returns:
            str: Generated text
        """
        if self.model is None or self.tokenizer is None:
            return "Model not loaded"
        
        try:
            
            
            # Prepare the conversation context
            conv = conv_templates["qwen_2"].copy()
            
            # Construct prompt with image token
            if self.model.config.mm_use_im_start_end:
                qs = self.DEFAULT_IM_START_TOKEN + self.DEFAULT_IMAGE_TOKEN + self.DEFAULT_IM_END_TOKEN + '\n' + self.prompt
            else:
                qs = self.DEFAULT_IMAGE_TOKEN + '\n' + self.prompt
                
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            
            # Tokenize prompt
            input_ids = tokenizer_image_token(
                prompt, 
                self.tokenizer, 
                self.IMAGE_TOKEN_INDEX, 
                return_tensors='pt'
            ).unsqueeze(0).to(self.device)
            
            # Process the image
            image_tensor = process_images([pil_image], self.image_processor, self.model.config)[0]
            
            # Measure time to first token (TTFT)
            start_time = time.time()
            
            # Run inference
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().to(self.device),
                    image_sizes=[pil_image.size],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=256,
                    use_cache=True
                )
                
            # Calculate TTFT
            ttft = (time.time() - start_time) * 1000  # Convert to ms
            
            # Decode outputs
            outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            return outputs, ttft
            
        except Exception as e:
            print(f"Inference error: {e}")
            return f"Inference error: {e}", 0
    
    def process_frame(self, frame):
        """
        Process frame using FastVLM
        
        Args:
            frame (numpy.ndarray): Input frame to process
            
        Returns:
            tuple: (processed_frame, text_result)
                - processed_frame (numpy.ndarray): Frame with output overlay
                - text_result (str): Generated text
        """
        # Create output frame as copy of input
        output = frame.copy()
        
        # Preprocess frame
        pil_image = self.preprocess_image(frame)
        
        # Run inference
        if self.model is not None:
            results, ttft = self.run_inference(pil_image)
            
            # Add overlays to output frame
            # Draw background for text
            h, w = output.shape[:2]
            overlay = output.copy()
            cv2.rectangle(overlay, (10, h-120), (w-10, h-20), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, output, 0.4, 0, output)
            
            # Add text
            self.font.putText(
                output,
                f"TTFT: {ttft:.1f}ms",
                (20, h-100),
                fontHeight=12,
                color=(255, 255, 255),
                thickness=1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False
            )
            
            # Add result text (truncate if too long)
            max_chars = 100
            display_text = results if len(results) <= max_chars else results[:max_chars] + "..."
            self.font.putText(
                output,
                display_text,
                (20, h-70),
                fontHeight=12,
                color=(255, 255, 255),
                thickness=1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False
            )
            
            # Return the processed frame and full text
            return output, results
        else:
            # Model not loaded, return original frame with error message
            self.font.putText(
                output,
                "FastVLM model not loaded",
                (20, 30),
                fontHeight=12,
                color=(0, 0, 255),
                thickness=1,
                line_type=cv2.LINE_AA,
                bottomLeftOrigin=False
            )
            return output, "Model not loaded"
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        # This processor generates point clouds, doesn't typically process existing ones.
        # Return the input point cloud and an informative message, or raise error.
        # raise NotImplementedError("DepthProcessor does not process existing point clouds.")
        return point_cloud_data, {"message": "DepthProcessor received point cloud data but does not process it further."}

processor = FastVLMProcessor()
app = processor.app