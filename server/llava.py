from transformers import (
    PaliGemmaProcessor,
    PaliGemmaForConditionalGeneration,
)
from PIL import Image
import torch
import base64
from io import BytesIO
from typing import Optional
import subprocess
import os
import re
import tempfile
import gc

class ImageDescriber:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not ImageDescriber._initialized:
            print("Initializing PaLI-Gemma model...")
            torch.cuda.empty_cache()
            gc.collect()
            
            model_id = "google/paligemma2-3b-mix-448"
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, 
                torch_dtype=torch.bfloat16, 
                device_map="auto"
            ).eval()
            self.processor = PaliGemmaProcessor.from_pretrained(model_id)
            self.model_path = "/home/znasif/llama.cpp/models/Qwen2-VL-2B-Instruct-Q6_K.gguf"
            self.mmproj_path = "/home/znasif/llama.cpp/models/qwen2-vl-2b-instruct-vision.gguf"
            ImageDescriber._initialized = True

    async def describe_with_gemma(
        self,
        encoded_image: str,
        prompt: str = "describe the image as if to a blind individual"
    ) -> Optional[str]:
        """
        Get image description using PaLI-Gemma model
        
        Args:
            encoded_image: Base64 encoded image string
            prompt: The prompt to use
        
        Returns:
            Optional[str]: Generated description or None if processing fails
        """
        try:
            # Clear CUDA cache before processing
            torch.cuda.empty_cache()
            gc.collect()
            
            # Decode base64 string to image
            image_binary = base64.b64decode(encoded_image)
            image = Image.open(BytesIO(image_binary))
            
            # Process with model
            model_inputs = self.processor(
                text=prompt, 
                images=image, 
                return_tensors="pt"
            ).to(torch.bfloat16).to(self.model.device)
            
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                try:
                    generation = self.model.generate(
                        **model_inputs, 
                        max_new_tokens=100, 
                        do_sample=False
                    )
                    generation = generation[0][input_len:]
                    decoded = self.processor.decode(generation, skip_special_tokens=True)
                    
                    # Clear inputs from GPU memory
                    del model_inputs
                    del generation
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    return decoded.strip()
                except RuntimeError as e:
                    print(f"CUDA error during generation: {e}")
                    # Clear GPU memory and try again
                    del model_inputs
                    torch.cuda.empty_cache()
                    gc.collect()
                    return None
                
        except Exception as e:
            print(f"Error processing image with Gemma: {str(e)}")
            return None
        finally:
            # Ensure GPU memory is cleared
            torch.cuda.empty_cache()
            gc.collect()

    # The describe_with_qwen method remains unchanged since it doesn't use CUDA
    async def describe_with_qwen(
        self,
        encoded_image: str,
        prompt: str = "describe the image as if to a blind individual"
    ) -> Optional[str]:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            try:
                image_binary = base64.b64decode(encoded_image)
                temp_file.write(image_binary)
                temp_file_path = temp_file.name
                
                cmd = [
                    "/home/znasif/llama.cpp/build/bin/llama-qwen2vl-cli",
                    "-m", self.model_path,
                    "--mmproj", self.mmproj_path,
                    "--image", temp_file_path,
                    "-p", prompt
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                pattern = r"encode_image_with_clip:.*?\n\n(.*?ms per image patch\))(.*)"
                match = re.search(pattern, result.stdout, re.DOTALL)
                print(result.stdout)
                if match:
                    return match.group(2)
                return None
                
            except subprocess.CalledProcessError as e:
                print(f"Error executing Qwen command: {e}")
                return None
            except Exception as e:
                print(f"Unexpected error with Qwen: {e}")
                return None
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)