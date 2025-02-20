import base64
from io import BytesIO
import subprocess
import os
import re
from typing import Optional

async def describe_image(
    encoded_image: str,
    prompt: str = "describe the image as if to a blind individual"
) -> Optional[str]:
    """
    Get image description from base64 encoded image string
    
    Args:
        model_path: Path to the Qwen2-VL model
        mmproj_path: Path to the vision model
        encoded_image: Base64 encoded image string from encode_image function
        prompt: The prompt to use (default: "describe")
    """
    import tempfile
    
    # Decode base64 string to binary
    image_binary = base64.b64decode(encoded_image)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
        temp_file.write(image_binary)
        temp_file_path = temp_file.name
    
    model_path="/home/znasif/llama.cpp/models/Qwen2-VL-2B-Instruct-Q6_K.gguf"
    mmproj_path="/home/znasif/llama.cpp/models/qwen2-vl-2b-instruct-vision.gguf"
    
    try:
        # Execute the command
        cmd = [
            "/home/znasif/llama.cpp/build/bin/llama-qwen2vl-cli",
            "-m", model_path,
            "--mmproj", mmproj_path,
            "--image", temp_file_path,
            "-p", prompt
        ]
        
        # Run the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract description
        pattern = r"encode_image_with_clip:.*?\n\n(.*?ms per image patch\))(.*)"
        match = re.search(pattern, result.stdout, re.DOTALL)
        print(result.stdout)
        if match:
            return match.group(1) + match.group(2)
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    finally:
        # Clean up by deleting the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
