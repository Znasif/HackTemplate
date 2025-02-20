import io
from io import BytesIO
import sys
from PIL import Image, ImageDraw
import base64
import matplotlib.pyplot as plt
import numpy as np

def encode_image(image, quality=100):
    """ Encode an image into a base64 string in JPEG format. """

    if image.mode != 'RGB':
        image = image.convert('RGB')  # Convert to RGB
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def exec_without_show(code, show_err=False):
    original_show = plt.show
    plt.show = lambda *args, **kwargs: None
    
    # Empty namespace is fine since plt and np are already in globals()
    namespace = {}
    
    try:
        exec(code, globals(), namespace)
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        img_data = buffer.getvalue()
        buffer.close()
        image = Image.open(io.BytesIO(img_data))
        
    except Exception as e:
        if show_err:
            print(f"Error executing code: {e}", file=sys.stderr)
        image = None
        
    finally:
        plt.show = original_show
        plt.close()
    
    return image

def extract_code_block(response_content):
    """
    Extracts the first code block (delimited by ```) from a string.
    Handles text before and after the code block.
    """
    start_index = response_content.find("```")
    end_index = response_content.find("```", start_index + 3)

    if start_index != -1 and end_index != -1:
        # Extract and return the code block, removing the backticks
        code_block = response_content[start_index + 3:end_index]

        # Strip any optional 'python' marker or extra whitespace
        if code_block.strip().startswith("python"):
            code_block = code_block.strip()[6:]  # Remove 'python' keyword

        return code_block.strip()
    else:
        return None  # No code block found

def render_code(response, show_err=False):
    """
    Executes the given Matplotlib code string, captures the plot as a PNG,
    and returns a PIL.Image object.
    """
    matplotlib_code = extract_code_block(response)
    print(f"Extracted code: \n{matplotlib_code}")
    return exec_without_show(matplotlib_code, show_err)