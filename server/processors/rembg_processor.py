from .base_processor import BaseProcessor
from rembg import remove
import numpy as np

class RembgProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        
    def process_frame(self, frame):
        """
        Remove background from frame using rembg
        """
        return remove(frame)