from .base_processor import BaseProcessor
from typing import Dict, Union, Tuple, List, Optional

class BasicProcessor(BaseProcessor):
    def __init__(self):
        """
        Initialize Basic processor
        """
        super().__init__()
        
    def process_frame(self, frame):
        """
        Just return frame as is
        """
        # Get original frame dimensions
        
        return frame, ""
    
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        # This processor generates point clouds, doesn't typically process existing ones.
        # Return the input point cloud and an informative message, or raise error.
        # raise NotImplementedError("DepthProcessor does not process existing point clouds.")
        return point_cloud_data, {"message": "DepthProcessor received point cloud data but does not process it further."}
    

processor = BasicProcessor()
app = processor.app