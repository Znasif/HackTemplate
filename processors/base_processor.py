from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union, Any, List, Tuple # Added typing imports

class ProcessRequest(BaseModel):
    image: Optional[str] = None       # Base64 encoded image
    point_cloud: Optional[Dict] = None # JSON serializable point cloud data
    # processor_specific_args: Optional[Dict] = None # For future use

class BaseProcessor(ABC):
    def __init__(self):
        self.app = FastAPI()

        @self.app.post("/process")
        async def process(request: ProcessRequest):
            try:
                if request.image:
                    if request.image.startswith('data:image/jpeg;base64,'):
                        encoded_data = request.image.split('base64,')[1]
                    else:
                        encoded_data = request.image
                    
                    decoded_data = base64.b64decode(encoded_data)
                    nparr = np.frombuffer(decoded_data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError("Failed to decode input frame")

                    processed_visual_output, result_payload = self.process_frame(frame)

                    response_dict = {"result": result_payload}
                    if isinstance(processed_visual_output, np.ndarray):
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                        success, buffer = cv2.imencode('.jpg', processed_visual_output, encode_param)
                        if not success:
                            raise ValueError("Failed to encode processed frame")
                        processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
                        response_dict["image"] = f"data:image/jpeg;base64,{processed_image_b64}"
                    # If processed_visual_output is None, "image" key won't be in response_dict.
                    # The chaining logic in the main server will handle this.
                    
                    return response_dict

                elif request.point_cloud:
                    processed_pcd_output, result_payload = self.process_pointcloud(request.point_cloud)
                    # processed_pcd_output is expected to be JSON-serializable (e.g., a Dict or None)
                    response_dict = {"result": result_payload}
                    if processed_pcd_output is not None:
                        response_dict["processed_point_cloud"] = processed_pcd_output
                    
                    return response_dict
                else:
                    raise ValueError("No valid input data provided (expected 'image' or 'point_cloud').")

            except Exception as e:
                import traceback
                error_detail = f"Error in BaseProcessor /process: {str(e)}. Traceback: {traceback.format_exc()}"
                print(error_detail) # Log detailed error on server
                raise HTTPException(status_code=500, detail=str(e)) # Send generic error to client

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Union[str, Dict]]:
        """
        Process the input image frame.
        Returns:
            - Processed visual output (np.ndarray for image, or None if no new visual output).
            - Result payload (string or dictionary).
        """
        pass

    @abstractmethod
    def process_pointcloud(self, point_cloud_data: Dict) -> Tuple[Optional[Dict], Union[str, Dict]]:
        """
        Process the input point cloud data (expected as a dictionary).
        Returns:
            - Processed point cloud data (dictionary, or None if no new point cloud data).
            - Result payload (string or dictionary).
        """
        pass