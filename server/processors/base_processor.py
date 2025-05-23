from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import numpy as np
import cv2
from abc import ABC, abstractmethod

class ProcessRequest(BaseModel):
    image: str

class BaseProcessor(ABC):
    def __init__(self):
        self.app = FastAPI()

        @self.app.post("/process")
        async def process(request: ProcessRequest):
            try:
                # Extract base64 data
                if request.image.startswith('data:image/jpeg;base64,'):
                    encoded_data = request.image.split('base64,')[1]
                else:
                    encoded_data = request.image

                # Decode base64 data
                decoded_data = base64.b64decode(encoded_data)
                nparr = np.frombuffer(decoded_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Failed to decode input frame")

                # Process frame
                processed_frame, result = self.process_frame(frame)

                # Encode processed frame
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
                success, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                if not success:
                    raise ValueError("Failed to encode processed frame")

                processed_data = base64.b64encode(buffer).decode('utf-8')
                image_data = f"data:image/jpeg;base64,{processed_data}"

                return {"image": image_data, "text": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """Process the input frame and return the processed frame and result text."""
        pass