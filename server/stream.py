from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from typing import List
import asyncio
import cv2, json, os
import numpy as np
import base64
from dotenv import load_dotenv
from processors.base_processor import BaseProcessor
from processors.yolo_processor import YOLOProcessor
from processors.aircanvas_processor import AirCanvasProcessor
from processors.mediapipe_processor import MediaPipeProcessor
from processors.camio_processor import MediaPipeGestureProcessor
from processors.ocr_processor import OCRProcessor
from processors.groq_processor import GroqProcessor
from processors.openai_processor import ChatGPTProcessor

app = FastAPI()

# At the module level, before your WebSocket endpoint function
class ProcessorManager:
    # Static processors initialized once for the entire application
    processors = None
    
    @classmethod
    def initialize(cls):
        if cls.processors is None:
            load_dotenv()
            print("Initializing processors...")
            cls.processors = {
                0: OCRProcessor('<DENSE_REGION_CAPTION>'),
                1: OCRProcessor(),
                2: YOLOProcessor("./models/yolo11n-seg.pt"),
                3: MediaPipeGestureProcessor(False),
                4: MediaPipeGestureProcessor(),
                5: GroqProcessor(api_key=os.getenv('GROQ_API_KEY')),
                6: ChatGPTProcessor(api_key=os.getenv('OPENAI_API_KEY'))
            }
    
    @classmethod
    def get_processor(cls, processor_id):
        # Initialize if not already done
        if cls.processors is None:
            cls.initialize()
        
        # Return requested processor or default to OCR
        return cls.processors.get(processor_id, cls.processors[1])

class ConnectionManager:
    def __init__(self):
        ProcessorManager.initialize()
        self.active_connections: List[WebSocket] = []
        self.last_sent_text = ""

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

def print_message(message):
    return
    print(f"Message: {message}", end="\r")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Default to OCR processor
    current_processor_id = 1
    print("\nINFO:     connection open")
    
    try:
        while True:
            print_message("\nWaiting for frame...")
            data = await websocket.receive_text()
            print_message(f"Received data length: {len(data)}")
            
            try:
                # Check if the data is in JSON format
                try:
                    json_data = json.loads(data)
                    # Extract image data and processor_id from JSON
                    if "image" in json_data:
                        data = json_data["image"]
                        
                        # Update processor based on received processor_id
                        if "processor" in json_data:
                            new_processor_id = json_data["processor"]
                            if new_processor_id != current_processor_id:
                                current_processor_id = new_processor_id
                                print_message(f"Switching to processor ID: {current_processor_id}")
                except json.JSONDecodeError:
                    # Not JSON, treat as raw image data
                    pass
                
                # Extract base64 data from data URL
                if data.startswith('data:image/jpeg;base64,'):
                    encoded_data = data.split('base64,')[1]
                else:
                    encoded_data = data
                
                print_message("Decoding base64 data...")
                decoded_data = base64.b64decode(encoded_data)
                print_message(f"Decoded data length: {len(decoded_data)}")
                
                print_message("Converting to numpy array...")
                nparr = np.frombuffer(decoded_data, np.uint8)
                print_message(f"Numpy array shape: {nparr.shape}")
                
                print_message("Decoding image...")
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    print_message("Error: Failed to decode input frame")
                    continue
                
                print_message(f"Frame shape: {frame.shape}")
                
                # Get the current processor from our dictionary
                # Default to processor 1 (OCR) if the requested ID isn't available
                processor = ProcessorManager.get_processor(current_processor_id)
                
                print_message(f"Processing frame with processor ID: {current_processor_id}...")
                processed_frame, detection_text = processor.process_frame(frame)
                print_message(f"Processed frame shape: {processed_frame.shape}")
                
                print_message("Encoding processed frame...")
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]  # Use same quality as client
                success, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                if not success:
                    print_message("Error: Failed to encode processed frame")
                    continue
                
                print_message("Converting to base64...")
                processed_data = base64.b64encode(buffer).decode('utf-8')
                
                # Add data URL prefix for consistency
                image_data = f"data:image/jpeg;base64,{processed_data}"
                response_data = {"image": image_data, "text": ""}
                if manager.last_sent_text != detection_text:
                    response_data["text"] = detection_text
                    manager.last_sent_text = detection_text
                    print(f"Text: {detection_text}")
                print_message(f"Response data length: {len(response_data)}")
                
                print_message("Sending response...")
                await websocket.send_text(json.dumps(response_data))
                print_message("Response sent successfully")
                
            except Exception as e:
                print(f"Error processing frame: {str(e)}")
                print(f"Error type: {type(e)}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                continue
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("INFO:     connection closed")
    except Exception as e:
        print(f"Error in websocket connection: {str(e)}")
        manager.disconnect(websocket)
