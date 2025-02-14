from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from typing import List
import asyncio
import cv2
import numpy as np
import base64
from processors.base_processor import BaseProcessor
from processors.rembg_processor import RembgProcessor
from processors.yolo_processor import YOLOProcessor
from processors.mediapipe_processor import MediaPipeProcessor

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

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
    print(f"Message: {message}", end="\r")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    processor = MediaPipeProcessor()# YOLOProcessor("./models/yolo11n-seg.pt")  # Or RembgProcessor()
    print("\nINFO:     connection open")
    
    try:
        while True:
            print_message("\nWaiting for frame...")
            data = await websocket.receive_text()
            print_message(f"Received data length: {len(data)}")
            
            try:
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
                
                print_message("Processing frame...")
                processed_frame = processor.process_frame(frame)
                print_message(f"Processed frame shape: {processed_frame.shape}")
                
                print_message("Encoding processed frame...")
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]  # Use same quality as client
                success, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                if not success:
                    print_message("Error: Failed to encode processed frame")
                    continue
                
                print_message("Converting to base64...")
                processed_data = base64.b64encode(buffer).decode('utf-8')
                
                # Add data URL prefix for consistency
                response_data = f"data:image/jpeg;base64,{processed_data}"
                print_message(f"Response data length: {len(response_data)}")
                
                print_message("Sending response...")
                await websocket.send_text(response_data)
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