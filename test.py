import asyncio
import httpx
from fastapi import FastAPI, WebSocket
from starlette.websockets import WebSocketDisconnect
from typing import List
import uvicorn
import cv2

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


def print_message(message):
    print(f"Message: {message}", end="\r")

class IntegratedServer:
    def __init__(self, model_path: str, llama_host: str = "127.0.0.1", 
                 llama_port: int = 8080, websocket_port: int = 8081):
        self.model_path = model_path
        self.llama_host = llama_host
        self.llama_port = llama_port
        self.websocket_port = websocket_port
        self.client = None
        self.app = FastAPI()
        self.manager = ConnectionManager()
        
        # Set up WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            processor = None
            print(f"WebSocket connection established on port {self.websocket_port}")
            
            try:
                while True:
                    print_message("\nWaiting for frame...")
                    data = await websocket.receive_text()
                    print_message(f"Received data length: {len(data)}")
                    
                    try:
                        # Check for processor selection in the data URL
                        if data.startswith('data:image/jpeg;'):
                            # Parse options from the data URL
                            options_part = data.split('data:image/jpeg;')[1].split('base64,')[0]
                            options = dict(opt.split('=') for opt in options_part.strip(';').split(';') if '=' in opt)
                            
                            # Initialize or update processor based on options
                            if 'processor' in options and (processor is None or options['processor'] != processor.__class__.__name__):
                                processor_type = options['processor']
                                if processor_type == 'MediaPipeProcessor':
                                    processor = MediaPipeProcessor()
                                elif processor_type == 'YOLOProcessor':
                                    processor = YOLOProcessor("./models/yolo11n-seg.pt")
                                elif processor_type == 'RembgProcessor':
                                    processor = RembgProcessor()
                                print_message(f"Initialized {processor_type}")
                            
                            # Extract base64 data
                            encoded_data = data.split('base64,')[1]
                        else:
                            encoded_data = data
                            
                        if processor is None:
                            processor = MediaPipeProcessor()  # Default processor
                            
                        # Process the frame
                        print_message("Decoding base64 data...")
                        decoded_data = base64.b64decode(encoded_data)
                        
                        print_message("Converting to numpy array...")
                        nparr = np.frombuffer(decoded_data, np.uint8)
                        
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
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
                        success, buffer = cv2.imencode('.jpg', processed_frame, encode_param)
                        if not success:
                            print_message("Error: Failed to encode processed frame")
                            continue
                        
                        print_message("Converting to base64...")
                        processed_data = base64.b64encode(buffer).decode('utf-8')
                        
                        # Add data URL prefix with current processor type
                        response_data = f"data:image/jpeg;processor={processor.__class__.__name__};base64,{processed_data}"
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
                self.manager.disconnect(websocket)
                print("WebSocket connection closed")

    async def _test_server_connection(self):
        """Test connection to llama.cpp server"""
        url = f"http://{self.llama_host}:{self.llama_port}/health"
        try:
            response = await self.client.get(url)
            return response.status_code == 200
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

    async def start_server(self):
        """
        Asynchronously start both llama.cpp and WebSocket servers.
        """
        # Construct the llama.cpp server launch command
        server_command = [
            '/home/znasif/llama.cpp/build/bin/llama-server',
            '-m', self.model_path,
            '--host', str(self.llama_host),
            '--port', str(self.llama_port)
        ]
        
        # Create async HTTP client
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Start both servers concurrently
        async def start_websocket_server():
            config = uvicorn.Config(
                app=self.app,
                host=self.llama_host,
                port=self.websocket_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()

        # Create tasks for both servers
        websocket_task = asyncio.create_task(start_websocket_server())
        
        # Wait for llama.cpp server to be ready with exponential backoff
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                response = await self._test_server_connection()
                if response:
                    print(f"Llama.cpp server started successfully on {self.llama_host}:{self.llama_port}")
                    break
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                print(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt == max_attempts - 1:
                    raise RuntimeError("Could not connect to the llama.cpp server after multiple attempts")
        
        # Wait for both servers indefinitely
        await asyncio.gather(websocket_task)

# Usage example
async def main():
    server = IntegratedServer(
        model_path="/path/to/your/model",
        llama_host="127.0.0.1",
        llama_port=8080,
        websocket_port=8081
    )
    await server.start_server()

if __name__ == "__main__":
    asyncio.run(main())