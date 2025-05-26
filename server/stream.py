import subprocess
import httpx
import asyncio
import json
import base64
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
from typing import List, Dict
from pydantic import BaseModel
import os
import signal
import time

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "https://your-localtunnel-subdomain.loca.lt",
    "*"  # TEMPORARY: For debugging, remove for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for processor servers
PROCESSOR_CONFIG = {0: {
        "host": "127.0.0.1",
        "port": 8001,
        "name": "flame_processor",
        "conda_env": "whatsai2"
    }}

PROCESSOR_CONFIG1 = {
    0: {
        "host": "127.0.0.1",
        "port": 8001,
        "name": "fastvlm_processor",
        "conda_env": "whatsai2"
    },
    1: {
        "host": "127.0.0.1",
        "port": 8002,
        "name": "dense_processor",
        "conda_env": "whatsai2"
    },
    2: {
        "host": "127.0.0.1",
        "port": 8003,
        "name": "camio_processor",
        "conda_env": "whatsai2"
    },
    3: {
        "host": "127.0.0.1",
        "port": 8004,
        "name": "depth_processor",
        "conda_env": "depth-pro"
    },
    4: {
        "host": "127.0.0.1",
        "port": 8005,
        "name": "yolo_processor",
        "conda_env": "whatsai2"
    },
    5: {
        "host": "127.0.0.1",
        "port": 8006,
        "name": "aircanvas_processor",
        "conda_env": "whatsai2"
    },
    6: {
        "host": "127.0.0.1",
        "port": 8007,
        "name": "spatial_processor",
        "conda_env": "spatiallm"
    },
}

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_sent_text = ""
        self.client = httpx.AsyncClient(timeout=30.0)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

    async def process_frame(self, processor_id: int, data: str) -> Dict:
        """Send data to the appropriate processor server and return the response."""
        processor = PROCESSOR_CONFIG.get(processor_id)
        if not processor:
            return {"image": data, "text": f"Processor {processor_id} not found"}

        url = f"http://{processor['host']}:{processor['port']}/process"
        try:
            # Send data to processor server
            response = await self.client.post(
                url,
                json={"image": data},
                timeout=10.0
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print_message(f"Error communicating with processor {processor_id}: {str(e)}")
            return {"image": data, "text": f"Error processing with {processor['name']}: {str(e)}"}

manager = ConnectionManager()

def print_message(message):
    #print(f"Message: {message}", end="\r")
    return

@app.get("/processors")
async def get_processors():
    """Return the list of available processors with their IDs and names."""
    processors = [
        {"id": pid, "name": config["name"]}
        for pid, config in PROCESSOR_CONFIG.items()
    ]
    return {"processors": processors}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    current_processor_id = 1  # Default to DepthProcessor
    print("\nINFO:     video connection open")

    try:
        while True:
            print_message(f"\nWaiting for frame...")
            data = await websocket.receive_text()

            try:
                # Check if data is JSON
                try:
                    json_data = json.loads(data)
                    if "image" in json_data:
                        data = json_data["image"]
                        if "processor" in json_data:
                            new_processor_id = json_data["processor"]
                            if new_processor_id != current_processor_id:
                                current_processor_id = new_processor_id
                                print_message(f"Switching to processor ID: {current_processor_id}")
                except json.JSONDecodeError:
                    pass

                # Forward data to processor server
                response_data = await manager.process_frame(current_processor_id, data)

                # Send response back to client
                if response_data["text"] != manager.last_sent_text:
                    manager.last_sent_text = response_data["text"]
                    print_message(f"Text: {response_data['text']}")
                await websocket.send_text(json.dumps(response_data))
                print_message(f"Response sent successfully")

            except Exception as e:
                print_message(f"Error processing frame: {str(e)}")
                import traceback
                print_message(f"Traceback: {traceback.format_exc()}")
                continue

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print_message("INFO:     video connection closed")
    except Exception as e:
        print_message(f"Error in websocket connection: {str(e)}")
        manager.disconnect(websocket)

# Store subprocesses for cleanup
processor_processes = []

async def start_processor_servers():
    global processor_processes
    processor_processes = []
    # Ensure the parent folder is the correct working directory
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    
    for processor_id, config in PROCESSOR_CONFIG.items():
        # Create a shell script for each processor with absolute path
        script_path = os.path.join(parent_dir, 'scripts', f"run_{config['name']}.sh")
        script_content = f"""#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate {config['conda_env']}
cd {parent_dir}
uvicorn processors.{config['name']}:app --host {config['host']} --port {config['port']} --log-level debug > {os.path.join(parent_dir, 'logs', f"{config['name']}.log")} 2>&1
"""
        # Write and verify the shell script
        try:
            with open(script_path, "w") as f:
                f.write(script_content)
            print(f"Generated script for {config['name']}: {script_path}")
            print(f"Script content:\n{script_content}")
            os.chmod(script_path, 0o755)
            # Verify script existence and permissions
            if not os.path.exists(script_path):
                raise FileNotFoundError(f"Script {script_path} was not created")
            # Brief delay to ensure filesystem sync
            time.sleep(0.1)
            # Check permissions
            if not os.access(script_path, os.X_OK):
                raise PermissionError(f"Script {script_path} is not executable")
        except Exception as e:
            print(f"Error creating script for {config['name']}: {str(e)}")
            continue
        
        # Start the server process with absolute script path
        try:
            process = subprocess.Popen(
                [os.path.abspath(script_path)],  # Use absolute path
                shell=True,
                preexec_fn=os.setsid,  # Create a new process group for clean termination
                text=True,
                cwd=parent_dir  # Set working directory explicitly
            )
            # Check if process started
            if process.poll() is not None:
                raise RuntimeError(f"Process for {config['name']} failed to start")
            processor_processes.append(process)
            print(f"Started {config['name']} in {config['conda_env']} on {config['host']}:{config['port']}, logging to {os.path.join(parent_dir, 'logs', config['name'] + '.log')}")
        except Exception as e:
            print(f"Error starting {config['name']}: {str(e)}")
        await asyncio.sleep(2)  # Give server time to start

@app.on_event("startup")
async def startup_event():
    print("Starting processor servers...")
    await start_processor_servers()

@app.on_event("shutdown")
async def shutdown_event():
    global processor_processes
    print("Shutting down processor servers...")
    for process in processor_processes:
        # Terminate the process group
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            print(f"Force terminating process {process.pid}")
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    processor_processes = []
    # Clean up shell scripts
    for processor_id, config in PROCESSOR_CONFIG.items():
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', f"run_{config['name']}.sh")
        if os.path.exists(script_path):
            os.remove(script_path)
            print(f"Removed {script_path}")