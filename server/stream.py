import subprocess
import httpx
import asyncio
import json
import base64
import socket
import os
import signal
import time
from typing import List, Dict, Optional, Union, Any # Added Any for flexibility

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect
# from pydantic import BaseModel # Not strictly needed in this file with current setup

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "*"  # TEMPORARY: For debugging, be more specific for production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Using the PROCESSOR_CONFIG provided in your request.
# Ensure names ('depth_processor', 'spatial_processor') match the actual processor script names
# if uvicorn uses them like 'processors.depth_processor:app'.
# Also, ensure conda_env and port details are accurate.

PROCESSOR_CONFIG = {
    0: {
        "host": "127.0.0.1", "port": 8001,
        "name": "base_processor", "conda_env": "whatsai2", "dependencies": [],
        "expects_input": "image",  # DepthProcessor takes an image
    },
    1: {
        "host": "127.0.0.1", "port": 8002,
        "name": "depth_processor", "conda_env": "depth-pro", "dependencies": [],
        "expects_input": "image",  # DepthProcessor takes an image
    },
    2: {
        "host": "127.0.0.1", "port": 8003,
        "name": "dense_processor", "conda_env": "whatsai2", "dependencies": [],
        "expects_input": "image",
    },
    3: {
        "host": "127.0.0.1", "port": 8004,
        "name": "aircanvas_processor", "conda_env": "whatsai2", "dependencies": [],
        "expects_input": "image",  # DepthProcessor takes an image
    },
    4: {
        "host": "127.0.0.1", "port": 8005,
        "name": "camio_processor", "conda_env": "whatsai2", "dependencies": [],
        "expects_input": "image",
    },
    5: {
        "host": "127.0.0.1", "port": 8006,
        "name": "yolo_processor", "conda_env": "whatsai2", "dependencies": [],
        "expects_input": "image",  # DepthProcessor takes an image
    },
    # 6: {
    #     "host": "127.0.0.1", "port": 8007,
    #     "name": "spatial_processor", "conda_env": "spatiallm", "dependencies": [1],
    #     "expects_input": "point_cloud", # SpatialProcessor takes a point cloud
    # },
    7: {
        "host": "127.0.0.1", "port": 8008,
        "name": "mediapipe_processor", "conda_env": "whatsai2", "dependencies": [0],
        "expects_input": "image", # SpatialProcessor takes a point cloud
    },
    # 8: {
    #     "host": "127.0.0.1", "port": 8009,
    #     "name": "parts_processor", "conda_env": "partsfield", "dependencies": [1],
    #     "expects_input": "point_cloud", # SpatialProcessor takes a point cloud
    # }
}


def log_message(message: str, level: str = "INFO"):
    return
    #print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")

# ... (FastAPI app, CORS, other imports, log_message function) ...
# ... (PROCESSOR_CONFIG defined above) ...

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = [] # type: ignore
        self.last_sent_text_summary: str = ""
        self.client = httpx.AsyncClient(timeout=60.0)

        # Instance variables reset for each execute_request call
        self.current_image_for_processing: Optional[str] = None
        self.current_point_cloud_for_processing: Optional[Dict] = None
        self.final_image_to_client: Optional[str] = None
        self.final_result_to_client: Union[str, Dict, None] = None
        self.target_processor_id_for_current_request: Optional[int] = None # Store target_id for nested function

    async def connect(self, websocket: WebSocket): # type: ignore
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket): # type: ignore
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def _call_processor(self, processor_id: int, processor_config: Dict, input_payload: Dict) -> Dict:
        # ... (implementation from previous response is good) ...
        url = f"http://{processor_config['host']}:{processor_config['port']}/process"
        processor_name = processor_config['name']
        log_message(f"Calling {processor_name} (ID: {processor_id}, URL: {url}) with payload keys: {list(input_payload.keys())}")
        try:
            response = await self.client.post(url, json=input_payload, timeout=90.0)
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            error_msg = f"Timeout communicating with {processor_name}."
            log_message(error_msg, level="ERROR")
            return {"error": error_msg, "detail": "The request to the processor timed out."}
        except httpx.ConnectError:
            error_msg = f"Connection error with {processor_name}. Is it running at {url}?"
            log_message(error_msg, level="ERROR")
            return {"error": error_msg, "detail": "Could not connect to the processor."}
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} from {processor_name}: {e.response.text[:200]}"
            log_message(error_msg, level="ERROR")
            return {"error": f"Error from {processor_name}: {e.response.status_code}", "detail": e.response.text[:200]}
        except Exception as e:
            import traceback
            error_msg = f"Error during call to {processor_name}: {str(e)}."
            log_message(f"{error_msg} Traceback: {traceback.format_exc()}", level="ERROR")
            return {"error": error_msg, "detail": "An unexpected error occurred."}


    async def execute_request(self, target_processor_id: int, initial_image_b64: Optional[str], initial_point_cloud_json: Optional[Dict] = None) -> Dict:
        self.current_image_for_processing = initial_image_b64
        self.current_point_cloud_for_processing = initial_point_cloud_json
        self.final_image_to_client = initial_image_b64
        self.final_result_to_client = None
        self.target_processor_id_for_current_request = target_processor_id # Make target_id accessible to nested func

        processed_nodes_in_this_run = set()

        async def _process_node_and_dependencies(proc_id_to_run: int) -> bool:
            if proc_id_to_run in processed_nodes_in_this_run:
                log_message(f"Node {proc_id_to_run} already processed, skipping.", level="DEBUG")
                return True

            node_config = PROCESSOR_CONFIG.get(proc_id_to_run)
            if not node_config:
                err_msg = f"Processor ID {proc_id_to_run} not found in PROCESSOR_CONFIG."
                # Set error on the main instance variable if this is the first error
                if self.final_result_to_client is None or not isinstance(self.final_result_to_client, dict) or "error" not in self.final_result_to_client :
                    self.final_result_to_client = {"error": err_msg}
                log_message(err_msg, level="ERROR")
                return False

            processor_name = node_config["name"]
            log_message(f"Processing node: {processor_name} (ID: {proc_id_to_run})")

            for dep_id in node_config.get("dependencies", []):
                log_message(f"{processor_name} depends on {dep_id}. Processing dependency.")
                if not await _process_node_and_dependencies(dep_id):
                    return False
            
            payload_for_current_node = {}
            expected_input_type = node_config.get("expects_input")

            if expected_input_type == "image":
                if self.current_image_for_processing:
                    payload_for_current_node = {"image": self.current_image_for_processing}
                else:
                    err = f"{processor_name} expects an image, but none is available."
                    self.final_result_to_client = {"error": err}
                    log_message(err, level="ERROR")
                    return False
            elif expected_input_type == "point_cloud":
                if self.current_point_cloud_for_processing:
                    payload_for_current_node = {"point_cloud": self.current_point_cloud_for_processing}
                else:
                    err = f"{processor_name} expects a point_cloud, but none is available."
                    self.final_result_to_client = {"error": err}
                    log_message(err, level="ERROR")
                    return False
            else: # Fallback or error if 'expects_input' is missing/invalid
                err = f"Configuration error for {processor_name}: 'expects_input' is '{expected_input_type}' or missing."
                if not self.current_image_for_processing and not self.current_point_cloud_for_processing:
                    err += " No image or point cloud data is currently available in the processing chain."
                self.final_result_to_client = {"error": err}
                log_message(err, level="ERROR")
                return False
            
            if not payload_for_current_node: # Should be caught by logic above
                err = f"Internal error: Failed to construct payload for {processor_name}."
                self.final_result_to_client = {"error": err}
                log_message(err, level="ERROR")
                return False

            response = await self._call_processor(proc_id_to_run, node_config, payload_for_current_node)

            if "error" in response:
                self.final_result_to_client = response
                log_message(f"Error from {processor_name}: {response.get('detail', response['error'])}", level="ERROR")
                return False

            log_message(f"Successfully processed {processor_name}. Response keys: {list(response.keys())}")
            
            # Update current image state if processor returned one
            if "image" in response: # Check for key existence
                self.current_image_for_processing = response["image"]
                self.final_image_to_client = self.current_image_for_processing
                log_message(f"  Updated current_image_for_processing from {processor_name}.")
            
            node_result_payload = response.get("result")
            # The "result" from the processor that IS the target_processor_id is the final "text" output for the client
            if proc_id_to_run == self.target_processor_id_for_current_request:
                self.final_result_to_client = node_result_payload
                log_message(f"  Set final_result_to_client from target {processor_name}. Type: {type(node_result_payload)}")

            # Update point cloud state for the next processor in the chain.
            # A processor either provides a point cloud in "processed_point_cloud" (if its input was PC and it modified it)
            # OR its "result" field itself might be a point cloud (e.g., DepthProcessor).
            new_point_cloud_data_candidate = None
            if "processed_point_cloud" in response and response["processed_point_cloud"] is not None:
                new_point_cloud_data_candidate = response["processed_point_cloud"]
                log_message(f"  {processor_name} provided 'processed_point_cloud'.")
            elif isinstance(node_result_payload, dict) and \
                 "points" in node_result_payload and \
                 isinstance(node_result_payload.get("points"), list): # Heuristic: if result looks like a PC
                new_point_cloud_data_candidate = node_result_payload
                log_message(f"  {processor_name}'s 'result' is identified as a point cloud.")
            
            if new_point_cloud_data_candidate is not None:
                self.current_point_cloud_for_processing = new_point_cloud_data_candidate
                log_message(f"  Updated current_point_cloud_for_processing from {processor_name}'s output.")
            
            processed_nodes_in_this_run.add(proc_id_to_run)
            return True

        log_message(f"Executing request for target processor ID: {self.target_processor_id_for_current_request}")
        success = await _process_node_and_dependencies(self.target_processor_id_for_current_request) # type: ignore

        if not success and (not isinstance(self.final_result_to_client, dict) or "error" not in self.final_result_to_client): # type: ignore
            self.final_result_to_client = {"error": "Processing chain failed due to an unspecified error."}
        
        if success and self.final_result_to_client is None:
            target_name = PROCESSOR_CONFIG.get(self.target_processor_id_for_current_request, {}).get("name", f"ID {self.target_processor_id_for_current_request}") # type: ignore
            log_message(f"Target processor {target_name} ran successfully but its 'result' was None.", level="WARNING")
            self.final_result_to_client = "" 

        return {"image": self.final_image_to_client, "text": self.final_result_to_client}

manager = ConnectionManager()

@app.get("/processors")
async def get_processors_info():
    processors_list = [
        {
            "id": pid, 
            "name": config["name"], 
            "dependencies": config.get("dependencies", []),
            "expects_input": config.get("expects_input", "unknown"), # Include expects_input
        }
        for pid, config in PROCESSOR_CONFIG.items()
    ]
    return {"processors": processors_list}

# ... (websocket_endpoint from your provided code, which calls manager.execute_request or manager.process_frame) ...
# Ensure the websocket_endpoint calls `await manager.execute_request(...)` correctly.
# The existing websocket_endpoint in your prompt is:
#   response_data = await manager.process_frame(current_processor_id, data)
# This should be changed to:
#   response_data = await manager.execute_request(target_processor_id, image_b64_from_client, point_cloud_from_client)
# where target_processor_id, image_b64_from_client, etc., are parsed from the client's message.

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket): # type: ignore
    await manager.connect(websocket)
    log_message("WebSocket connection opened.")
    default_target_processor_id = 0 # Example: default if client doesn't specify
    if not PROCESSOR_CONFIG:
        log_message("PROCESSOR_CONFIG is empty. Websocket may not function correctly.", level="ERROR")

    try:
        while True:
            received_payload_str = await websocket.receive_text()
            log_message(f"Received from client: {received_payload_str[:200]}")

            try:
                client_request = json.loads(received_payload_str)
                image_b64_from_client = client_request.get("image")
                point_cloud_from_client = client_request.get("point_cloud")
                target_processor_id = client_request.get("processor") # Client specifies target ID

                if target_processor_id is None:
                    # Option 1: Use a default if none specified
                    # target_processor_id = default_target_processor_id
                    # log_message(f"No 'processor' ID specified by client, defaulting to {target_processor_id}", level="WARNING")
                    # Option 2: Require it
                    await websocket.send_text(json.dumps({"error": "Client must specify a 'processor' ID.", "text": "Processor ID missing."}))
                    continue
                
                # Validate target_processor_id
                if not isinstance(target_processor_id, int) or PROCESSOR_CONFIG.get(target_processor_id) is None:
                    await websocket.send_text(json.dumps({"error": f"Invalid 'processor' ID: {target_processor_id}.", "text": "Invalid processor ID."}))
                    continue
                
                if not image_b64_from_client and not point_cloud_from_client: # Must have at least one initial input
                    await websocket.send_text(json.dumps({"error": "No 'image' or 'point_cloud' data provided.", "text": "Input data missing."}))
                    continue

                # Call the refactored execution logic
                response_data = await manager.execute_request(target_processor_id, image_b64_from_client, point_cloud_from_client)
                
                await websocket.send_text(json.dumps(response_data))
                text_summary = str(response_data.get("text", ""))[:100]
                log_message(f"Response for target processor {target_processor_id} sent. Text summary: {text_summary}")

            except json.JSONDecodeError:
                log_message("Invalid JSON received from client.", level="ERROR")
                await websocket.send_text(json.dumps({"error": "Invalid JSON payload.", "text": "Bad request."}))
            except Exception as e:
                import traceback
                tb_str = traceback.format_exc()
                log_message(f"Error processing client request: {str(e)}\n{tb_str}", level="CRITICAL")
                try:
                    await websocket.send_text(json.dumps({"error": f"Server error: {str(e)}", "text": "Server error."}))
                except Exception as ws_send_err:
                     log_message(f"Could not send error to client (websocket likely closed): {ws_send_err}", level="ERROR")
                # continue # Keep the loop running for the websocket connection

    except WebSocketDisconnect:
        log_message("WebSocket connection closed by client.")
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        log_message(f"Unhandled error in WebSocket connection: {str(e)}\n{tb_str}", level="CRITICAL")
    finally:
        manager.disconnect(websocket)
        log_message("WebSocket connection resources cleaned up.")


# ... (start_processor_servers, startup_event, shutdown_event from your provided code, ensure they use log_message) ...
# The start_processor_servers should correctly use the final PROCESSOR_CONFIG and module paths.
# (The version of start_processor_servers from the previous AI response, with checks for processor file paths, was good)
async def start_processor_servers():
    global processor_processes # type: ignore
    processor_processes = []  # type: ignore
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir = os.path.join(parent_dir, 'scripts')
    logs_dir = os.path.join(parent_dir, 'logs')
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    def is_port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind((host, port))
                return False
            except socket.error:
                return True

    if not PROCESSOR_CONFIG:
        log_message("PROCESSOR_CONFIG is empty. No processor servers will be started.", level="WARNING")
        return

    for processor_id, config in PROCESSOR_CONFIG.items():
        if not all(k in config for k in ["host", "port", "name", "conda_env"]):
            log_message(f"Processor ID {processor_id} config is incomplete: {config}. Skipping.", level="ERROR")
            continue

        if is_port_in_use(config['host'], config['port']):
            log_message(f"Skipping {config['name']} - {config['host']}:{config['port']} is already in use.", level="WARNING")
            continue

        module_path = f"processors.{config['name']}:app" # Assumes 'processors' subdirectory
        processor_file_actual_path = os.path.join(parent_dir, "processors", f"{config['name']}.py")
        if not os.path.exists(processor_file_actual_path):
            log_message(f"Processor file not found: {processor_file_actual_path}. For {config['name']}.", level="ERROR")
            continue

        script_path = os.path.join(scripts_dir, f"run_{config['name']}.sh")
        log_file_path = os.path.join(logs_dir, f"{config['name']}.log")
        
        script_content = f"""#!/bin/bash
echo "Attempting to start {config['name']}..."
CONDA_BASE_DIR=$(conda info --base)
if [ -z "$CONDA_BASE_DIR" ]; then echo "Conda base directory not found." >&2; exit 1; fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
if ! conda activate {config['conda_env']}; then echo "Failed to activate conda: {config['conda_env']}" >&2; exit 1; fi
echo "Conda env '{config['conda_env']}' activated for {config['name']}."
cd "{parent_dir}"
echo "Starting uvicorn for {module_path} on {config['host']}:{config['port']}..."
mkdir -p "{logs_dir}"
exec uvicorn {module_path} --host {config['host']} --port {config['port']} --log-level info >> "{log_file_path}" 2>&1
"""
        try:
            with open(script_path, "w") as f: f.write(script_content)
            os.chmod(script_path, 0o755)
            log_message(f"Generated script for {config['name']}: {script_path}")
            
            process = subprocess.Popen( [script_path], preexec_fn=os.setsid if os.name != "nt" else None, cwd=parent_dir )
            await asyncio.sleep(0.5) 

            if process.poll() is not None:
                log_content_snippet = ""
                if os.path.exists(log_file_path):
                    with open(log_file_path, "r") as lf: log_content_snippet = lf.read(500)
                raise RuntimeError(f"Process for {config['name']} failed (exit code {process.returncode}). Log: {log_file_path}\nSnippet:\n{log_content_snippet}")
            
            processor_processes.append(process) # type: ignore
            log_message(f"Started {config['name']} (PID {process.pid}) on {config['host']}:{config['port']}. Log: {log_file_path}")
        except Exception as e:
            log_message(f"Error starting {config['name']}: {str(e)}", level="ERROR")
        await asyncio.sleep(3)

@app.on_event("startup")
async def startup_event():
    log_message("Main server starting up...")
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(parent_dir, 'scripts'), exist_ok=True)
    os.makedirs(os.path.join(parent_dir, 'logs'), exist_ok=True)
    await start_processor_servers()

@app.on_event("shutdown")
async def shutdown_event():
    global processor_processes # type: ignore
    log_message("Main server shutting down. Terminating processor servers...")
    for process in processor_processes: # type: ignore
        if process.poll() is None: 
            pgid = os.getpgid(process.pid) if os.name != "nt" and hasattr(os, 'getpgid') else process.pid
            log_message(f"Terminating process group/PID {pgid} (from Popen PID: {process.pid})...")
            try:
                if os.name != "nt" and hasattr(os, 'killpg'): os.killpg(pgid, signal.SIGTERM) 
                else: process.terminate() 
                process.wait(timeout=5)
                log_message(f"Process PID {process.pid} terminated.")
            except subprocess.TimeoutExpired:
                log_message(f"PID {process.pid} (PGID/PID {pgid}) timeout. Forcing kill...", level="WARNING")
                if os.name != "nt" and hasattr(os, 'killpg'): os.killpg(pgid, signal.SIGKILL)
                else: process.kill()
            except ProcessLookupError: 
                log_message(f"PID {process.pid} (PGID/PID {pgid}) already exited.", level="WARNING")
            except Exception as e:
                 log_message(f"Error terminating PID {process.pid} (PGID/PID {pgid}): {e}", level="ERROR")
    processor_processes = [] # type: ignore
    # Clean up shell scripts
    for processor_id, config in PROCESSOR_CONFIG.items():
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts', f"run_{config['name']}.sh")
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f"{config['name']}.log")
        if os.path.exists(script_path):
            os.remove(script_path)
            print(f"Removed {script_path}")
        if os.path.exists(log_path):
            os.remove(log_path)
            print(f"Removed {log_path}")
    log_message("Shutdown complete.")