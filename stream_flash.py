import subprocess
import httpx
import asyncio
from asyncio import Queue
import json
import base64
import socket
import os, sys
import signal
import time
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from io import BytesIO
import tempfile
from dotenv import load_dotenv

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

from audio_processors.gemini_processor import GeminiFlashStreamManager

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

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
    return {
        int(k): v for k, v in config.items() if v.get("enabled", True)
    }

PROCESSOR_CONFIG = load_config("processor_config.json")

def log_message(message: str, level: str = "INFO"):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")
    #sys.stdout.flush()

class AudioStreamRecorder:
    """Handles recording of streaming audio chunks in memory and conversion to MP3"""
    
    def __init__(self, output_dir: str = "audio_recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.active_recordings = {}  # session_id -> recording_info
        
    def start_recording(self, session_id: str) -> str:
        """Start a new audio recording session (in memory)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_stream_{session_id}_{timestamp}.mp3"
        filepath = self.output_dir / filename
        
        self.active_recordings[session_id] = {
            'filepath': filepath,
            'filename': filename,
            'audio_buffer': BytesIO(),  # Store audio in memory
            'start_time': datetime.now(),
            'chunk_count': 0,
            'total_bytes': 0,
        }
        
        log_message(f"Started audio recording session {session_id} (in memory)")
        return str(filepath)
    
    def add_audio_chunk(self, session_id: str, audio_data: bytes) -> bool:
        """Add an audio chunk to the in-memory buffer"""
        if session_id not in self.active_recordings:
            log_message(f"No active recording session: {session_id}", level="WARNING")
            return False
            
        try:
            recording = self.active_recordings[session_id]
            
            # Write to memory buffer
            recording['audio_buffer'].write(audio_data)
            
            recording['chunk_count'] += 1
            recording['total_bytes'] += len(audio_data)
            
            log_message(f"Added audio chunk to session {session_id}: {len(audio_data)} bytes (total: {recording['total_bytes']} bytes, chunks: {recording['chunk_count']})")
            return True
            
        except Exception as e:
            log_message(f"Error adding audio chunk to session {session_id}: {e}", level="ERROR")
            return False
    
    def stop_recording_and_convert(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Stop recording, convert to MP3"""
        if session_id not in self.active_recordings:
            log_message(f"No active recording for session {session_id}", level="WARNING")
            return None
            
        try:
            recording = self.active_recordings[session_id]
            filepath = recording['filepath']
            
            # Get the complete audio data from buffer
            recording['audio_buffer'].seek(0)
            pcm_data = recording['audio_buffer'].read()
            
            # Convert PCM to MP3 using ffmpeg
            mp3_data = self._convert_pcm_to_mp3(pcm_data)
            
            if mp3_data:
                # Save MP3 file
                with open(filepath, 'wb') as f:
                    f.write(mp3_data)
                
                # Log recording statistics
                duration = datetime.now() - recording['start_time']
                log_message(f"Stopped audio recording for session {session_id}")
                log_message(f"  File: {filepath}")
                log_message(f"  Duration: {duration}")
                log_message(f"  Total chunks: {recording['chunk_count']}")
                log_message(f"  Total bytes (PCM): {recording['total_bytes']}")
                log_message(f"  MP3 size: {len(mp3_data)} bytes")
                
                # Prepare result
                result = {
                    'filepath': str(filepath),
                    'mp3_data': mp3_data,  # Converted MP3 data
                    'total_chunks': recording['chunk_count'],
                    'duration': str(duration)
                }
                
                # Clean up
                recording['audio_buffer'].close()
                del self.active_recordings[session_id]
                
                return result
            else:
                log_message(f"Failed to convert audio to MP3 for session {session_id}", level="ERROR")
                return None
                
        except Exception as e:
            log_message(f"Error stopping recording for session {session_id}: {e}", level="ERROR")
            return None
    
    def _convert_pcm_to_mp3(self, pcm_data: bytes) -> Optional[bytes]:
        """Convert raw PCM audio data (s16le, 16kHz, mono) to MP3 using ffmpeg"""
        try:
            # Create temporary files for conversion
            with tempfile.NamedTemporaryFile(suffix='.pcm', delete=False) as temp_pcm:
                temp_pcm.write(pcm_data)
                temp_pcm_path = temp_pcm.name
            
            temp_mp3_path = temp_pcm_path.replace('.pcm', '.mp3')
            
            # Run ffmpeg conversion, specifying input format for raw PCM
            cmd = [
                'ffmpeg',
                '-f', 's16le',      # Input format: signed 16-bit little-endian
                '-ar', '24000',     # Input sample rate
                '-ac', '1',         # Input channels (mono)
                '-i', temp_pcm_path,
                '-acodec', 'mp3',
                '-ab', '128k',
                '-ar', '24000',
                temp_mp3_path,
                '-y'  # Overwrite output file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                log_message(f"FFmpeg conversion failed: {result.stderr}", level="ERROR")
                return None
            
            # Read the converted MP3
            with open(temp_mp3_path, 'rb') as f:
                mp3_data = f.read()
            
            # Clean up temporary files
            os.unlink(temp_pcm_path)
            os.unlink(temp_mp3_path)
            
            return mp3_data
            
        except Exception as e:
            log_message(f"Error during PCM to MP3 conversion: {e}", level="ERROR")
            return None
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active recording sessions"""
        return list(self.active_recordings.keys())
    
    def cleanup_session(self, session_id: str):
        """Clean up a session without proper stop (e.g., on disconnect)"""
        if session_id in self.active_recordings:
            try:
                self.active_recordings[session_id]['audio_buffer'].close()
                del self.active_recordings[session_id]
                log_message(f"Cleaned up recording session {session_id}")
            except Exception as e:
                log_message(f"Error cleaning up session {session_id}: {e}", level="ERROR")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.last_sent_text_summary: str = ""
        self.client = httpx.AsyncClient(timeout=60.0)
        
        # Audio recording components
        self.audio_recorder = AudioStreamRecorder()
        self.gemini_manager = GeminiFlashStreamManager(self.audio_recorder, self)
        self.active_audio_sessions = {}  # websocket -> session_id mapping

        self.audio_queues = {}  # websocket -> Queue
        self.image_queues = {}

        # Instance variables reset for each execute_request call
        self.current_image_for_processing: Optional[str] = None
        self.current_point_cloud_for_processing: Optional[Dict] = None
        self.final_image_to_client: Optional[str] = None
        self.final_result_to_client: Union[str, Dict, None] = None
        self.target_processor_id_for_current_request: Optional[int] = None

        load_dotenv()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Clean up any active audio recording and Gemini session
        self.stop_audio_recording(websocket)

    async def handle_audio_stream(self, websocket: WebSocket, message_data: Dict[str, Any]):
        """Handle audio streaming data from client."""
        try:
            message_type = message_data.get('type')
            
            if message_type == 'audio_stream':
                audio_chunk_b64 = message_data.get('audio_chunk')
                if not audio_chunk_b64:
                    log_message("No audio chunk data received", level="WARNING")
                    return
                
                # Get or create session for this websocket
                session_id = self.active_audio_sessions.get(websocket)
                if not session_id:
                    # Start new recording session
                    session_id = f"ws_{id(websocket)}_{int(time.time())}"
                    self.active_audio_sessions[websocket] = session_id
                    filepath = self.audio_recorder.start_recording(session_id)
                    
                    if filepath:
                        # Start Gemini streaming session
                        asyncio.create_task(self.gemini_manager.start_session(websocket, session_id))
                        await websocket.send_text(json.dumps({
                            "status": "audio_recording_started",
                            "session_id": session_id,
                            "filepath": filepath
                        }))
                    else:
                        await websocket.send_text(json.dumps({
                            "error": "Failed to start audio recording"
                        }))
                        return
                
                # Process audio chunk through Gemini manager
                await self.gemini_manager.handle_audio_chunk(websocket, audio_chunk_b64, session_id)
            
            elif message_type == 'audio_stream_stop':
                session_id = self.active_audio_sessions.get(websocket)
                if session_id:
                    # Stop Gemini session and get recording results and Gemini responses
                    result = await self.gemini_manager.stop_session(websocket)
                    if result and result['recording']:
                        # Send completion status
                        recording = result['recording']
                        await websocket.send_text(json.dumps({
                            "status": "audio_recording_stopped",
                            "session_id": session_id,
                            "filepath": recording['filepath'],
                            "total_chunks": len(result['gemini_audio_responses']),
                            "duration": recording['duration'],
                            "streaming_back": True
                        }))
                        
                        # Stream Gemini audio responses back to client for playback
                        for i, chunk in enumerate(result['gemini_audio_responses']):
                            chunk_b64 = base64.b64encode(chunk).decode('utf-8')
                            await websocket.send_text(json.dumps({
                                "type": "audio_stream_playback",
                                "audio_chunk": chunk_b64,
                                "chunk_index": i,
                                "total_chunks": len(result['gemini_audio_responses']),
                                "is_last_chunk": i == len(result['gemini_audio_responses']) - 1
                            }))
                            await asyncio.sleep(0.01)
                        
                        log_message(f"Finished streaming {len(result['gemini_audio_responses'])} Gemini audio response chunks back to client")
                    else:
                        await websocket.send_text(json.dumps({
                            "error": "Failed to process audio recording or Gemini responses"
                        }))
                    del self.active_audio_sessions[websocket]
                else:
                    await websocket.send_text(json.dumps({
                        "status": "no_active_audio_session"
                    }))
                    
        except Exception as e:
            log_message(f"Error handling audio stream: {e}", level="ERROR")
            await websocket.send_text(json.dumps({
                "error": f"Audio stream processing error: {str(e)}"
            }))

    def stop_audio_recording(self, websocket: WebSocket) -> Optional[str]:
        """Stop audio recording and Gemini session for a websocket connection."""
        session_id = self.active_audio_sessions.get(websocket)
        if session_id:
            # Clean up Gemini session
            asyncio.create_task(self.gemini_manager.cleanup_session(websocket))
            # Clean up recording
            self.audio_recorder.cleanup_session(session_id)
            del self.active_audio_sessions[websocket]
            return session_id
        return None

    async def _call_processor(self, processor_id: int, processor_config: Dict, input_payload: Dict) -> Dict:
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

    async def handle_websocket_with_parallel_processing(self, websocket: WebSocket):
        """Handle both audio and image processing in parallel"""
        
        # Create queues for this connection
        audio_queue = Queue()
        image_queue = Queue()
        self.audio_queues[websocket] = audio_queue
        self.image_queues[websocket] = image_queue
        
        # Create parallel tasks
        audio_task = asyncio.create_task(
            self.audio_processor_task(websocket, audio_queue)
        )
        image_task = asyncio.create_task(
            self.image_processor_task(websocket, image_queue)
        )
        message_router_task = asyncio.create_task(
            self.message_router(websocket, audio_queue, image_queue)
        )
        
        try:
            # Wait for any task to complete (likely due to disconnect)
            done, pending = await asyncio.wait(
                [audio_task, image_task, message_router_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                
        finally:
            # Cleanup
            del self.audio_queues[websocket]
            del self.image_queues[websocket]

    async def message_router(self, websocket: WebSocket, audio_queue: Queue, image_queue: Queue):
        """Route incoming messages to appropriate queues"""
        try:
            while True:
                received_payload_str = await websocket.receive_text()
                client_request = json.loads(received_payload_str)
                
                if 'type' in client_request and client_request['type'].startswith('audio_stream'):
                    await audio_queue.put(client_request)
                else:
                    await image_queue.put(client_request)
                    
        except WebSocketDisconnect:
            # Signal other tasks to stop
            await audio_queue.put(None)
            await image_queue.put(None)
            raise

    async def audio_processor_task(self, websocket: WebSocket, queue: Queue):
        """Dedicated task for processing audio messages"""
        while True:
            message = await queue.get()
            if message is None:  # Shutdown signal
                break
                
            try:
                await self.handle_audio_stream(websocket, message)
            except Exception as e:
                log_message(f"Error in audio processor: {e}", level="ERROR")

    async def image_processor_task(self, websocket: WebSocket, queue: Queue):
        """Dedicated task for processing image/processor messages"""
        while True:
            # Get frame data from queue
            message = await queue.get()
            
            if message is None:  # Shutdown signal
                break
            try:
                # Process the frame
                image_b64 = message.get("image")
                point_cloud = message.get("point_cloud")
                processor_id = message.get("processor")
                
                # Store the last image frame for potential Gemini tool calls
                if image_b64:
                    self.gemini_manager.last_frame_by_websocket[websocket] = image_b64
                
                if processor_id is not None:
                    response_data = await self.execute_request(
                        processor_id, image_b64, point_cloud
                    )
                    await websocket.send_text(json.dumps(response_data))
                
            except Exception as e:
                log_message(f"Error in image processor: {e}", level="ERROR")
                # Send error response
                await websocket.send_text(json.dumps({
                    "error": f"Processing error: {str(e)}"
                }))

    async def execute_request(self, target_processor_id: int, initial_image_b64: Optional[str], initial_point_cloud_json: Optional[Dict] = None) -> Dict:
        self.current_image_for_processing = initial_image_b64
        self.current_point_cloud_for_processing = initial_point_cloud_json
        self.final_image_to_client = initial_image_b64
        self.final_result_to_client = None
        self.target_processor_id_for_current_request = target_processor_id

        processed_nodes_in_this_run = set()

        async def _process_node_and_dependencies(proc_id_to_run: int) -> bool:
            if proc_id_to_run in processed_nodes_in_this_run:
                log_message(f"Node {proc_id_to_run} already processed, skipping.", level="DEBUG")
                return True

            node_config = PROCESSOR_CONFIG.get(proc_id_to_run)
            if not node_config:
                err_msg = f"Processor ID {proc_id_to_run} not found in PROCESSOR_CONFIG."
                if self.final_result_to_client is None or not isinstance(self.final_result_to_client, dict) or "error" not in self.final_result_to_client:
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
            else:
                err = f"Configuration error for {processor_name}: 'expects_input' is '{expected_input_type}' or missing."
                if not self.current_image_for_processing and not self.current_point_cloud_for_processing:
                    err += " No image or point cloud data is currently available in the processing chain."
                self.final_result_to_client = {"error": err}
                log_message(err, level="ERROR")
                return False
            
            if not payload_for_current_node:
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
            if "image" in response:
                self.current_image_for_processing = response["image"]
                self.final_image_to_client = self.current_image_for_processing
                log_message(f"  Updated current_image_for_processing from {processor_name}.")
            
            node_result_payload = response.get("result")
            # The "result" from the processor that IS the target_processor_id is the final "text" output for the client
            if proc_id_to_run == self.target_processor_id_for_current_request:
                self.final_result_to_client = node_result_payload
                log_message(f"  Set final_result_to_client from target {processor_name}. Type: {type(node_result_payload)}")

            # Update point cloud state for the next processor in the chain.
            new_point_cloud_data_candidate = None
            if "processed_point_cloud" in response and response["processed_point_cloud"] is not None:
                new_point_cloud_data_candidate = response["processed_point_cloud"]
                log_message(f"  {processor_name} provided 'processed_point_cloud'.")
            elif isinstance(node_result_payload, dict) and \
                    "points" in node_result_payload and \
                    isinstance(node_result_payload.get("points"), list):
                new_point_cloud_data_candidate = node_result_payload
                log_message(f"  {processor_name}'s 'result' is identified as a point cloud.")
            
            if new_point_cloud_data_candidate is not None:
                self.current_point_cloud_for_processing = new_point_cloud_data_candidate
                log_message(f"  Updated current_point_cloud_for_processing from {processor_name}'s output.")
            
            processed_nodes_in_this_run.add(proc_id_to_run)
            return True

        log_message(f"Executing request for target processor ID: {self.target_processor_id_for_current_request}")
        success = await _process_node_and_dependencies(self.target_processor_id_for_current_request)

        if not success and (not isinstance(self.final_result_to_client, dict) or "error" not in self.final_result_to_client):
            self.final_result_to_client = {"error": "Processing chain failed due to an unspecified error."}
        
        if success and self.final_result_to_client is None:
            target_name = PROCESSOR_CONFIG.get(self.target_processor_id_for_current_request, {}).get("name", f"ID {self.target_processor_id_for_current_request}")
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
            "expects_input": config.get("expects_input", "unknown"),
            "description": config.get("description", "No description available"),
        }
        for pid, config in PROCESSOR_CONFIG.items()
    ]
    return {"processors": processors_list}

@app.get("/audio/sessions")
async def get_audio_sessions():
    """Get information about active audio recording sessions"""
    active_sessions = manager.audio_recorder.get_active_sessions()
    return {
        "active_sessions": active_sessions,
        "session_count": len(active_sessions)
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    log_message("WebSocket connection opened.")
    
    try:
        # Use parallel processing handler
        await manager.handle_websocket_with_parallel_processing(websocket)
    except WebSocketDisconnect:
        log_message("WebSocket connection closed by client.")
    except Exception as e:
        log_message(f"Unhandled error: {e}", level="CRITICAL")
    finally:
        manager.disconnect(websocket)
        log_message("WebSocket connection resources cleaned up.")

async def start_processor_servers():
    global processor_processes
    processor_processes = []
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

        module_path = f"processors.{config['name']}:app"
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
            
            process = subprocess.Popen([script_path], preexec_fn=os.setsid if os.name != "nt" else None, cwd=parent_dir)
            await asyncio.sleep(0.5) 

            if process.poll() is not None:
                log_content_snippet = ""
                if os.path.exists(log_file_path):
                    with open(log_file_path, "r") as lf: log_content_snippet = lf.read(500)
                raise RuntimeError(f"Process for {config['name']} failed (exit code {process.returncode}). Log: {log_file_path}\nSnippet:\n{log_content_snippet}")
            
            processor_processes.append(process)
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
    os.makedirs(os.path.join(parent_dir, 'audio_recordings'), exist_ok=True)
    await start_processor_servers()

@app.on_event("shutdown")
async def shutdown_event():
    global processor_processes
    log_message("Main server shutting down. Terminating processor servers...")
    
    for process in processor_processes:
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
    processor_processes = []
    
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

    # Stop all active audio recordings and Gemini sessions
    for session_id in manager.audio_recorder.get_active_sessions():
        manager.audio_recorder.cleanup_session(session_id)
        for ws, session_data in list(manager.gemini_manager.active_sessions.items()):
            if session_data['session_id'] == session_id:
                await manager.gemini_manager.cleanup_session(ws)

    log_message("Shutdown complete.")