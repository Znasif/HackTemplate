import subprocess
import httpx
import asyncio
from asyncio import Queue, Event
import json
import base64
import socket
import os
import signal
from audio_processors.sonic_processor import NovaSonicStreamManager
from audio_processors.flash_processor import GeminiFlashStreamManager
import time
import wave
from datetime import datetime
from typing import List, Dict, Optional, Union, Any, Callable, AsyncGenerator
from pathlib import Path
from io import BytesIO
import tempfile
from dotenv import load_dotenv
import traceback
import numpy as np
from pydub import AudioSegment

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
    "*"
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

class AudioStreamRecorder:
    """Handles recording of streaming PCM audio chunks and saves them as an MP3 file."""
    def __init__(self, output_dir: str = "audio_recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.active_recordings: Dict[str, Dict[str, Any]] = {}

    def start_recording(self, session_id: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_stream_{session_id}_{timestamp}.mp3"
        filepath = self.output_dir / filename
        self.active_recordings[session_id] = {'filepath': filepath, 'audio_buffer': BytesIO()}
        log_message(f"Started audio recording session {session_id}")
        return str(filepath)
    
    def add_audio_chunk(self, session_id: str, audio_data: bytes) -> bool:
        if session_id not in self.active_recordings: return False
        self.active_recordings[session_id]['audio_buffer'].write(audio_data)
        return True
    
    def stop_recording_and_convert(self, session_id: str) -> Optional[str]:
        if session_id not in self.active_recordings:
            log_message(f"No active recording for session {session_id}", level="WARNING")
            return None
        try:
            recording = self.active_recordings[session_id]
            filepath = recording['filepath']
            recording['audio_buffer'].seek(0)
            pcm_data = recording['audio_buffer'].read()
            if pcm_data:
                mp3_data = self._convert_pcm_to_mp3(pcm_data)
                if mp3_data:
                    with open(filepath, 'wb') as f: f.write(mp3_data)
                    log_message(f"Stopped and saved audio recording to {filepath}")
                    return str(filepath)
                else:
                    log_message(f"Failed to convert PCM to MP3 for session {session_id}", level="ERROR")
                    return None
            else:
                log_message(f"No PCM data to convert for session {session_id}", level="ERROR")
                return None
        except Exception as e:
            log_message(f"Error stopping recording: {e}", level="ERROR")
            return None
        finally:
            if session_id in self.active_recordings:
                self.active_recordings[session_id]['audio_buffer'].close()
                del self.active_recordings[session_id]

    def _convert_pcm_to_mp3(self, pcm_data: bytes) -> Optional[bytes]:
        if not pcm_data: return None
        try:
            # Convert raw PCM to WAV format first
            wav_buffer = BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(16000)  # 16 kHz
                wav_file.writeframes(pcm_data)
            wav_buffer.seek(0)
            # Convert WAV to MP3 using pydub
            audio = AudioSegment.from_wav(wav_buffer)
            mp3_buffer = BytesIO()
            audio.export(mp3_buffer, format="mp3", bitrate="128k")
            return mp3_buffer.getvalue()
        except Exception as e:
            log_message(f"PCM to MP3 conversion failed: {e}", level="ERROR")
            return None

    def cleanup_all_dangling_sessions(self):
        sessions_to_clean = list(self.active_recordings.keys())
        if not sessions_to_clean:
            return
        log_message(f"Cleaning up {len(sessions_to_clean)} dangling audio session(s)...")
        for session_id in sessions_to_clean:
            try:
                self.active_recordings[session_id]['audio_buffer'].close()
                del self.active_recordings[session_id]
            except Exception as e:
                log_message(f"Error cleaning up dangling session {session_id}: {e}", level="ERROR")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client = httpx.AsyncClient(timeout=60.0)
        self.audio_queues: Dict[WebSocket, Queue] = {}
        self.image_queues: Dict[WebSocket, Queue] = {}
        self.audio_recorders: Dict[WebSocket, AudioStreamRecorder] = {}
        self.nova_sessions: Dict[WebSocket, GeminiFlashStreamManager] = {}
        self.active_audio_sessions: Dict[WebSocket, str] = {}
        self.last_images: Dict[WebSocket, Optional[str]] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.audio_queues[websocket] = Queue()
        self.image_queues[websocket] = Queue()
        self.audio_recorders[websocket] = AudioStreamRecorder()
        self.nova_sessions[websocket] = GeminiFlashStreamManager(websocket, self)
        self.last_images[websocket] = None

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections: self.active_connections.remove(websocket)
        if websocket in self.nova_sessions: await self.nova_sessions.pop(websocket).close()
        self.audio_queues.pop(websocket, None)
        self.image_queues.pop(websocket, None)
        self.audio_recorders.pop(websocket, None)
        self.active_audio_sessions.pop(websocket, None)
        self.last_images.pop(websocket, None)
        log_message("Cleaned up resources for disconnected client.")

    async def audio_processor_task(self, websocket: WebSocket, queue: Queue):
        while True:
            message = await queue.get()
            if message is None: break
            message_type = message.get('type')
            try:
                if message_type == 'audio_stream':
                    nova_session = self.nova_sessions[websocket]
                    if not nova_session.is_active:
                        if not await nova_session.initialize_stream():
                            await websocket.send_text(json.dumps({"error": "Failed to connect to speech service."}))
                            break
                        session_id = f"ws_{id(websocket)}_{int(time.time())}"
                        self.active_audio_sessions[websocket] = session_id
                        self.audio_recorders[websocket].start_recording(session_id)
                    audio_chunk_b64 = message.get('audio_chunk')
                    if audio_chunk_b64:
                        audio_bytes = base64.b64decode(audio_chunk_b64)
                        session_id = self.active_audio_sessions[websocket]
                        self.audio_recorders[websocket].add_audio_chunk(session_id, audio_bytes)
                        await nova_session.process_audio_chunk(audio_chunk_b64)
                elif message_type == 'audio_stream_stop':
                    if websocket in self.active_audio_sessions:
                        session_id = self.active_audio_sessions.pop(websocket)
                        filepath = self.audio_recorders[websocket].stop_recording_and_convert(session_id)
                        await websocket.send_text(json.dumps({"status": "audio_recording_stopped", "session_id": session_id, "filepath": filepath, "streaming_back": True}))
                    nova_session = self.nova_sessions[websocket]
                    if nova_session.is_active: await nova_session.end_audio_input()
                    await nova_session.ai_stream_complete.wait()
                    ai_chunks = nova_session.ai_response_audio_chunks
                    total_chunks = len(ai_chunks)
                    if not ai_chunks:
                        await websocket.send_text(json.dumps({"type": "audio_stream_playback", "audio_chunk": "", "chunk_index": 0, "total_chunks": 1, "is_last_chunk": True}))
                        continue
                    for i, chunk_bytes in enumerate(ai_chunks):
                        chunk_b64 = base64.b64encode(chunk_bytes).decode('utf-8')
                        playback_message = {"type": "audio_stream_playback", "audio_chunk": chunk_b64, "chunk_index": i, "total_chunks": total_chunks, "is_last_chunk": i == total_chunks - 1}
                        await websocket.send_text(json.dumps(playback_message))
                        await asyncio.sleep(0.02)
            except Exception as e:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "error": "Audio processing failed",
                    "details": str(e)
                }))
                if websocket in self.nova_sessions:
                    await self.nova_sessions[websocket].close()

    async def image_processor_task(self, websocket: WebSocket, queue: Queue):
        while True:
            message = await queue.get()
            if message is None: break
            image_b64 = message.get("image")
            processor_id = message.get("processor")
            if image_b64: self.last_images[websocket] = image_b64
            if processor_id is not None and image_b64:
                await self.execute_request_and_send(websocket, processor_id, image_b64)
            elif image_b64:
                await websocket.send_text(json.dumps({"status": "image_received", "text": "Image ready."}))

    async def execute_request_and_send(self, websocket: WebSocket, processor_id: int, image_b64: str):
        response_data = await self.execute_request(processor_id, image_b64, None)
        await websocket.send_text(json.dumps(response_data))

    async def execute_request(self, target_processor_id: int, initial_image_b64: Optional[str], initial_point_cloud_json: Optional[Dict]) -> Dict:
        log_message(f"Executing full request for target processor ID: {target_processor_id}")
        current_image_for_processing = initial_image_b64
        final_result_to_client = None
        processed_nodes = set()

        async def _process_node(proc_id):
            nonlocal current_image_for_processing, final_result_to_client
            if proc_id in processed_nodes:
                return True
            node_config = PROCESSOR_CONFIG.get(proc_id)
            if not node_config:
                final_result_to_client = {"error": f"Processor ID {proc_id} not found."}
                return False
            for dep_id in node_config.get("dependencies", []):
                if not await _process_node(dep_id):
                    return False
            payload = {"image": current_image_for_processing} 
            response = await self._call_processor(proc_id, node_config, payload)
            if "error" in response:
                final_result_to_client = response
                return False
            if "image" in response:
                current_image_for_processing = response["image"]
            if proc_id == target_processor_id:
                final_result_to_client = response.get("result")
            processed_nodes.add(proc_id)
            return True

        success = await _process_node(target_processor_id)
        if not success and not final_result_to_client:
            final_result_to_client = {"error": "Processing chain failed."}
        return {"image": current_image_for_processing, "text": final_result_to_client}

    async def _call_processor(self, processor_id: int, processor_config: Dict, input_payload: Dict) -> Dict:
        url = f"http://{processor_config['host']}:{processor_config['port']}/process"
        processor_name = processor_config['name']
        log_message(f"Calling {processor_name} (ID: {processor_id}, URL: {url})")
        try:
            response = await self.client.post(url, json=input_payload, timeout=90.0)
            response.raise_for_status()
            return response.json()
        except httpx.ReadTimeout:
            error_msg = f"Timeout communicating with {processor_name}."
            log_message(error_msg, level="ERROR")
            return {"error": error_msg}
        except httpx.ConnectError:
            error_msg = f"Connection error with {processor_name}. Is it running at {url}?"
            log_message(error_msg, level="ERROR")
            return {"error": error_msg}
        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error {e.response.status_code} from {processor_name}: {e.response.text[:200]}"
            log_message(error_msg, level="ERROR")
            return {"error": f"Error from {processor_name}", "detail": e.response.text[:200]}
        except Exception as e:
            error_msg = f"Error during call to {processor_name}: {str(e)}."
            log_message(f"{error_msg} Traceback: {traceback.format_exc()}", level="ERROR")
            return {"error": error_msg}

    async def handle_websocket_with_parallel_processing(self, websocket: WebSocket):
        audio_task = asyncio.create_task(self.audio_processor_task(websocket, self.audio_queues[websocket]))
        image_task = asyncio.create_task(self.image_processor_task(websocket, self.image_queues[websocket]))
        try:
            while True:
                received_payload_str = await websocket.receive_text()
                client_request = json.loads(received_payload_str)
                if 'type' in client_request and client_request['type'].startswith('audio_stream'):
                    await self.audio_queues[websocket].put(client_request)
                elif 'image' in client_request:
                    await self.image_queues[websocket].put(client_request)
        finally:
            audio_task.cancel()
            image_task.cancel()
            await asyncio.gather(audio_task, image_task, return_exceptions=True)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await manager.handle_websocket_with_parallel_processing(websocket)
    except WebSocketDisconnect:
        log_message("WebSocket connection closed by client.")
    except Exception as e:
        log_message(f"Unhandled error in websocket endpoint: {traceback.format_exc()}", level="CRITICAL")
    finally:
        await manager.disconnect(websocket)

manager = ConnectionManager()

@app.get("/processors")
async def get_processors_info():
    processors_list = [{"id": pid, "name": config["name"], "dependencies": config.get("dependencies", []), "expects_input": config.get("expects_input", "unknown")} for pid, config in PROCESSOR_CONFIG.items()]
    return {"processors": processors_list}

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
    os.makedirs('audio_recordings', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs('scripts', exist_ok=True)
    await start_processor_servers()

@app.on_event("shutdown")
async def shutdown_event():
    global processor_processes
    log_message("Main server shutting down. Terminating processor servers...")
    
    # Stop all active audio recordings
    for session_id in manager.audio_recorder.get_active_sessions():
        manager.audio_recorder.cleanup_session(session_id)
    
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
    log_message("Shutdown complete.")
