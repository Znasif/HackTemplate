# ====================================================================================
# Main Server Orchestrator using WebRTC
#
# This application uses FastAPI and aiortc to create a WebRTC-based media
# processing pipeline.
#
# Architecture:
# 1. Client (Browser): Connects via WebRTC, sending video and audio streams,
#    and commands over a data channel.
# 2. This Server (Orchestrator): Receives streams and commands.
#    - For video frames, it calls the appropriate external processor.
#    - For audio, it uses the GeminiFlashStreamManager.
# 3. Processors (Microservices): Independent FastAPI apps that perform
#    specific tasks on the data they receive via HTTP from this orchestrator.
# ====================================================================================

import subprocess
import httpx
from collections import deque
import asyncio
import json
import base64
import socket
import os
import sys
import signal
import time
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
from io import BytesIO
import tempfile
import textwrap

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCDataChannel

# --- Configuration and Globals ---

app = FastAPI()

origins = ["*"]  # Allow all origins for easy development

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

try:
    PROCESSOR_CONFIG = load_config("processor_config.json")
except FileNotFoundError:
    print("[ERROR] 'processor_config.json' not found. Please create it. Exiting.")
    sys.exit(1)

def log_message(message: str, level: str = "INFO"):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")
    sys.stdout.flush()

# WebRTC Connection Management
pcs = set()
pc_states: Dict[RTCPeerConnection, Dict[str, Any]] = {}


# --- Core Application Logic ---

class AudioStreamRecorder:
    """Handles recording of streaming audio chunks in memory and conversion to MP3"""
    def __init__(self, output_dir: str = "audio_recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.active_recordings: Dict[str, Dict[str, Any]] = {}

    def start_recording(self, session_id: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_stream_{session_id}_{timestamp}.mp3"
        filepath = self.output_dir / filename
        self.active_recordings[session_id] = {
            'filepath': filepath,
            'filename': filename,
            'audio_buffer': BytesIO(),
            'start_time': datetime.now(),
            'chunk_count': 0,
            'total_bytes': 0,
        }
        log_message(f"Started audio recording session {session_id}")
        return str(filepath)

    def add_audio_chunk(self, session_id: str, audio_data: bytes):
        if session_id not in self.active_recordings:
            return
        recording = self.active_recordings[session_id]
        recording['audio_buffer'].write(audio_data)
        recording['chunk_count'] += 1
        recording['total_bytes'] += len(audio_data)

    # [FIXED] Full implementation of stop_recording_and_convert was missing.
    def stop_recording_and_convert(self, session_id: str) -> Optional[Dict[str, Any]]:
        if session_id not in self.active_recordings:
            return None
        try:
            recording = self.active_recordings[session_id]
            filepath = recording['filepath']
            recording['audio_buffer'].seek(0)
            pcm_data = recording['audio_buffer'].read()
            mp3_data = self._convert_pcm_to_mp3(pcm_data)
            if mp3_data:
                with open(filepath, 'wb') as f:
                    f.write(mp3_data)
                duration = datetime.now() - recording['start_time']
                log_message(f"Stopped audio recording for session {session_id}, saved to {filepath}")
                result = {
                    'filepath': str(filepath),
                    'mp3_data': mp3_data,
                    'total_chunks': recording['chunk_count'],
                    'duration': str(duration)
                }
                self.cleanup_session(session_id)
                return result
            else:
                log_message(f"Failed to convert audio for session {session_id}", level="ERROR")
                self.cleanup_session(session_id)
                return None
        except Exception as e:
            log_message(f"Error stopping recording for {session_id}: {e}", level="ERROR")
            self.cleanup_session(session_id)
            return None

    # [FIXED] Full implementation of _convert_pcm_to_mp3 was missing.
    def _convert_pcm_to_mp3(self, pcm_data: bytes) -> Optional[bytes]:
        try:
            cmd = [
                'ffmpeg',
                '-f', 's16le',
                '-ar', '48000', # [NOTE] WebRTC audio is often 48kHz, adjust if needed
                '-ac', '1',
                '-i', 'pipe:0',
                '-acodec', 'libmp3lame',
                '-b:a', '128k',
                '-f', 'mp3',
                'pipe:1'
            ]
            process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            mp3_data, stderr = process.communicate(input=pcm_data)
            if process.returncode != 0:
                log_message(f"FFmpeg conversion failed: {stderr.decode()}", level="ERROR")
                return None
            return mp3_data
        except Exception as e:
            log_message(f"Error during PCM to MP3 conversion: {e}", level="ERROR")
            return None

    def get_active_sessions(self) -> List[str]:
        return list(self.active_recordings.keys())
        
    def cleanup_session(self, session_id: str):
        if session_id in self.active_recordings:
            try:
                self.active_recordings[session_id]['audio_buffer'].close()
            except Exception:
                pass
            del self.active_recordings[session_id]
            log_message(f"Cleaned up recording session {session_id}")


class GeminiFlashStreamManager:
    """[PLACEHOLDER] Manages streaming audio to Gemini and handling responses."""
    def __init__(self, audio_recorder, rtc_manager):
        self.audio_recorder = audio_recorder
        self.rtc_manager = rtc_manager
        self.active_sessions: Dict[RTCPeerConnection, Dict] = {}
        self.last_frame_by_pc: Dict[RTCPeerConnection, str] = {} # Keyed by peer connection now

    async def start_session(self, pc: RTCPeerConnection, session_id: str):
        pc_id = id(pc)
        log_message(f"[Gemini STUB] Starting session for peer {pc_id} with session_id {session_id}")
        self.active_sessions[pc] = {'session_id': session_id, 'gemini_audio_responses': []}

    async def handle_audio_chunk(self, pc: RTCPeerConnection, audio_chunk_b64: str, session_id: str):
        self.audio_recorder.add_audio_chunk(session_id, base64.b64decode(audio_chunk_b64))

    async def stop_session(self, pc: RTCPeerConnection) -> Optional[Dict]:
        pc_id = id(pc)
        log_message(f"[Gemini STUB] Stopping session for peer {pc_id}")
        if pc in self.active_sessions:
            session_id = self.active_sessions[pc]['session_id']
            recording_result = self.audio_recorder.stop_recording_and_convert(session_id)
            return {
                'recording': recording_result,
                'gemini_audio_responses': self.active_sessions[pc].get('gemini_audio_responses', [])
            }
        return None

    async def cleanup_session(self, pc: RTCPeerConnection):
        if pc in self.active_sessions:
            del self.active_sessions[pc]


class RTCConnectionManager:
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=60.0)
        self.audio_recorder = AudioStreamRecorder()
        self.gemini_manager = GeminiFlashStreamManager(self.audio_recorder, self)
        self.max_frame_queue_size = 5
        load_dotenv()

    def add_peer(self, pc: RTCPeerConnection):
        pc_id = id(pc)
        pcs.add(pc)
        pc_states[pc] = {
            "id": pc_id,
            "data_channel": None,
            "processor_id": None, # [REFACTORED] Start with no processor selected.
            "video_task": None,
            "audio_task": None,
        }
        log_message(f"PeerConnection {pc_id} created. Total peers: {len(pcs)}")

    async def remove_peer(self, pc: RTCPeerConnection):
        if pc not in pcs:
            return
        
        state = pc_states.pop(pc, {})
        pcs.discard(pc)
        pc_id = state.get("id", "N/A")
        log_message(f"Removing peer {pc_id}...")
        
        if state.get("video_task"): state["video_task"].cancel()
        if state.get("audio_task"): state["audio_task"].cancel()
        
        # Cleanup associated sessions
        await self.gemini_manager.cleanup_session(pc)
        self.audio_recorder.cleanup_session(f"rtc_{pc_id}")
        log_message(f"Peer {pc_id} removed and resources cleaned.")


    async def video_track_processor_task(self, pc: RTCPeerConnection, track):
        state = pc_states[pc]
        pc_id = state["id"]
        log_message(f"Video track {track.kind} ({track.id}) received from peer {pc_id}")
        
        # NEW: Initialize a fixed-size queue for frames
        frame_queue = deque(maxlen=self.max_frame_queue_size)
        last_overload_state = False  # Track whether we last sent an overload message

        while True:
            try:
                frame = await track.recv()
                # NEW: Add frame to queue
                frame_queue.append(frame)
                
                # Check for overload
                if len(frame_queue) >= self.max_frame_queue_size:
                    if not last_overload_state and state.get("data_channel"):
                        # Send overload signal to client
                        try:
                            state["data_channel"].send(json.dumps({
                                "type": "server_overload",
                                "drop_frames": True,
                                "reason": "Queue full"
                            }))
                            log_message(f"Peer {pc_id}: Queue full, signaled overload to client")
                            last_overload_state = True
                        except Exception as e:
                            log_message(f"Peer {pc_id}: Failed to send overload signal: {e}", level="ERROR")
                
                # Process only the latest frame if queue is not empty
                if frame_queue and state.get("processor_id") is not None and state.get("data_channel"):
                    # NEW: Always process the most recent frame
                    frame = frame_queue[-1]
                    frame_queue.clear()  # Drop all older frames
                    
                    img = frame.to_ndarray(format="bgr24")
                    success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        continue
                    
                    image_b64 = base64.b64encode(buffer).decode('utf-8')
                    self.gemini_manager.last_frame_by_pc[pc] = image_b64

                    response_data = await self.execute_request(
                        target_processor_id=state["processor_id"],
                        initial_image_b64=image_b64
                    )

                    data_channel = state["data_channel"]
                    if data_channel and data_channel.readyState == "open":
                        try:
                            data_channel.send(json.dumps(response_data))
                        except Exception as e:
                            log_message(f"Failed to send to data channel for peer {pc_id}: {e}", level="ERROR")
                    
                    # NEW: Check if queue cleared to send recovery signal
                    if len(frame_queue) < self.max_frame_queue_size and last_overload_state and state.get("data_channel"):
                        try:
                            state["data_channel"].send(json.dumps({
                                "type": "server_overload",
                                "drop_frames": False,
                                "reason": "Queue cleared"
                            }))
                            log_message(f"Peer {pc_id}: Queue cleared, signaled recovery to client")
                            last_overload_state = False
                        except Exception as e:
                            log_message(f"Peer {pc_id}: Failed to send recovery signal: {e}", level="ERROR")

            except Exception as e:
                log_message(f"Video track for peer {pc_id} ended or failed: {e}", level="WARNING")
                break

    async def audio_track_processor_task(self, pc: RTCPeerConnection, track):
        state = pc_states[pc]
        pc_id = state["id"]
        session_id = f"rtc_{pc_id}"
        self.audio_recorder.start_recording(session_id)
        await self.gemini_manager.start_session(pc, session_id)
        
        try:
            while True:
                audio_frame = await track.recv()
                # resample if needed, then convert to 16-bit PCM
                raw_samples = audio_frame.to_ndarray().astype(np.int16).tobytes()
                audio_chunk_b64 = base64.b64encode(raw_samples).decode('utf-8')
                await self.gemini_manager.handle_audio_chunk(pc, audio_chunk_b64, session_id)
        except Exception as e:
            log_message(f"Audio track for peer {pc_id} ended or failed: {e}", level="WARNING")
        finally:
            log_message(f"Stopping Gemini/Audio session for peer {pc_id}")
            await self.gemini_manager.stop_session(pc)

    # [FIXED] Full implementation of _call_processor added to the class.
    async def _call_processor(self, processor_id: int, processor_config: Dict, input_payload: Dict) -> Dict:
        url = f"http://{processor_config['host']}:{processor_config['port']}/process"
        name = processor_config['name']
        try:
            response = await self.client.post(url, json=input_payload, timeout=90.0)
            response.raise_for_status()
            return response.json()
        except httpx.RequestError as e:
            msg = f"Error calling {name}: {e}"
            log_message(msg, level="ERROR")
            return {"error": msg}
        except Exception as e:
            msg = f"Unexpected error with {name}: {e}"
            log_message(msg, level="ERROR")
            return {"error": msg}

    # [REFACTORED] This method is now stateless and safe for concurrent calls from multiple peers.
    async def execute_request(self, target_processor_id: int, initial_image_b64: Optional[str], initial_point_cloud_json: Optional[Dict] = None) -> Dict:
        # State is kept local to this function call, not in self.
        current_image = initial_image_b64
        current_point_cloud = initial_point_cloud_json
        final_image = initial_image_b64
        final_result = None
        
        processed_nodes = set()

        async def _process_node(proc_id: int) -> bool:
            nonlocal current_image, current_point_cloud, final_image, final_result
            if proc_id in processed_nodes:
                return True

            node_config = PROCESSOR_CONFIG.get(proc_id)
            if not node_config:
                final_result = {"error": f"Processor ID {proc_id} not found."}
                return False
            
            name = node_config["name"]
            for dep_id in node_config.get("dependencies", []):
                if not await _process_node(dep_id):
                    return False
            
            payload = {}
            input_type = node_config.get("expects_input")
            if input_type == "image":
                if not current_image:
                    final_result = {"error": f"{name} expects an image, but none is available."}
                    return False
                payload["image"] = current_image
            elif input_type == "point_cloud":
                if not current_point_cloud:
                    final_result = {"error": f"{name} expects a point cloud, but none is available."}
                    return False
                payload["point_cloud"] = current_point_cloud
            else:
                 final_result = {"error": f"Processor {name} has invalid 'expects_input' type."}
                 return False

            response = await self._call_processor(proc_id, node_config, payload)
            if "error" in response:
                final_result = response
                return False

            if "image" in response:
                current_image = response["image"]
                final_image = current_image
            
            node_payload = response.get("result")
            if proc_id == target_processor_id:
                final_result = node_payload

            # Update point cloud state for the next processor
            if "processed_point_cloud" in response and response["processed_point_cloud"] is not None:
                current_point_cloud = response["processed_point_cloud"]
            elif isinstance(node_payload, dict) and "points" in node_payload:
                current_point_cloud = node_payload

            processed_nodes.add(proc_id)
            return True

        success = await _process_node(target_processor_id)
        if not success and final_result is None:
            final_result = {"error": "Processing chain failed."}

        return {"image": final_image, "text": final_result}


# --- FastAPI Routes & Lifecycle ---

manager = RTCConnectionManager()

@app.get("/processors")
async def get_processors_info():
    processors_list = [{"id": pid, "name": config["name"], **config} for pid, config in PROCESSOR_CONFIG.items()]
    return {"processors": processors_list}

@app.get("/audio/sessions")
async def get_audio_sessions():
    return {"active_sessions": manager.audio_recorder.get_active_sessions()}

@app.post("/offer")
async def offer(request: Request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    pc = RTCPeerConnection()
    manager.add_peer(pc)
    state = pc_states[pc]
    pc_id = state["id"]

    @pc.on("datachannel")
    def on_datachannel(channel: RTCDataChannel):
        log_message(f"DataChannel '{channel.label}' created by client for peer {pc_id}")
        state["data_channel"] = channel

        @channel.on("message")
        def on_message(message: str):
            try:
                data = json.loads(message)
                if "processor" in data:
                    proc_id = data["processor"]
                    state["processor_id"] = int(proc_id) if proc_id is not None else None
                    log_message(f"Peer {pc_id} set to use processor: {state['processor_id']}")
            except Exception as e:
                log_message(f"Error processing data channel message for peer {pc_id}: {e}", level="ERROR")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log_message(f"Peer {pc_id} connection state is {pc.connectionState}")
        if pc.connectionState in ["failed", "closed", "disconnected"]:
            await pc.close()
            await manager.remove_peer(pc)

    @pc.on("track")
    def on_track(track):
        log_message(f"Track {track.kind} received from peer {pc_id}")
        if track.kind == "audio" and not state.get("audio_task"):
            state["audio_task"] = asyncio.create_task(manager.audio_track_processor_task(pc, track))
        elif track.kind == "video" and not state.get("video_task"):
            state["video_task"] = asyncio.create_task(manager.video_track_processor_task(pc, track))

    try:
        await pc.setRemoteDescription(offer)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
    except Exception as e:
        log_message(f"Error during WebRTC handshake for peer {pc_id}: {e}", level="ERROR")
        await manager.remove_peer(pc)
        raise HTTPException(status_code=500, detail=str(e))

    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}

processor_processes = []

# [FIXED] This function is now correctly placed at the module level.
async def start_processor_servers():
    global processor_processes
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    scripts_dir, logs_dir = os.path.join(parent_dir, 'scripts'), os.path.join(parent_dir, 'logs')
    os.makedirs(scripts_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    def is_port_in_use(host: str, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex((host, port)) == 0

    for pid, config in PROCESSOR_CONFIG.items():
        if is_port_in_use(config['host'], config['port']):
            log_message(f"Skipping {config['name']} - Port {config['port']} is in use.", level="WARNING")
            continue
        
        module_path = f"processors.{config['name']}:app"
        script_path = os.path.join(scripts_dir, f"run_{config['name']}.sh")
        log_path = os.path.join(logs_dir, f"{config['name']}.log")
        
        # [FIXED] Script content is dedented to avoid shell syntax errors.
        script_content = textwrap.dedent(f"""\
            #!/bin/bash
            echo "Attempting to start {config['name']}..."
            CONDA_BASE_DIR=$(conda info --base)
            if [ -z "$CONDA_BASE_DIR" ]; then echo "Conda base not found." >&2; exit 1; fi
            source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
            if ! conda activate {config['conda_env']}; then echo "Failed to activate conda env: {config['conda_env']}" >&2; exit 1; fi
            cd "{parent_dir}"
            echo "Starting uvicorn for {module_path}..."
            exec uvicorn {module_path} --host {config['host']} --port {config['port']} --workers 1 --log-level info >> "{log_path}" 2>&1
            """)
        
        try:
            with open(script_path, "w") as f: f.write(script_content)
            os.chmod(script_path, 0o755)
            process = subprocess.Popen([script_path], preexec_fn=os.setsid if os.name != "nt" else None)
            await asyncio.sleep(1.0) # Give it a moment to start or fail

            if process.poll() is not None:
                raise RuntimeError(f"Process for {config['name']} failed on startup.")
            
            processor_processes.append(process)
            log_message(f"Started {config['name']} (PID {process.pid}) on port {config['port']}. Log: {log_path}")
        except Exception as e:
            log_message(f"Error starting {config['name']}: {e}", level="ERROR")
    await asyncio.sleep(2) # Wait for all servers to be ready

@app.on_event("startup")
async def startup_event():
    log_message("Main server starting up...")
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(parent_dir, 'audio_recordings'), exist_ok=True)
    await start_processor_servers()

@app.on_event("shutdown")
async def shutdown_event():
    log_message("Main server shutting down...")
    
    # Close all peer connections gracefully
    await asyncio.gather(*(pc.close() for pc in pcs))
    
    # Terminate processor subprocesses
    for process in processor_processes:
        if process.poll() is None:
            log_message(f"Terminating processor PID {process.pid}...")
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                log_message(f"Error terminating PID {process.pid}: {e}", level="WARNING")
                if process.poll() is None: process.kill()

    # Clean up temporary scripts
    scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
    if os.path.exists(scripts_dir):
        for filename in os.listdir(scripts_dir):
            os.remove(os.path.join(scripts_dir, filename))
    
    # Clean up temporary scripts
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    if os.path.exists(logs_dir):
        for filename in os.listdir(logs_dir):
            os.remove(os.path.join(logs_dir, filename))
    
    log_message("Shutdown complete.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)