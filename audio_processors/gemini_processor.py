import asyncio
from asyncio import Queue
import json
import base64
import os
import time
from dotenv import load_dotenv
import re

from google import genai
from google.genai import types

# Gemini configuration constants
MODEL = "models/gemini-2.0-flash-live-001"

def load_config(path: str) -> dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, path) # Assumes 'path' is relative like "processor_config.json"
    if not os.path.exists(config_path):
        # Fallback for cases where path might be already absolute or relative to CWD
        config_path = path

    with open(config_path, 'r') as f:
        config = json.load(f)
    return {
        int(k): v for k, v in config.items() if v.get("enabled", True)
    }

PROCESSOR_CONFIG = load_config("../processor_config.json")

def log_message(message: str, level: str = "INFO"):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")

class GeminiFlashStreamManager:
    def __init__(self, audio_recorder, connection_manager):
        """Initialize the Gemini Flash Stream Manager.
        
        Args:
            audio_recorder: Instance of AudioStreamRecorder to store audio chunks.
            connection_manager: Reference to ConnectionManager for processor execution
        """
        self.audio_recorder = audio_recorder
        self.connection_manager = connection_manager
        self.active_sessions = {}  # WebSocket -> session info
        self.last_frame_by_websocket = {}  # Store last image frame from each websocket
        load_dotenv()
        self.gemini_client = genai.Client(
            http_options={"api_version": "v1beta"},
            api_key=os.environ.get("GEMINI_API_KEY"),
        )
        
        # Create processor descriptions for tool calling
        self.processor_descriptions = self._generate_processor_descriptions()
        
        # Define tool functions
        self.tools = [types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name="execute_processor",
                description="Execute a specific processor to handle image/video processing tasks",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "processor_id": types.Schema(
                            type="INTEGER",
                            description="The ID of the processor to execute"
                        ),
                        "reason": types.Schema(
                            type="STRING",
                            description="Brief explanation of why this processor was chosen"
                        )
                    },
                    required=["processor_id", "reason"]
                )
            ),
            types.FunctionDeclaration(
                name="list_available_processors",
                description="List all available processors with their descriptions",
                parameters=types.Schema(type="OBJECT", properties={})
            )
        ])]
        
        # Update Gemini configuration with tools
        self.config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            media_resolution="MEDIA_RESOLUTION_MEDIUM",
            speech_config=types.SpeechConfig(
                language_code="en-US",
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            ),
            context_window_compression=types.ContextWindowCompressionConfig(
                trigger_tokens=25600,
                sliding_window=types.SlidingWindow(target_tokens=12800),
            ),
            tools=self.tools,
            system_instruction=self._create_system_instruction()
        )
    
    def _generate_processor_descriptions(self) -> str:
        """Generate a description of all available processors"""
        descriptions = []
        for proc_id, config in PROCESSOR_CONFIG.items():
            desc = f"Processor {proc_id}: {config['name']}"
            if 'description' in config:
                desc += f" - {config['description']}"
            if 'dependencies' in config and config['dependencies']:
                desc += f" (depends on: {', '.join(map(str, config['dependencies']))})"
            descriptions.append(desc)
        return "\n".join(descriptions)
    
    def _create_system_instruction(self) -> str:
        """Create system instruction for Gemini including processor information"""
        return f"""You are a helpful AI assistant with access to various image and video processing capabilities.

When a user asks you to perform a task related to image or video processing, you should:
1. Analyze their request to understand what they want to accomplish
2. Check if any of the available processors match their needs
3. If a matching processor is found, use the execute_processor function to run it
4. If no processor matches, inform the user that the requested capability is not available

Available processors:
{self.processor_descriptions}

You can also use the list_available_processors function if the user asks what processors are available.
Do not ask clarifying questions and keep your responses short."""
        
    def log_message(self, message: str, level: str = "INFO"):
        """Log messages with timestamp and level."""
        # Using the global log_message function from this module
        log_message(f"GeminiFlashStreamManager: {message}", level)

    async def start_session(self, websocket, session_id: str): # WebSocket type hint removed for now
        """Start a Gemini streaming session for a WebSocket connection.
        
        Args:
            websocket: WebSocket connection.
            session_id: Unique session ID for audio streaming.
        """
        try:
            self.active_sessions[websocket] = {
                'session_id': session_id,
                'audio_in_queue': Queue(),
                'out_queue': Queue(maxsize=5),
                'gemini_session': None,
                'tasks': [],
                'gemini_audio_responses': [],
                'websocket': websocket,
            }
            
            session_data = self.active_sessions[websocket]
            
            async with self.gemini_client.aio.live.connect(model=MODEL, config=self.config) as gemini_session:
                session_data['gemini_session'] = gemini_session
                
                async with asyncio.TaskGroup() as tg:
                    session_data['tasks'] = [
                        tg.create_task(self.send_audio_to_gemini(websocket)),
                        tg.create_task(self.receive_audio_from_gemini(websocket)),
                        tg.create_task(self.play_audio_to_client(websocket)),
                    ]
                    self.log_message(f"Started Gemini streaming session for {session_id}")
                    await asyncio.Event().wait()
                    
        except Exception as e:
            self.log_message(f"Error starting session {session_id}: {e}", level="ERROR")
            await websocket.send_text(json.dumps({"error": f"Failed to start Gemini streaming: {str(e)}"}))
        finally:
            await self.cleanup_session(websocket)

    async def send_audio_to_gemini(self, websocket):
        session_data = self.active_sessions.get(websocket)
        if not session_data or not session_data['gemini_session']: return
        out_queue = session_data['out_queue']
        while True:
            try:
                msg = await out_queue.get()
                if msg is None: break
                await session_data['gemini_session'].send(input=msg)
                self.log_message(f"Sent audio chunk to Gemini")
            except Exception as e:
                self.log_message(f"Error sending audio to Gemini: {e}", level="ERROR")

    async def receive_audio_from_gemini(self, websocket):
        session_data = self.active_sessions.get(websocket)
        if not session_data or not session_data['gemini_session']: return
        audio_in_queue = session_data['audio_in_queue']
        while True:
            try:
                turn = session_data['gemini_session'].receive()
                async for response in turn:
                    # response_info = {
                    #     'data': bool(response.data) if hasattr(response, 'data') else None,
                    #     'text': response.text if hasattr(response, 'text') else None,
                    #     'server_content': bool(response.server_content) if hasattr(response, 'server_content') else None,
                    #     'tool_call': bool(response.tool_call) if hasattr(response, 'tool_call') else None,
                    #     'setup_complete': response.setup_complete if hasattr(response, 'setup_complete') else None,
                    #     'usage_metadata': bool(response.usage_metadata) if hasattr(response, 'usage_metadata') else None,
                    # }
                    # # Remove None values
                    # response_info = {k: v for k, v in response_info.items() if v is not None}
                    # self.log_message(f"Full Gemini response: {response}", level="DEBUG")
                    if data := response.data:
                        session_data['gemini_audio_responses'].append(data)
                        await audio_in_queue.put(data)
                        self.log_message(f"Received audio chunk from Gemini")
                        continue
                    if hasattr(response, 'server_content') and response.server_content:
                        await self.handle_server_content(websocket, response.server_content)
                        continue
                    # Fix: Check for tool_call.function_calls instead of function_calls
                    if hasattr(response, 'tool_call') and response.tool_call and response.tool_call.function_calls:
                        for func_call in response.tool_call.function_calls:
                            await self.handle_tool_call(websocket, func_call)
                        continue
                    self.log_message(f"Unhandled response part: {response}", level="WARNING")
                self.log_message(f"Received audio back from Gemini, clearing audio queue")
                while not audio_in_queue.empty():
                    audio_in_queue.get_nowait()
            except Exception as e:
                self.log_message(f"Error receiving audio from Gemini: {e}", level="ERROR")

    async def handle_server_content(self, websocket, server_content):
        try:
            if model_turn := server_content.model_turn:
                for part in model_turn.parts:
                    if executable_code := part.executable_code:
                        self.log_message(f"Executable code: {executable_code.code}", level="DEBUG")
                        # Let the separate tool call handler deal with function calls
                        # Just log the executable code for debugging
                        
                    if code_execution_result := part.code_execution_result:
                        self.log_message(f"Code execution result: {code_execution_result.output}", level="DEBUG")
                        session_data = self.active_sessions.get(websocket)
                        if session_data and session_data['gemini_session']:
                            await session_data['gemini_session'].send(
                                input=f"Code execution result: {code_execution_result.output}",
                                end_of_turn=True
                            )
        except Exception as e:
            self.log_message(f"Error handling server content: {e}", level="ERROR")
            await websocket.send_text(json.dumps({"error": f"Server content processing error: {str(e)}"}))

    async def handle_tool_call(self, websocket, func_call):
        try:
            func_name = func_call.name
            func_args = func_call.args
            self.log_message(f"Handling tool call: {func_name} with args: {func_args}", level="DEBUG")
            
            if func_name == "execute_processor":
                processor_id = func_args.get("processor_id")
                reason = func_args.get("reason", "No reason provided")
                self.log_message(f"Executing processor: processor_id={processor_id}, reason={reason}")
                
                # Send the processor change to the client
                await websocket.send_text(json.dumps({
                    "text": "set_processor", 
                    "processor_id": processor_id, 
                    "reason": reason
                }))
                
                # Send tool response back to Gemini
                session_data = self.active_sessions.get(websocket)
                if session_data and session_data['gemini_session']:
                    function_response = types.FunctionResponse(
                        id=func_call.id,
                        name=func_name,
                        response={"result": f"Successfully set processor to {processor_id}"}
                    )
                    await session_data['gemini_session'].send_tool_response(
                        function_responses=[function_response]
                    )
                    
            elif func_name == "list_available_processors":
                processors_info = [
                    {
                        "id": proc_id, 
                        "name": config["name"], 
                        "description": config.get("description", "No description available"), 
                        "dependencies": config.get("dependencies", []), 
                        "expects_input": config.get("expects_input", "unknown")
                    } 
                    for proc_id, config in PROCESSOR_CONFIG.items()
                ]
                
                await websocket.send_text(json.dumps({
                    "type": "available_processors", 
                    "processors": processors_info
                }))
                
                # Send tool response back to Gemini
                session_data = self.active_sessions.get(websocket)
                if session_data and session_data['gemini_session']:
                    function_response = types.FunctionResponse(
                        id=func_call.id,
                        name=func_name,
                        response={"result": f"Listed {len(processors_info)} processors"}
                    )
                    await session_data['gemini_session'].send_tool_response(
                        function_responses=[function_response]
                    )
                    
            else:
                self.log_message(f"Unknown function call: {func_name}", level="ERROR")
                await websocket.send_text(json.dumps({"error": f"Unknown function call: {func_name}"}))
                
        except Exception as e:
            self.log_message(f"Error handling tool call: {e}", level="ERROR")
            await websocket.send_text(json.dumps({"error": f"Tool execution error: {str(e)}"}))

    async def play_audio_to_client(self, websocket):
        session_data = self.active_sessions.get(websocket)
        if not session_data: return
        audio_in_queue = session_data['audio_in_queue']
        while True:
            try:
                bytestream = await audio_in_queue.get()
                chunk_b64 = base64.b64encode(bytestream).decode('utf-8')
                await websocket.send_text(json.dumps({"type": "gemini_audio_response", "audio_chunk": chunk_b64}))
                self.log_message(f"Streaming back to client: {len(bytestream)} bytes")
            except Exception as e:
                self.log_message(f"Error streaming audio to client: {e}", level="ERROR")

    async def handle_audio_chunk(self, websocket, audio_chunk_b64: str, session_id: str):
        try:
            audio_bytes = base64.b64decode(audio_chunk_b64)
            success = self.audio_recorder.add_audio_chunk(session_id, audio_bytes)
            if not success:
                await websocket.send_text(json.dumps({"error": "Failed to save audio chunk in recorder"}))
                return
            if websocket in self.active_sessions:
                session_data = self.active_sessions[websocket]
                await session_data['out_queue'].put({"data": audio_bytes, "mime_type": "audio/pcm"})
        except Exception as e:
            self.log_message(f"Error handling audio chunk: {e}", level="ERROR")
            await websocket.send_text(json.dumps({"error": f"Audio chunk processing error: {str(e)}"}))

    async def stop_session(self, websocket):
        session_id = self.active_sessions.get(websocket, {}).get('session_id')
        if session_id:
            recording_result = self.audio_recorder.stop_recording_and_convert(session_id)
            session_data = self.active_sessions.get(websocket, {})
            result = {'recording': recording_result, 'gemini_audio_responses': session_data.get('gemini_audio_responses', [])}
            await self.cleanup_session(websocket)
            return result
        return None

    async def cleanup_session(self, websocket):
        if websocket in self.active_sessions:
            session_data = self.active_sessions[websocket]
            for task in session_data['tasks']:
                task.cancel()
            session_data['gemini_session'] = None
            if session_data['session_id']:
                self.audio_recorder.cleanup_session(session_data['session_id'])
            del self.active_sessions[websocket]
            self.log_message(f"Cleaned up Gemini session for {session_data['session_id']}")
        if websocket in self.last_frame_by_websocket:
            del self.last_frame_by_websocket[websocket]