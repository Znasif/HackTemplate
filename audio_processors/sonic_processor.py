import asyncio
from asyncio import Queue
import json
import base64
import os
import time
import uuid
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

def load_config(path: str) -> dict:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, path)
    if not os.path.exists(config_path):
        config_path = path

    with open(config_path, 'r') as f:
        config = json.load(f)
    return {
        int(k): v for k, v in config.items() if v.get("enabled", True)
    }

PROCESSOR_CONFIG = load_config("../processor_config.json")

def log_message(message: str, level: str = "INFO"):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")

class SonicStreamManager:
    """Manages bidirectional streaming with Amazon Nova Sonic for audio processing and tool calling."""
    
    def __init__(self, audio_recorder, connection_manager, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
        """Initialize the Sonic Stream Manager.
        
        Args:
            audio_recorder: Instance of AudioStreamRecorder to store audio chunks.
            connection_manager: Reference to ConnectionManager for processor execution
            model_id: The Amazon Nova Sonic model ID
            region: AWS region
        """
        self.audio_recorder = audio_recorder
        self.connection_manager = connection_manager
        self.model_id = model_id
        self.region = region
        self.active_sessions = {}  # WebSocket -> session info
        self.last_frame_by_websocket = {}  # Store last image frame from each websocket
        
        load_dotenv()
        
        # Initialize Bedrock client
        self.bedrock_client = None
        self._initialize_client()
        
        # Create processor descriptions for tool calling
        self.processor_descriptions = self._generate_processor_descriptions()
        
    def _initialize_client(self):
        """Initialize the Bedrock client."""
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
    
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
        """Create system instruction for Sonic including processor information"""
        return f"""You are a helpful AI assistant with access to various image and video processing capabilities.

When a user asks you to perform a task related to image or video processing, you should:
1. Analyze their request to understand what they want to accomplish
2. Check if any of the available processors match their needs
3. If a matching processor is found, use the execute_processor function to run it
4. If no processor matches, inform the user that the requested capability is not available

Available processors:
{self.processor_descriptions}

You can also use the list_available_processors function if the user asks what processors are available.
Keep your responses concise and conversational. Do not ask clarifying questions unless absolutely necessary."""

    def log_message(self, message: str, level: str = "INFO"):
        """Log messages with timestamp and level."""
        log_message(f"SonicStreamManager: {message}", level)

    async def start_session(self, websocket, session_id: str):
        """Start a Sonic streaming session for a WebSocket connection.
        
        Args:
            websocket: WebSocket connection.
            session_id: Unique session ID for audio streaming.
        """
        try:
            # Initialize session data
            self.active_sessions[websocket] = {
                'session_id': session_id,
                'audio_in_queue': Queue(),
                'out_queue': Queue(maxsize=5),
                'sonic_stream': None,
                'tasks': [],
                'sonic_audio_responses': [],
                'websocket': websocket,
                'is_active': True,
                'prompt_name': str(uuid.uuid4()),
                'content_name': str(uuid.uuid4()),
                'audio_content_name': str(uuid.uuid4()),
                'barge_in': False
            }
            
            session_data = self.active_sessions[websocket]
            
            # Initialize the bidirectional stream
            session_data['sonic_stream'] = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )
            
            # Send initialization events
            await self._initialize_stream(session_data)
            
            # Start concurrent tasks
            async with asyncio.TaskGroup() as tg:
                session_data['tasks'] = [
                    tg.create_task(self.send_audio_to_sonic(websocket)),
                    tg.create_task(self.receive_from_sonic(websocket)),
                    tg.create_task(self.play_audio_to_client(websocket)),
                ]
                self.log_message(f"Started Sonic streaming session for {session_id}")
                await asyncio.Event().wait()
                    
        except Exception as e:
            self.log_message(f"Error starting session {session_id}: {e}", level="ERROR")
            await websocket.send_text(json.dumps({"error": f"Failed to start Sonic streaming: {str(e)}"}))
        finally:
            await self.cleanup_session(websocket)

    async def _initialize_stream(self, session_data: Dict[str, Any]):
        """Initialize the stream with start events and system prompt"""
        stream = session_data['sonic_stream']
        prompt_name = session_data['prompt_name']
        content_name = session_data['content_name']
        audio_content_name = session_data['audio_content_name']
        
        # Event templates
        start_session_event = {
            "event": {
                "sessionStart": {
                    "inferenceConfiguration": {
                        "maxTokens": 1024,
                        "topP": 0.9,
                        "temperature": 0.7
                    }
                }
            }
        }
        
        # Prompt start with tools
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": prompt_name,
                    "textOutputConfiguration": {
                        "mediaType": "text/plain"
                    },
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {
                        "mediaType": "application/json"
                    },
                    "toolConfiguration": {
                        "tools": [
                            {
                                "toolSpec": {
                                    "name": "execute_processor",
                                    "description": "Execute a specific processor to handle image/video processing tasks",
                                    "inputSchema": {
                                        "json": json.dumps({
                                            "type": "object",
                                            "properties": {
                                                "processor_id": {
                                                    "type": "integer",
                                                    "description": "The ID of the processor to execute"
                                                },
                                                "reason": {
                                                    "type": "string",
                                                    "description": "Brief explanation of why this processor was chosen"
                                                }
                                            },
                                            "required": ["processor_id", "reason"]
                                        })
                                    }
                                }
                            },
                            {
                                "toolSpec": {
                                    "name": "list_available_processors",
                                    "description": "List all available processors with their descriptions",
                                    "inputSchema": {
                                        "json": json.dumps({
                                            "type": "object",
                                            "properties": {},
                                            "required": []
                                        })
                                    }
                                }
                            }
                        ]
                    }
                }
            }
        }
        
        # System message
        system_content_start = {
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "role": "SYSTEM",
                    "interactive": True,
                    "textInputConfiguration": {
                        "mediaType": "text/plain"
                    }
                }
            }
        }
        
        system_text_input = {
            "event": {
                "textInput": {
                    "promptName": prompt_name,
                    "contentName": content_name,
                    "content": self._create_system_instruction()
                }
            }
        }
        
        system_content_end = {
            "event": {
                "contentEnd": {
                    "promptName": prompt_name,
                    "contentName": content_name
                }
            }
        }
        
        # Audio content start
        audio_content_start = {
            "event": {
                "contentStart": {
                    "promptName": prompt_name,
                    "contentName": audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }
                }
            }
        }
        
        # Send initialization events in sequence
        init_events = [
            start_session_event,
            prompt_start_event,
            system_content_start,
            system_text_input,
            system_content_end,
            audio_content_start
        ]
        
        for event in init_events:
            await self._send_event(stream, event)
            await asyncio.sleep(0.05)  # Small delay between events

    async def _send_event(self, stream, event_dict: Dict):
        """Send an event to the Sonic stream."""
        event_json = json.dumps(event_dict)
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await stream.input_stream.send(event)
            self.log_message(f"Sent event: {list(event_dict['event'].keys())[0]}")
        except Exception as e:
            self.log_message(f"Error sending event: {str(e)}", level="ERROR")

    async def send_audio_to_sonic(self, websocket):
        """Send audio from queue to Sonic"""
        session_data = self.active_sessions.get(websocket)
        if not session_data:
            return
            
        out_queue = session_data['out_queue']
        stream = session_data['sonic_stream']
        prompt_name = session_data['prompt_name']
        audio_content_name = session_data['audio_content_name']
        
        while session_data['is_active']:
            try:
                audio_b64 = await out_queue.get()
                if audio_b64 is None:
                    break
                    
                # Create audio event
                audio_event = {
                    "event": {
                        "audioInput": {
                            "promptName": prompt_name,
                            "contentName": audio_content_name,
                            "content": audio_b64
                        }
                    }
                }
                
                await self._send_event(stream, audio_event)
                
            except Exception as e:
                self.log_message(f"Error sending audio to Sonic: {e}", level="ERROR")

    async def receive_from_sonic(self, websocket):
        """Receive and process responses from Sonic"""
        session_data = self.active_sessions.get(websocket)
        if not session_data:
            return
            
        audio_in_queue = session_data['audio_in_queue']
        stream = session_data['sonic_stream']
        
        while session_data['is_active']:
            try:
                output = await stream.await_output()
                result = await output[1].receive()
                
                if result.value and result.value.bytes_:
                    response_data = result.value.bytes_.decode('utf-8')
                    json_data = json.loads(response_data)
                    
                    # Handle different response types
                    if 'event' in json_data:
                        await self._handle_sonic_event(websocket, json_data['event'])
                        
            except StopAsyncIteration:
                break
            except Exception as e:
                self.log_message(f"Error receiving from Sonic: {e}", level="ERROR")
                break

    async def _handle_sonic_event(self, websocket, event: Dict[str, Any]):
        """Handle different types of events from Sonic"""
        session_data = self.active_sessions.get(websocket)
        if not session_data:
            return
            
        audio_in_queue = session_data['audio_in_queue']
        
        try:
            if 'textOutput' in event:
                # Handle text responses
                text_content = event['textOutput']['content']
                
                # Check for barge-in
                if '{ "interrupted" : true }' in text_content:
                    self.log_message("Barge-in detected")
                    session_data['barge_in'] = True
                    # Clear audio queue
                    while not audio_in_queue.empty():
                        try:
                            audio_in_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                else:
                    # Send text to client if it's not a barge-in signal
                    if text_content.strip():
                        await websocket.send_text(json.dumps({
                            "type": "sonic_text_response",
                            "text": text_content
                        }))
                    
            elif 'audioOutput' in event:
                # Handle audio responses - only if no barge-in
                if not session_data.get('barge_in', False):
                    audio_content = event['audioOutput']['content']
                    audio_bytes = base64.b64decode(audio_content)
                    session_data['sonic_audio_responses'].append(audio_bytes)
                    await audio_in_queue.put(audio_bytes)
                
            elif 'toolUse' in event:
                # Handle tool use
                tool_content = event['toolUse']
                tool_name = tool_content.get('toolName', '')
                tool_use_id = tool_content.get('toolUseId', '')
                self.log_message(f"Tool use detected: {tool_name}, ID: {tool_use_id}")
                
                # Store tool info for processing
                session_data['pending_tool_info'] = {
                    'name': tool_name,
                    'content': tool_content,
                    'id': tool_use_id
                }
                
            elif 'contentEnd' in event and event.get('contentEnd', {}).get('type') == 'TOOL':
                # Process the tool when content ends
                if 'pending_tool_info' in session_data:
                    tool_info = session_data['pending_tool_info']
                    await self._handle_tool_request(websocket, tool_info['name'], tool_info['content'], tool_info['id'])
                    del session_data['pending_tool_info']
                    
            elif 'contentStart' in event:
                # Reset barge-in flag when new content starts
                session_data['barge_in'] = False
                
        except Exception as e:
            self.log_message(f"Error handling Sonic event: {e}", level="ERROR")

    async def _handle_tool_request(self, websocket, tool_name: str, tool_content: Dict, tool_use_id: str):
        """Handle tool requests from Sonic"""
        session_data = self.active_sessions.get(websocket)
        if not session_data:
            return
            
        try:
            self.log_message(f"Processing tool: {tool_name}")
            
            if tool_name == "execute_processor":
                # Extract processor ID and reason from tool content
                processor_id = tool_content.get("processor_id")
                reason = tool_content.get("reason", "No reason provided")
                
                # Send processor execution request to client
                await websocket.send_text(json.dumps({
                    "text": "set_processor",
                    "processor_id": processor_id,
                    "reason": reason
                }))
                    
            elif tool_name == "list_available_processors":
                # List available processors
                processors_info = []
                for proc_id, config in PROCESSOR_CONFIG.items():
                    processors_info.append({
                        "id": proc_id,
                        "name": config["name"],
                        "description": config.get("description", "No description available")
                    })
                
                # Send to client
                await websocket.send_text(json.dumps({
                    "type": "available_processors",
                    "processors": processors_info
                }))
                
                # Send result back to Sonic
                await self._send_tool_result(websocket, tool_use_id, {
                    "status": "success",
                    "processors": processors_info
                })
                
        except Exception as e:
            self.log_message(f"Error handling tool request: {e}", level="ERROR")
            await self._send_tool_result(websocket, tool_use_id, {
                "status": "error",
                "message": str(e)
            })

    async def _send_tool_result(self, websocket, tool_use_id: str, result: Dict):
        """Send tool execution result back to Sonic"""
        session_data = self.active_sessions.get(websocket)
        if not session_data:
            return
            
        stream = session_data['sonic_stream']
        prompt_name = session_data['prompt_name']
        content_name = str(uuid.uuid4())
        
        try:
            # Send tool result sequence
            tool_start = {
                "event": {
                    "contentStart": {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": tool_use_id,
                            "type": "TEXT",
                            "textInputConfiguration": {
                                "mediaType": "text/plain"
                            }
                        }
                    }
                }
            }
            
            tool_result_event = {
                "event": {
                    "toolResult": {
                        "promptName": prompt_name,
                        "contentName": content_name,
                        "content": json.dumps(result)
                    }
                }
            }
            
            tool_end = {
                "event": {
                    "contentEnd": {
                        "promptName": prompt_name,
                        "contentName": content_name
                    }
                }
            }
            
            # Send events in sequence
            await self._send_event(stream, tool_start)
            await self._send_event(stream, tool_result_event)
            await self._send_event(stream, tool_end)
            
        except Exception as e:
            self.log_message(f"Error sending tool result: {e}", level="ERROR")

    async def play_audio_to_client(self, websocket):
        """Stream audio responses to the client"""
        session_data = self.active_sessions.get(websocket)
        if not session_data:
            return
            
        audio_in_queue = session_data['audio_in_queue']
        
        while session_data['is_active']:
            try:
                audio_bytes = await audio_in_queue.get()
                
                # Convert to base64 and send to client
                chunk_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send_text(json.dumps({
                    "type": "sonic_audio_response",
                    "audio_chunk": chunk_b64
                }))
                
                self.log_message(f"Streaming audio to client: {len(audio_bytes)} bytes")
                
            except Exception as e:
                self.log_message(f"Error streaming audio to client: {e}", level="ERROR")

    async def handle_audio_chunk(self, websocket, audio_chunk_b64: str, session_id: str):
        """Handle incoming audio chunk from client"""
        try:
            # Decode audio
            audio_bytes = base64.b64decode(audio_chunk_b64)
            
            # Save to recorder
            success = self.audio_recorder.add_audio_chunk(session_id, audio_bytes)
            if not success:
                await websocket.send_text(json.dumps({
                    "error": "Failed to save audio chunk"
                }))
                return
            
            # Add to Sonic queue (Sonic expects base64 encoded audio)
            if websocket in self.active_sessions:
                session_data = self.active_sessions[websocket]
                await session_data['out_queue'].put(audio_chunk_b64)
                
        except Exception as e:
            self.log_message(f"Error handling audio chunk: {e}", level="ERROR")
            await websocket.send_text(json.dumps({
                "error": f"Audio chunk processing error: {str(e)}"
            }))

    async def stop_session(self, websocket):
        """Stop the streaming session and return results"""
        session_data = self.active_sessions.get(websocket)
        if not session_data:
            return None
            
        session_id = session_data['session_id']
        
        try:
            # Send end events to Sonic
            if session_data['sonic_stream']:
                prompt_name = session_data['prompt_name']
                audio_content_name = session_data['audio_content_name']
                
                audio_end = {
                    "event": {
                        "contentEnd": {
                            "promptName": prompt_name,
                            "contentName": audio_content_name
                        }
                    }
                }
                
                prompt_end = {
                    "event": {
                        "promptEnd": {
                            "promptName": prompt_name
                        }
                    }
                }
                
                session_end = {
                    "event": {
                        "sessionEnd": {}
                    }
                }
                
                await self._send_event(session_data['sonic_stream'], audio_end)
                await self._send_event(session_data['sonic_stream'], prompt_end)
                await self._send_event(session_data['sonic_stream'], session_end)
            
            # Get recording results
            recording_result = self.audio_recorder.stop_recording_and_convert(session_id)
            
            result = {
                'recording': recording_result,
                'sonic_audio_responses': session_data.get('sonic_audio_responses', [])
            }
            
            return result
            
        except Exception as e:
            self.log_message(f"Error stopping session: {e}", level="ERROR")
            return None
        finally:
            await self.cleanup_session(websocket)

    async def cleanup_session(self, websocket):
        """Clean up session resources"""
        if websocket in self.active_sessions:
            session_data = self.active_sessions[websocket]
            
            # Mark as inactive
            session_data['is_active'] = False
            
            # Cancel tasks
            for task in session_data.get('tasks', []):
                task.cancel()
            
            # Close stream
            if session_data.get('sonic_stream'):
                try:
                    await session_data['sonic_stream'].input_stream.close()
                except:
                    pass
            
            # Clean up recorder
            if session_data.get('session_id'):
                self.audio_recorder.cleanup_session(session_data['session_id'])
            
            del self.active_sessions[websocket]
            self.log_message(f"Cleaned up Sonic session")
            
        # Clean up last frame
        if websocket in self.last_frame_by_websocket:
            del self.last_frame_by_websocket[websocket]