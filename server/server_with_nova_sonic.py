import subprocess
import httpx
import asyncio
import json
import base64
import socket
import os
import signal
from dotenv import load_dotenv
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Union, Any
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

# Import Nova Sonic components
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

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
PROCESSOR_CONFIG = {
    0: {
        "host": "127.0.0.1", "port": 8001,
        "name": "base_processor", "conda_env": "whatsai2", "dependencies": [],
        "expects_input": "image",
        "description": "Basic image processor for general analysis"
    },
}

def log_message(message: str, level: str = "INFO"):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")

class NovaSonicStreamManager:
    """Manages Nova Sonic integration for audio processing and processor selection"""
    
    def __init__(self, websocket: WebSocket, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
        self.websocket = websocket
        self.model_id = model_id
        self.region = region
        self.bedrock_client = None
        self.stream_response = None
        self.is_active = False
        self.selected_processor_id = 0  # Default processor
        self.prompt_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        
        # Initialize client
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the Bedrock client."""
        load_dotenv()
        os.environ['AWS_DEFAULT_REGION'] = "us-east-1"
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
    
    def _create_processor_selection_tool(self):
        """Create the tool configuration for processor selection"""
        processor_descriptions = "\n".join([
            f"- Processor {pid}: {cfg['name']} - {cfg.get('description', 'No description')}"
            for pid, cfg in PROCESSOR_CONFIG.items()
        ])
        
        return {
            "toolSpec": {
                "name": "selectProcessor",
                "description": f"Select the appropriate image processor based on the user's request. Available processors:\n{processor_descriptions}",
                "inputSchema": {
                    "json": json.dumps({
                        "type": "object",
                        "properties": {
                            "processorId": {
                                "type": "integer",
                                "description": "The ID of the processor to use",
                                "enum": list(PROCESSOR_CONFIG.keys())
                            },
                            "reason": {
                                "type": "string",
                                "description": "Brief reason for selecting this processor"
                            }
                        },
                        "required": ["processorId", "reason"]
                    })
                }
            }
        }
    
    def _create_prompt_start_event(self):
        """Create a promptStart event with processor selection tool"""
        prompt_start_event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
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
                        "tools": [self._create_processor_selection_tool()]
                    }
                }
            }
        }
        return json.dumps(prompt_start_event)
    
    async def initialize_stream(self):
        """Initialize the bidirectional stream with Bedrock."""
        try:
            self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )
            self.is_active = True
            
            # System prompt
            system_prompt = """You are an AI assistant helping users process images using various processors. 
            Listen to the user's request and select the most appropriate processor. 
            After selecting a processor, briefly confirm your choice in a natural, conversational way.
            Keep your responses concise and friendly."""
            
            # Send initialization events
            start_session = json.dumps({"event": {"sessionStart": {"inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}}}})
            prompt_start = self._create_prompt_start_event()
            
            # System message
            text_start = json.dumps({
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                        "type": "TEXT",
                        "role": "SYSTEM",
                        "interactive": True,
                        "textInputConfiguration": {"mediaType": "text/plain"}
                    }
                }
            })
            
            text_input = json.dumps({
                "event": {
                    "textInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name,
                        "content": system_prompt
                    }
                }
            })
            
            text_end = json.dumps({
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.content_name
                    }
                }
            })
            
            # Send all initialization events
            for event in [start_session, prompt_start, text_start, text_input, text_end]:
                await self.send_raw_event(event)
                await asyncio.sleep(0.1)
            
            # Start processing responses
            asyncio.create_task(self._process_responses())
            
            log_message("Nova Sonic stream initialized successfully")
            return True
            
        except Exception as e:
            log_message(f"Failed to initialize Nova Sonic stream: {str(e)}", level="ERROR")
            self.is_active = False
            return False
    
    async def send_raw_event(self, event_json):
        """Send a raw event JSON to the Bedrock stream."""
        if not self.stream_response or not self.is_active:
            return
        
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        
        try:
            await self.stream_response.input_stream.send(event)
        except Exception as e:
            log_message(f"Error sending event to Nova Sonic: {str(e)}", level="ERROR")
    
    async def process_audio_chunk(self, audio_chunk_b64: str):
        """Process an audio chunk through Nova Sonic"""
        if not self.is_active:
            return
        
        # Send audio content start if this is the first chunk
        if not hasattr(self, '_audio_started'):
            content_start = json.dumps({
                "event": {
                    "contentStart": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": {
                            "mediaType": "audio/lpcm",
                            "sampleRateHertz": 16000,
                            "sampleSizeBits": 16,
                            "channelCount": 1,
                            "audioType": "SPEECH",
                            "encoding": "base64"
                        }
                    }
                }
            })
            await self.send_raw_event(content_start)
            self._audio_started = True
        
        # Send audio chunk
        audio_event = json.dumps({
            "event": {
                "audioInput": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "content": audio_chunk_b64
                }
            }
        })
        await self.send_raw_event(audio_event)
    
    async def end_audio_input(self):
        """Signal end of audio input"""
        if hasattr(self, '_audio_started') and self._audio_started:
            content_end = json.dumps({
                "event": {
                    "contentEnd": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name
                    }
                }
            })
            await self.send_raw_event(content_end)
            self._audio_started = False
            
            # End the prompt to trigger response
            prompt_end = json.dumps({
                "event": {
                    "promptEnd": {
                        "promptName": self.prompt_name
                    }
                }
            })
            await self.send_raw_event(prompt_end)
    
    async def _process_responses(self):
        """Process responses from Nova Sonic"""
        try:
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                    
                    if result.value and result.value.bytes_:
                        response_data = result.value.bytes_.decode('utf-8')
                        json_data = json.loads(response_data)
                        
                        if 'event' in json_data:
                            # Handle audio output
                            if 'audioOutput' in json_data['event']:
                                audio_content = json_data['event']['audioOutput']['content']
                                # Send audio directly to client
                                await self.websocket.send_text(json.dumps({
                                    "type": "audio_response",
                                    "audio": audio_content
                                }))
                            
                            # Handle tool use (processor selection)
                            elif 'toolUse' in json_data['event']:
                                tool_content = json_data['event']['toolUse']
                                if tool_content['toolName'] == 'selectProcessor':
                                    content_data = json.loads(tool_content['content'])
                                    self.selected_processor_id = content_data['processorId']
                                    log_message(f"Processor selected: {self.selected_processor_id} - {content_data.get('reason', '')}")
                                    
                                    # Send processor selection to client
                                    await self.websocket.send_text(json.dumps({
                                        "type": "processor_selected",
                                        "processor_id": self.selected_processor_id,
                                        "reason": content_data.get('reason', '')
                                    }))
                            
                            # Handle completion
                            elif 'completionEnd' in json_data['event']:
                                log_message("Nova Sonic response completed")
                
                except StopAsyncIteration:
                    break
                except Exception as e:
                    log_message(f"Error processing Nova Sonic response: {e}", level="ERROR")
                    break
                    
        except Exception as e:
            log_message(f"Nova Sonic response processing error: {e}", level="ERROR")
        finally:
            self.is_active = False
    
    async def close(self):
        """Close the Nova Sonic stream"""
        if self.is_active:
            self.is_active = False
            
            # Send session end event
            session_end = json.dumps({"event": {"sessionEnd": {}}})
            await self.send_raw_event(session_end)
            
            if self.stream_response:
                await self.stream_response.input_stream.close()

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client = httpx.AsyncClient(timeout=60.0)
        self.nova_sonic_sessions: Dict[WebSocket, NovaSonicStreamManager] = {}
        
        # Instance variables for image processing
        self.current_image_for_processing: Optional[str] = None
        self.final_image_to_client: Optional[str] = None
        self.final_result_to_client: Union[str, Dict, None] = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        
        # Clean up Nova Sonic session
        if websocket in self.nova_sonic_sessions:
            nova_session = self.nova_sonic_sessions[websocket]
            asyncio.create_task(nova_session.close())
            del self.nova_sonic_sessions[websocket]

    async def handle_audio_stream(self, websocket: WebSocket, message_data: Dict[str, Any]):
        """Handle audio streaming data from client using Nova Sonic"""
        try:
            message_type = message_data.get('type')
            
            if message_type == 'audio_stream':
                # Get or create Nova Sonic session
                if websocket not in self.nova_sonic_sessions:
                    nova_session = NovaSonicStreamManager(websocket)
                    success = await nova_session.initialize_stream()
                    if not success:
                        await websocket.send_text(json.dumps({
                            "error": "Failed to initialize audio processing"
                        }))
                        return
                    self.nova_sonic_sessions[websocket] = nova_session
                else:
                    nova_session = self.nova_sonic_sessions[websocket]
                
                # Process audio chunk
                audio_chunk_b64 = message_data.get('audio_chunk')
                if audio_chunk_b64:
                    await nova_session.process_audio_chunk(audio_chunk_b64)
            
            elif message_type == 'audio_stream_stop':
                # End audio input for Nova Sonic
                if websocket in self.nova_sonic_sessions:
                    nova_session = self.nova_sonic_sessions[websocket]
                    await nova_session.end_audio_input()
                    
                    # Get the selected processor
                    selected_processor = nova_session.selected_processor_id
                    
                    # Process the current image with the selected processor
                    if hasattr(self, '_current_image_b64') and self._current_image_b64:
                        response_data = await self.execute_request(
                            selected_processor,
                            self._current_image_b64
                        )
                        
                        # Send processed image back
                        await websocket.send_text(json.dumps({
                            "type": "processed_image",
                            "image": response_data.get("image"),
                            "processor_used": selected_processor
                        }))
            
        except Exception as e:
            log_message(f"Error handling audio stream: {e}", level="ERROR")
            await websocket.send_text(json.dumps({
                "error": f"Audio stream processing error: {str(e)}"
            }))

    async def _call_processor(self, processor_id: int, processor_config: Dict, input_payload: Dict) -> Dict:
        url = f"http://{processor_config['host']}:{processor_config['port']}/process"
        processor_name = processor_config['name']
        log_message(f"Calling {processor_name} (ID: {processor_id}, URL: {url})")
        try:
            response = await self.client.post(url, json=input_payload, timeout=90.0)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            error_msg = f"Error calling {processor_name}: {str(e)}"
            log_message(error_msg, level="ERROR")
            return {"error": error_msg}

    async def execute_request(self, target_processor_id: int, initial_image_b64: Optional[str]) -> Dict:
        self.current_image_for_processing = initial_image_b64
        self.final_image_to_client = initial_image_b64
        self.final_result_to_client = None

        if target_processor_id not in PROCESSOR_CONFIG:
            return {"error": f"Processor ID {target_processor_id} not found"}

        processor_config = PROCESSOR_CONFIG[target_processor_id]
        
        # Call processor
        response = await self._call_processor(
            target_processor_id,
            processor_config,
            {"image": self.current_image_for_processing}
        )

        if "error" not in response:
            if "image" in response:
                self.final_image_to_client = response["image"]
            self.final_result_to_client = response.get("result", "")

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
            "description": config.get("description", "")
        }
        for pid, config in PROCESSOR_CONFIG.items()
    ]
    return {"processors": processors_list}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    log_message("WebSocket connection opened.")

    try:
        while True:
            received_payload_str = await websocket.receive_text()
            
            try:
                client_request = json.loads(received_payload_str)
                
                # Check if this is audio streaming data
                if 'type' in client_request and client_request['type'].startswith('audio_stream'):
                    await manager.handle_audio_stream(websocket, client_request)
                    continue
                
                # Handle regular image processing (store image for later use with audio-selected processor)
                image_b64_from_client = client_request.get("image")
                if image_b64_from_client:
                    manager._current_image_b64 = image_b64_from_client
                    # Just acknowledge image received, wait for audio to select processor
                    await websocket.send_text(json.dumps({
                        "type": "image_received",
                        "status": "Image stored, awaiting processor selection via audio"
                    }))
                    continue
                
                # Handle direct processor requests (backward compatibility)
                target_processor_id = client_request.get("processor")
                if target_processor_id is not None and image_b64_from_client:
                    response_data = await manager.execute_request(
                        target_processor_id,
                        image_b64_from_client
                    )
                    await websocket.send_text(json.dumps(response_data))

            except json.JSONDecodeError:
                log_message("Invalid JSON received from client.", level="ERROR")
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON payload."
                }))
            except Exception as e:
                log_message(f"Error processing client request: {str(e)}", level="ERROR")
                await websocket.send_text(json.dumps({
                    "error": f"Server error: {str(e)}"
                }))

    except WebSocketDisconnect:
        log_message("WebSocket connection closed by client.")
    finally:
        manager.disconnect(websocket)
        log_message("WebSocket connection resources cleaned up.")

# Keep the existing processor startup logic
async def start_processor_servers():
    # ... (keep existing implementation)
    pass

@app.on_event("startup")
async def startup_event():
    await start_processor_servers()

@app.on_event("shutdown")
async def shutdown_event():
    # Clean up any active Nova Sonic sessions
    for session in manager.nova_sonic_sessions.values():
        await session.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
