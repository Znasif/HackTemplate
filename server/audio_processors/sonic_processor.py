import uuid, json, traceback, base64, os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketDisconnect

import asyncio, time
from dotenv import load_dotenv

from typing import List, Dict, Optional, Union, Any, Callable, AsyncGenerator
# --- Using the specific AWS SDK imports as requested ---
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from smithy_aws_core.credentials_resolvers.environment import EnvironmentCredentialsResolver

def load_config(path: str) -> dict:
    with open(path, 'r') as f:
        config = json.load(f)
    
    # Filter out disabled processors and convert keys to integers
    return {
        int(k): v for k, v in config.items() if v.get("enabled", True)
    }

PROCESSOR_CONFIG = load_config("processor_config.json")

def log_message(message: str, level: str = "INFO"):
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}")

class NovaSonicStreamManager:
    """Manages Nova Sonic integration using the correct event-based protocol."""
    def __init__(self, websocket: WebSocket, manager, model_id='amazon.nova-sonic-v1:0', region='us-east-1'):
        self.websocket = websocket
        self.manager = manager
        self.model_id = model_id
        self.region = region
        self.bedrock_client: Optional[BedrockRuntimeClient] = None
        self.stream_response: Optional[Any] = None # Will hold the DuplexEventStream object
        self.is_active = False
        self.prompt_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.ai_response_audio_chunks: List[bytes] = []
        self.ai_stream_complete = asyncio.Event()
        self._initialize_client()

    def _initialize_client(self):
        load_dotenv()
        os.environ['AWS_DEFAULT_REGION'] = self.region
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
            http_auth_scheme_resolver=HTTPAuthSchemeResolver(),
            http_auth_schemes={"aws.auth#sigv4": SigV4AuthScheme()}
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)

    def _create_processor_selection_tool(self):
        processor_descriptions = "\n".join([f"- ID {pid}: {cfg['name']} - {cfg.get('description', 'No description')}" for pid, cfg in PROCESSOR_CONFIG.items()])
        return { "toolSpec": {
            "name": "selectProcessor",
            "description": f"Select the appropriate image processor. Available processors:\n{processor_descriptions}",
            "inputSchema": { "json": json.dumps({
                "type": "object",
                "properties": {
                    "processorId": { "type": "integer", "description": "Processor ID", "enum": list(PROCESSOR_CONFIG.keys()) },
                    "reason": { "type": "string", "description": "Reason for selection" }
                }, "required": ["processorId", "reason"]
            })}
        }}

    def _create_prompt_start_event(self):
        return json.dumps({ "event": { "promptStart": {
            "promptName": self.prompt_name,
            "textOutputConfiguration": { "mediaType": "text/plain" },
            "audioOutputConfiguration": {
                "mediaType": "audio/lpcm", "sampleRateHertz": 24000, "sampleSizeBits": 16,
                "channelCount": 1, "voiceId": "matthew", "encoding": "base64", "audioType": "SPEECH"
            },
            "toolUseOutputConfiguration": { "mediaType": "application/json" },
            "toolConfiguration": { "tools": [self._create_processor_selection_tool()] }
        }}})

    async def initialize_stream(self):
        try:
            operation_input = InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            # This call returns the DuplexEventStream object directly
            self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(operation_input)
            self.is_active = True
            
            system_prompt = "You are an AI assistant. Listen to the user and select the best processor for their request. Confirm your choice concisely."
            start_session = json.dumps({"event": {"sessionStart": {"inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}}}})
            prompt_start = self._create_prompt_start_event()
            text_start = json.dumps({"event": {"contentStart": {"promptName": self.prompt_name, "contentName": self.content_name, "type": "TEXT", "role": "SYSTEM", "textInputConfiguration": {"mediaType": "text/plain"}}}})
            text_input = json.dumps({"event": {"textInput": {"promptName": self.prompt_name, "contentName": self.content_name, "content": system_prompt}}})
            text_end = json.dumps({"event": {"contentEnd": {"promptName": self.prompt_name, "contentName": self.content_name}}})
            for event in [start_session, prompt_start, text_start, text_input, text_end]:
                await self.send_raw_event(event)
            
            asyncio.create_task(self._process_responses())
            log_message("Nova Sonic stream initialized with correct event protocol.")
            return True
        except Exception as e:
            log_message(f"Failed to initialize Nova Sonic stream: {traceback.format_exc()}", level="ERROR")
            return False

    async def send_raw_event(self, event_json_str: str):
        if not self.is_active or not self.stream_response: return
        payload_part = BidirectionalInputPayloadPart(bytes_=event_json_str.encode('utf-8'))
        event_chunk = InvokeModelWithBidirectionalStreamInputChunk(value=payload_part)
        # --- FIXED: Access input_stream as an attribute ---
        await self.stream_response.input_stream.send(event_chunk)

    async def process_audio_chunk(self, audio_chunk_b64: str):
        if not self.is_active: return
        if not hasattr(self, '_audio_started') or not self._audio_started:
            content_start = json.dumps({"event": {"contentStart": { "promptName": self.prompt_name, "contentName": self.audio_content_name, "type": "AUDIO", "role": "USER", "interactive": True, "audioInputConfiguration": {"mediaType": "audio/webm", "encoding": "base64"}}}})
            await self.send_raw_event(content_start)
            self._audio_started = True
        audio_event = json.dumps({"event": {"audioInput": {"promptName": self.prompt_name, "contentName": self.audio_content_name, "content": audio_chunk_b64}}})
        await self.send_raw_event(audio_event)

    async def end_audio_input(self):
        if hasattr(self, '_audio_started') and self._audio_started:
            content_end = json.dumps({"event": {"contentEnd": {"promptName": self.prompt_name, "contentName": self.audio_content_name}}})
            await self.send_raw_event(content_end)
            prompt_end = json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}})
            await self.send_raw_event(prompt_end)
            self._audio_started = False
            log_message("Signaled end of audio to Nova Sonic.")

    async def _process_responses(self):
        if not self.is_active or not self.stream_response:
            self.ai_stream_complete.set()
            return
        # --- FIXED: Access output_stream as an attribute ---
        output_stream = self.stream_response.output_stream
        try:
            async for event in output_stream:
                if event.value and event.value.payload_part and event.value.payload_part.bytes_:
                    chunk = event.value.payload_part.bytes_.decode('utf-8')
                    response_data = json.loads(chunk)
                    if 'toolUse' in response_data.get('event', {}):
                        tool_use = response_data['event']['toolUse']
                        if tool_use.get('toolName') == 'selectProcessor':
                            content = json.loads(tool_use['content'])
                            processor_id = content.get('processorId')
                            last_image = self.manager.last_images.get(self.websocket)
                            if processor_id is not None and last_image:
                                asyncio.create_task(self.manager.execute_request_and_send(self.websocket, processor_id, last_image))
                    elif 'audioOutput' in response_data.get('event', {}):
                        audio_out_b64 = response_data['event']['audioOutput']['content']
                        self.ai_response_audio_chunks.append(base64.b64decode(audio_out_b64))
        except Exception as e:
            log_message(f"Error processing Nova Sonic response stream: {e}", level="ERROR")
        finally:
            log_message("AI response stream finished. Setting completion event.")
            self.ai_stream_complete.set()

    async def close(self):
        try:
            if self.is_active and self.stream_response:
                session_end = json.dumps({"event": {"sessionEnd": {}}})
                await self.send_raw_event(session_end)
            log_message("Nova Sonic manager cleaned up.")
        except Exception as e:
            log_message(f"Error during Nova Sonic manager cleanup: {e}", level="WARNING")
