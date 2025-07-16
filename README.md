# WhatsAI Web Client

This project was created to be used by Blind and Low Vision users to have an AAII (Accessible Artificial Intelligence Implementation) available to them wherever they may be. Most components of this project have minimal dependency on a stable internet connection once all the components have been installed if the user wants to work solely on their workstation (PC/laptop). If the user wants to access it from anywhere, a WhatsApp connection needs to be set up. You would need both the server and the client web applications running for it to work. Make sure to start the server first. You can access the client by opening client/screen_wss.html in your browser.

## Quick Start with Remote Server

If you want to use an already running server:

1. Obtain the server WebSocket URL (format: `wss://[server-address]/ws`)
2. Open the WhatsAI Web Client by opening the client/screen_wss.html in your browser.
3. Paste the server URL into the "Server URL" text field
4. Click "Select Screen to Share" and choose your desired screen
5. Click "Start Streaming" to begin sharing your screen with the server
6. Select a processor from the dropdown menu to analyze your screen

## Available Processors

The system includes several processors, ordered by complexity:

### Basic Processor (ID: 0)
- **Description**: A simple pass-through processor that returns the original image without modifications
- **Dependencies**: None
- **Use Case**: Testing connectivity and stream quality

### Scene Object Processor (ID: 4)
- **Description**: Performs real-time object detection and segmentation using YOLO11
- **Dependencies**: None
- **Use Case**: Identifying and locating objects in your screen
- **Reference**: Based on Ultralytics YOLO - https://github.com/ultralytics/ultralytics

### Scene Captioning Processor (ID: 2)
- **Description**: Generates detailed region-based captions and performs OCR using Florence-2
- **Dependencies**: None
- **Use Case**: Understanding text and visual content on screen
- **Reference**: Microsoft Florence-2 - https://huggingface.co/microsoft/Florence-2-large

### CamIO Processor (ID: 3)
- **Description**: Provides specialized object recognition for curated items
- **Dependencies**: None
- **Use Case**: Learning about specific pre-trained objects
- **Reference**: Based on Simple CamIO - https://github.com/Coughlan-Lab/simple_camio

### Finger Count Processor (ID: 1)
- **Description**: Detects hands and counts raised fingers using MediaPipe
- **Dependencies**: Basic Processor (ID: 0)
- **Use Case**: Gesture recognition and finger counting
- **Reference**: Adapted from Finger Counter using MediaPipe - https://github.com/HarshitDolu/Finger-Counter-using-mediapipe

## Setting Up Your Own Server

### Local Server Setup

#### Prerequisites
1. WSL2 on Windows or Linux system
2. Docker Desktop with WSL2 integration enabled
3. Gemini API key

#### Installation Steps

1. Clone the repository:
   ```bash
   git clone --single-branch --branch workshop https://github.com/Znasif/HackTemplate.git
   cd HackTemplate
   ```

2. Create and configure the `.env` file:
   ```bash
   GEMINI_API_KEY="your-gemini-api-key"
   ```

3. Build the Docker container:
   ```bash
   docker-compose build
   ```

4. Start the server:
   ```bash
   docker-compose up
   ```

5. Access the server:
   - Local access: `ws://localhost:8000/ws`
   - Remote access: Use localtunnel (https://github.com/localtunnel/localtunnel) to expose the port

### Deployment to RunPod

#### Prerequisites
1. Docker Hub account
2. RunPod account with credits

#### Deployment Steps

1. Tag and push your local Docker image to Docker Hub:
   ```bash
   docker tag whatsai-server:latest yourusername/whatsai:latest
   docker push yourusername/whatsai:latest
   ```

2. Create a new pod on RunPod:
   - Container Image: `yourusername/whatsai:latest`
   - Container Start Command: `bash -c "cd /app && /app/start_server.sh"`
   - Container Disk: 50 GB
   - Volume Disk: 20 GB (optional, for persistent storage)
   - Volume Mount Path: `/workspace`
   - Expose HTTP Ports: `8000`
   - GPU Selection: RTX 4000 Ada or similar

3. Add environment variables in RunPod:
   - GEMINI_API_KEY: your-api-key
   - PYTHONPATH: /app

4. Deploy the pod and obtain your URL:
   - Format: `wss://[pod-id]-8000.proxy.runpod.net/ws`


## Audio Streaming Features

The system supports real-time audio streaming for voice-based interactions:

1. **Direct Audio**: Screen reader and system audio output work automatically
2. **Remote Audio via Start Audio Button**: Dictate processor by pressing the "Start Audio" button and saying which processor to start and then "Stop Audio" to initiate.
3. **Virtual Audio Cable**: For WhatsApp audio streaming:
   - Install VB-Audio Virtual Cable from https://vb-audio.com/Cable/
   - Set VB-Audio Virtual Cable Output as default system audio output
   - In WhatsApp calls, set audio input to VB-Audio Virtual Cable Input

## API Endpoints

- **WebSocket**: `ws://[server]/ws` - Main streaming endpoint
- **HTTP GET**: `http://[server]:8000/processors` - List available processors

## Troubleshooting

### Connection Issues
- Verify the server URL format includes `/ws` at the end
- Check if the server is running: `curl http://[server]:8000/processors`
- Ensure your firewall allows WebSocket connections

### Performance Optimization
- For best results, use a wired internet connection
- Close unnecessary applications to reduce screen capture overhead
- Select specific application windows instead of full screen when possible

## Brief Demo

Click on the following image which will take you to a playlist:

[![Demo Link for Whatsapp Livestream AI processing](https://i.ytimg.com/vi/ExhlwkUW_gc/hqdefault.jpg?sqp=-oaymwExCNACELwBSFryq4qpAyMIARUAAIhCGAHwAQH4Af4JgALQBYoCDAgAEAEYZSBRKEAwDw==&rs=AOn4CLDxzMwlnE3AVdbFIucWFV93J9Jg3g)](https://www.youtube.com/playlist?list=PLk3VM_Y78PILin5BQJ0cYq_OdmuT7v1VY)