# WhatsApp Client

This project was created to be used by Blind and Low Vision users to have an AAII (Accessible Artificial Intelligence Implementation) available to them wherever they may be. Most components of this project have minimal dependency on a stable internet connection once all the components have been installed if the user wants to work solely on their workstation (PC/laptop). If the user wants to access it from anywhere, a WhatsApp connection needs to be set up.

## Client Setup (Windows Only)

Currently, the client only works in Windows, and the shared monitor is not selectable in the GUI.

#### a. Install the following dependencies in Windows after installing Python:

```powershell
pip install numpy opencv-python asyncio websockets pillow pywin32 pyttsx3
```
1. Someone with Meta Ray-Ban glasses or a smartphone video calls your WhatsApp.  
2. You accept the video call from your computer.  
3. The monitor with the focused participant's video feed is captured and sent to a local server for processing:

### Server Setup

#### a. Set up the server
- Open WSL on your Windows machine and clone this repository.
- Fill in the `.env` file in the project root directory:

  ```bash
  PORT=8080
  GROQ_API_KEY="get a Groq API key for llama-3.2-90b-vision-preview live query"
  OPENAI_API_KEY="get OpenAI API key"
  ```

#### b. Install CUDA Toolkit & Dependencies after installing Conda

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6
sudo apt install -y libcudnn9-cuda-12
sudo apt install libnvinfer10 libnvinfer-plugin10
conda env create -f environment.yml
conda activate whatsapp-vision
```

#### c. Start the server

```bash
cd path/to/server/
uvicorn main:app --reload
```

#### b. Run the client from Windows PowerShell:

```powershell
cd path\to\client
python main.py
```

#### c. Start the client

- Press **Start Streaming** → This will start sharing the PC screen with the server.  
- Then, you will have the option to process the stream in any of the following ways by selecting from the dropdown menu:

  ```python
  processor_options = [
      "Dense Region Caption",
      "OCR",
      "YOLO Detection",
      "MediaPipe",
      "Base Processor",
      "Groq",
      "OpenAI"
  ]
  ```

#### d. Processing Options

The resulting feed is screen-shared, and the description is read aloud. The following types of processors are already implemented:

1. **Live description** with Groq and `llama-3.2-90b-vision-preview`.
2. **Live description** with OpenAI `GPT-4o`.
3. **Segmentation and Detection** of objects in the scene (YOLO11).
4. **Face, Hand, and Body Pose Estimation** (MediaPipe).
5. **OCR and Region-based Captioning** (Florence 2).

#### e. Remote Audio Sharing

If you want to remotely share the audio response, you would need to use another video call service like Zoom and share the client app window with **Share Audio** turned on. There is currently no option to turn on audio streaming to Whatsapp without Business account.


---
---

# Discord-A11y

This is a project that was created to be used by Blind and Low Vision users to have a AAII (Accessible Artificial Intelligence Implementation) available to them wherever they may be. Most components of this project have minimal dependency on a stable internet once all the components have been installed, if the user wants to work solely on their workstation (pc/laptop). If the user wants to access it from anywhere, the discord connection would need to be set up.

It has the following features:

1. /querycode prompt : discord slash command allows the user to ask the local llama.cpp server to generate runnable python code. If the code generates an image, the code will be executed locally and sent as attachment to that thread. The generated image will then be sent to a VLM (vision language model) for consistency detection. (Qwen2-VL and Qwen2-Code)
2. /qyeryimage image, prompt : this slash command allows user to ask any question of an image. (PaliGemma2)
3. the voice channel's focused participant's video feed is captured and sent to a local server for processing and the resulting feed is screen shared and description is read out loud. The following types of processors are already implemented:
    a. Segmentation and Detection of objects in the scene (Yolo11)
    b. Face, Hand and Body pose estimation (Mediapipe)
    c. OCR and Region based captioning (Florence 2)

# Brief Demo
Click on the following image which will take you to a playlist:

[![Demo Link for Whatsapp Livestream AI processing](https://i.ytimg.com/vi/ExhlwkUW_gc/hqdefault.jpg?sqp=-oaymwExCNACELwBSFryq4qpAyMIARUAAIhCGAHwAQH4Af4JgALQBYoCDAgAEAEYZSBRKEAwDw==&rs=AOn4CLDxzMwlnE3AVdbFIucWFV93J9Jg3g)](https://www.youtube.com/playlist?list=PLk3VM_Y78PILin5BQJ0cYq_OdmuT7v1VY)

# 🚀 Setup Guide

This guide walks you through the installation and setup of necessary dependencies, servers, and services for this discord-bot based accessibility project.

---

## Prerequisites

Have the following things ready:

1. Install Llama.cpp following: https://github.com/ggml-org/llama.cpp and in llama.cpp/models folder download the following: models/Qwen2-VL-7B-Instruct-Q6_K.gguf, Qwen2-VL-2B-Instruct-Q6_K.gguf,qwen2-vl-2b-instruct-vision.gguf
2. Add a .env file in server folder with the following filled in

#### Fill in the .env file in the project root directory
```bash
    MODEL_PATH=path/to/Qwen2-VL-7B-Instruct-Q6_K.gguf
    VLM_MODEL_PATH =path/to/Qwen2-VL-2B-Instruct-Q6_K.gguf
    VISION_MODEL_PATH =path/to/qwen2-vl-2b-instruct-vision.gguf
    COMMAND_PREFIX=!
    PORT=8080
    DISCORD_TOKEN="get this for the discord bot you create"
    DISCORD_GUILD_ID="the server you want the bot to be active in"
    DISCORD_CHANNEL_ID="the channel where the bot will reply"
    PERMISSIONS="the permission integer"
    GROQ_API_KEY="get a groq api key for llama-3.2-90b-vision-preview live query"
    OPENAI_API_KEY="get openai api key"
```

2. In server/models download: AtkinsonHyperlegible-Regular.ttf and yolo11n-seg.pt

## 📌 Installation Steps

The following instruction are for WSL:

### 1️⃣ Install CUDA Toolkit & Dependencies
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6
sudo apt install -y libcudnn9-cuda-12
sudo apt install libnvinfer10 libnvinfer-plugin10
```

### 2️⃣ Install Python Dependencies
```bash
conda env create -f environment.yml
conda activate discord-vision
```

---

## 🦙 Starting LLaMA Server
This needs to be started before the discord bot can be activated
```bash
llama-server -m ./llama.cpp/models/Qwen2.5.1-Coder-7B-Instruct-Q6_K --host 127.0.0.1 --port 8080
```

---

## 🦙 Testing LLaVA agent
```bash
./llama-qwen2vl-cli -m ../../models/Qwen2-VL-2B-Instruct-Q6_K.gguf --mmproj ../../models/qwen2-vl-2b-instruct-vision.gguf --image ../../../vidServer/james.
jpg -p "describe"
```

---


## 🎥 Starting Discord Video Server
### Windows Client:

Currently the client only works in Windows. And the shared monitor is not selectable in GUI.
```powershell
cd path\to\client
python main.py
```
Then start the client by pressing start streaming -> This will start to share the PC screen to the server. Then you will have the option to process the stream in any of the following ways by selecting from the dropdown menu:

```python
processor_options = [
        "Dense Region Caption",
        "OCR",
        "YOLO Detection",
        "MediaPipe",
        "Base Processor",
        "Groq",
        "OpenAI"
    ]
```

### WSL Server:
```bash
cd path/to/server/
uvicorn main:app --reload
```

---

## 🤖 Starting Discord Bot
```bash
python bot.py
```

---

# For Hackathon

The main three classes that you would want to extend are the following:

1. BaseProcessor in server/processors/base_processor.py for video stream processing. Example extensions can be found in the server/processors/ folder.
2. MyClient in server/bot.py and add more slash commands.
3. LlamaCppServerModifier in server/llama.py and add functions similar to modify_text.

### 🎯 You're all set! Happy coding! 🚀

