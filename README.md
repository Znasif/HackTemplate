# 🚀 Setup Guide

This guide walks you through the installation and setup of necessary dependencies, servers, and services for your project.

---

## 📌 Installation Steps

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
pip install -r requirements.txt
```

---

## 🦙 Starting LLaMA Server
```bash
llama-server -m ./llama.cpp/models/Qwen2.5.1-Coder-7B-Instruct-Q6_K --host 127.0.0.1 --port 8080
```

---

## 🦙 Starting LLaVA Server
```bash
./llama-qwen2vl-cli -m ../../models/Qwen2-VL-2B-Instruct-Q6_K.gguf --mmproj ../../models/qwen2-vl-2b-instruct-vision.gguf --image ../../../vidServer/james.
jpg -p "describe"
```

---


## 🎥 Starting Discord Video Server
### Windows Client:
```powershell
cd \wsl$\Ubuntu\home\znasif\vidServer\client
```

### WSL Server:
```bash
uvicorn main:app --reload
```

---

## 🤖 Starting Discord Bot
```bash
python bot.py
```

---

### 🎯 You're all set! Happy coding! 🚀

