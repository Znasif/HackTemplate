Installation:
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6
sudo apt install -y libcudnn9-cuda-12
sudo apt install libnvinfer10 libnvinfer-plugin10
pip install -r requirements.txt

Command to start llama-server:

llama-server -m ./llama.cpp/models/Qwen2.5.1-Coder-7B-Instruct-Q6_K --host 127.0.0.1 --port 8080

Command to start llava-server:

Command to start discord video server:

windows client: cd \\wsl$\Ubuntu\home\znasif\vidServer\client
wsl server: uvicorn main:app --reload

Command to start discord bot:
python bot.py
