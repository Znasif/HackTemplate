#!/bin/bash
set -e

# Initialize conda for this shell session.
# This makes `conda activate` available.
eval "$(/opt/conda/bin/conda shell.bash hook)"

# Run the Python script to start all background processors.
# It will read the config and launch them.
python3 /app/start_processor.py

# Give the background processors a moment to initialize.
echo "--- Waiting for processors to start up... ---"
sleep 10

# --- Start the main server (foreground process) ---
# This server will be the main process for the container.
echo "--- Starting main server in 'aws' environment... ---"
conda activate aws

# The rest of this script is the 'exec' command, which replaces the shell
# process with the uvicorn process.
if [ -f "stream_flash.py" ]; then
    echo "Starting stream_flash server..."
    exec uvicorn stream_flash:app --host 0.0.0.0 --port 8080
elif [ -f "stream_sonic.py" ]; then
    echo "Starting stream_sonic server..."
    exec uvicorn stream_sonic:app --host 0.0.0.0 --port 8080
else
    echo "No server found..."
fi