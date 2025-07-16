#!/bin/bash
set -e

# Initialize conda for this shell session.
# This makes `conda activate` available.
eval "$(/opt/conda/bin/conda shell.bash hook)"

# --- Start the main server (foreground process) ---
# This server will be the main process for the container.
echo "--- Starting main server in 'aws' environment... ---"
conda activate aws

# The rest of this script is the 'exec' command, which replaces the shell
# process with the uvicorn process.
if [ -f "stream_wss.py" ]; then
    echo "Starting stream_wss server..."
    exec uvicorn stream_wss:app --host 0.0.0.0 --port 8000
elif [ -f "stream_sonic.py" ]; then
    echo "Starting stream_sonic server..."
    exec uvicorn stream_sonic:app --host 0.0.0.0 --port 8000
else
    echo "No server found..."
fi