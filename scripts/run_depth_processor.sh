#!/bin/bash
echo "Attempting to start depth_processor..."
CONDA_BASE_DIR=$(conda info --base)
if [ -z "$CONDA_BASE_DIR" ]; then echo "Conda base directory not found." >&2; exit 1; fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
if ! conda activate depth-pro; then echo "Failed to activate conda: depth-pro" >&2; exit 1; fi
echo "Conda env 'depth-pro' activated for depth_processor."
cd "/home/znasif/vidServer"
echo "Starting uvicorn for processors.depth_processor:app on 127.0.0.1:8002..."
mkdir -p "/home/znasif/vidServer/logs"
exec uvicorn processors.depth_processor:app --host 127.0.0.1 --port 8002 --log-level info >> "/home/znasif/vidServer/logs/depth_processor.log" 2>&1
