#!/bin/bash
echo "Attempting to start scene_object_processor..."
CONDA_BASE_DIR=$(conda info --base)
if [ -z "$CONDA_BASE_DIR" ]; then echo "Conda base directory not found." >&2; exit 1; fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
if ! conda activate whatsai2; then echo "Failed to activate conda: whatsai2" >&2; exit 1; fi
echo "Conda env 'whatsai2' activated for scene_object_processor."
cd "/home/znasif/vidServer"
echo "Starting uvicorn for processors.scene_object_processor:app on 127.0.0.1:8005..."
mkdir -p "/home/znasif/vidServer/logs"
exec uvicorn processors.scene_object_processor:app --host 127.0.0.1 --port 8005 --log-level info >> "/home/znasif/vidServer/logs/scene_object_processor.log" 2>&1
