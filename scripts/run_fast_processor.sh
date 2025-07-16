#!/bin/bash
echo "Attempting to start fast_processor..."
CONDA_BASE_DIR=$(conda info --base)
if [ -z "$CONDA_BASE_DIR" ]; then echo "Conda base directory not found." >&2; exit 1; fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
if ! conda activate whatsai2; then echo "Failed to activate conda: whatsai2" >&2; exit 1; fi
echo "Conda env 'whatsai2' activated for fast_processor."
cd "/home/znasif/vidServer"
echo "Starting uvicorn for processors.fast_processor:app on 127.0.0.1:8006..."
mkdir -p "/home/znasif/vidServer/logs"
exec uvicorn processors.fast_processor:app --host 127.0.0.1 --port 8006 --log-level info >> "/home/znasif/vidServer/logs/fast_processor.log" 2>&1
