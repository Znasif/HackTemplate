#!/bin/bash
echo "Attempting to start spatial_processor..."
CONDA_BASE_DIR=$(conda info --base)
if [ -z "$CONDA_BASE_DIR" ]; then echo "Conda base directory not found." >&2; exit 1; fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
if ! conda activate spatiallm; then echo "Failed to activate conda: spatiallm" >&2; exit 1; fi
echo "Conda env 'spatiallm' activated for spatial_processor."
cd "/home/znasif/vidServer/server"
echo "Starting uvicorn for processors.spatial_processor:app on 127.0.0.1:8007..."
mkdir -p "/home/znasif/vidServer/server/logs"
exec uvicorn processors.spatial_processor:app --host 127.0.0.1 --port 8007 --log-level info >> "/home/znasif/vidServer/server/logs/spatial_processor.log" 2>&1
