#!/bin/bash
set -e

# This script now expects a single argument: the name of the environment to create.
if [ -z "$1" ]; then
    echo "Error: No environment name supplied. Usage: $0 <env_name>"
    exit 1
fi

# The single environment name to process.
ENV_NAME_TO_CREATE=$1
echo "=== Preparing to build single environment: $ENV_NAME_TO_CREATE ==="

# --- Sourcing and Setup (No changes needed here) ---
CONDA_BASE_DIR=$(conda info --base)
if [ -z "$CONDA_BASE_DIR" ]; then 
    echo "Conda base directory not found." >&2
    exit 1
fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
REQ_DIR="/tmp/requirements"

# --- Functions (Using the robust 'conda run' versions) ---

create_environment() {
    local env_name=$1
    echo ""
    echo "=== Creating Environment: $env_name ==="
    
    local yml_file="$REQ_DIR/${env_name}.yml"
    if [ -f "$yml_file" ]; then
        echo "Creating environment from $yml_file..."
        conda env create -f "$yml_file"
    else
        echo "YML file not found. Creating environment $env_name manually..."
        conda create -n "$env_name" python=3.10 -y
    fi

    # local req_pip_file="$REQ_DIR/${env_name}-pip.txt"
    # if [ -f "$req_pip_file" ]; then
    #     echo "Installing pip requirements from $req_pip_file using conda run..."
    #     # Add special PyTorch handling for depth-pro here if needed, or keep it in the yml/pip file
    #     if [ "$env_name" = "depth-pro" ] || [ "$env_name" = "whatsai2" ]; then
    #         echo "Installing PyTorch with CUDA support using conda run..."
    #         conda run -n "$env_name" pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121
    #     fi
    #     conda run -n "$env_name" pip install --no-cache-dir -r "$req_pip_file"
    # fi
    # echo "Environment $env_name setup completed"
}

verify_environment() {
    local env_name=$1
    echo ""
    echo "=== Verifying Environment: $env_name ==="
    
    local cmd_to_run=""
    case $env_name in
        "aws") cmd_to_run="import fastapi, uvicorn, httpx" ;;
        "whatsai2" | "depth-pro") cmd_to_run="import torch, numpy" ;;
        *) echo "Unknown environment for verification: $env_name"; return 1 ;;
    esac

    if conda run -n "$env_name" python -c "$cmd_to_run"; then
        echo "✅ $env_name environment OK"
        return 0
    else
        echo "⚠️ Verification import check failed for $env_name"
        return 1
    fi
}


# --- Main Execution (No more loop) ---

echo "--- Starting build for $ENV_NAME_TO_CREATE ---"

if create_environment "$ENV_NAME_TO_CREATE" && verify_environment "$ENV_NAME_TO_CREATE"; then
    echo "✅ Successfully created and verified $ENV_NAME_TO_CREATE"
else
    echo "❌ Failed to create or verify $ENV_NAME_TO_CREATE"
    exit 1 # Exit with a failure code
fi

echo "--- Cleaning conda state to prevent conflicts in subsequent RUN commands ---"
conda clean --all -y

echo "🎉 Script finished successfully for $ENV_NAME_TO_CREATE."
exit 0