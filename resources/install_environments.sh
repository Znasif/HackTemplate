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

# --- Sourcing and Setup ---
CONDA_BASE_DIR=$(conda info --base)
if [ -z "$CONDA_BASE_DIR" ]; then 
    echo "Conda base directory not found." >&2
    exit 1
fi
source "$CONDA_BASE_DIR/etc/profile.d/conda.sh"
REQ_DIR="/tmp/requirements"

# --- Functions ---

create_environment() {
    local env_name=$1
    echo ""
    echo "=== Creating Environment: $env_name ==="
    
    # File paths
    local explicit_file="$REQ_DIR/${env_name}-explicit.txt"
    local pip_file="$REQ_DIR/${env_name}-pip.txt"
    local yml_file="$REQ_DIR/${env_name}.yml"
    
    if [ -f "$explicit_file" ]; then
        echo "Found explicit.txt file - using exact package URLs for conda packages..."
        echo "Creating environment from $explicit_file..."
        
        # Create environment using explicit file (conda packages only)
        conda create -n "$env_name" --file "$explicit_file"
        
        # Now install pip packages if pip.txt exists
        if [ -f "$pip_file" ]; then
            echo "Installing pip requirements from $pip_file..."
            
            # Special PyTorch handling for certain environments if needed
            if [ "$env_name" = "depth-pro" ] || [ "$env_name" = "whatsai2" ]; then
                echo "Note: PyTorch already installed via conda explicit.txt"
            fi
            
            # Install pip requirements
            conda run -n "$env_name" pip install --no-cache-dir -r "$pip_file"
        else
            echo "No pip.txt file found for $env_name"
        fi
        
    elif [ -f "$yml_file" ]; then
        echo "No explicit.txt found, falling back to yml file..."
        echo "Creating environment from $yml_file..."
        conda env create -f "$yml_file"
        
        # Still check for additional pip requirements
        if [ -f "$pip_file" ]; then
            echo "Installing additional pip requirements from $pip_file..."
            conda run -n "$env_name" pip install --no-cache-dir -r "$pip_file"
        fi
        
    else
        echo "Neither explicit.txt nor yml file found. Creating basic environment..."
        conda create -n "$env_name" python=3.10 -y
        
        # Still try pip requirements
        if [ -f "$pip_file" ]; then
            echo "Installing pip requirements from $pip_file..."
            conda run -n "$env_name" pip install --no-cache-dir -r "$pip_file"
        fi
    fi

    echo "Environment $env_name creation completed"
}

verify_environment() {
    local env_name=$1
    echo ""
    echo "=== Verifying Environment: $env_name ==="
    
    local cmd_to_run=""
    case $env_name in
        "aws") cmd_to_run="import fastapi, uvicorn, httpx" ;;
        "whatsai2" | "depth-pro") 
            cmd_to_run="import torch, numpy; print(f'PyTorch CUDA: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
            ;;
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

# --- Main Execution ---

echo "--- Starting build for $ENV_NAME_TO_CREATE ---"

if create_environment "$ENV_NAME_TO_CREATE" && verify_environment "$ENV_NAME_TO_CREATE"; then
    echo "✅ Successfully created and verified $ENV_NAME_TO_CREATE"
else
    echo "❌ Failed to create or verify $ENV_NAME_TO_CREATE"
    exit 1
fi

echo "--- Cleaning conda state to prevent conflicts in subsequent RUN commands ---"
conda clean --all -y

echo "🎉 Script finished successfully for $ENV_NAME_TO_CREATE."
exit 0