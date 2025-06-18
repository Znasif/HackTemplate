#!/bin/bash

# Enhanced script to extract requirements from your conda environments
# This captures source-built dependencies, git installs, and custom builds
# Run this in your WSL2 environment where your conda envs are set up

set -e

echo "🔍 Enhanced requirements extraction for source-built dependencies..."

# Initialize conda properly
echo "🔧 Initializing conda..."

# Find conda installation
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/miniconda3"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="$HOME/anaconda3"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="/opt/conda"
elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_PATH="/usr/local/miniconda3"
else
    echo "❌ Conda installation not found. Please ensure conda is installed and accessible."
    echo "Common locations checked:"
    echo "  - $HOME/miniconda3/etc/profile.d/conda.sh"
    echo "  - $HOME/anaconda3/etc/profile.d/conda.sh" 
    echo "  - /opt/conda/etc/profile.d/conda.sh"
    echo "  - /usr/local/miniconda3/etc/profile.d/conda.sh"
    exit 1
fi

echo "✅ Found conda at: $CONDA_PATH"

# Source conda configuration
source "$CONDA_PATH/etc/profile.d/conda.sh"

# Verify conda is working
if ! command -v conda &> /dev/null; then
    echo "❌ Conda command not available after sourcing. Please check your conda installation."
    exit 1
fi

echo "✅ Conda initialized successfully"

# Create requirements directory
mkdir -p requirements

# Function to extract comprehensive environment info
extract_env_details() {
    local env_name=$1
    echo "📦 Analyzing environment: $env_name"
    
    # Check if environment exists
    if ! conda env list | grep -q "^$env_name "; then
        echo "⚠️  Environment '$env_name' not found. Skipping..."
        echo "Available environments:"
        conda env list
        return 1
    fi
    
    # Activate environment with proper conda initialization
    echo "  🔄 Activating environment: $env_name"
    if ! conda activate "$env_name"; then
        echo "❌ Failed to activate environment: $env_name"
        return 1
    fi
    
    echo "  📋 Basic exports..."
    # Standard exports
    conda env export --name "$env_name" > "requirements/${env_name}.yml"
    pip freeze > "requirements/${env_name}-pip.txt"
    
    echo "  🔍 Detailed package analysis..."
    # Detailed conda package info (shows build strings and channels)
    conda list --explicit > "requirements/${env_name}-explicit.txt"
    conda list --json > "requirements/${env_name}-packages.json"
    
    echo "  🌐 Git and editable packages..."
    # Find packages installed from git or in editable mode
    pip list --format=freeze | grep -E "(git\+|file://|@ file)" > "requirements/${env_name}-git-packages.txt" 2>/dev/null || echo "# No git packages found" > "requirements/${env_name}-git-packages.txt"
    
    # Find editable installs
    pip list --editable --format=freeze > "requirements/${env_name}-editable.txt" 2>/dev/null || echo "# No editable packages found" > "requirements/${env_name}-editable.txt"
    
    echo "  🏗️ Build and system info..."
    # System and build information
    cat > "requirements/${env_name}-system-info.txt" << EOF
# System Information for $env_name environment
# Generated on: $(date)
# Python version: $(python --version 2>/dev/null || echo "Python not available")
# Python executable: $(which python 2>/dev/null || echo "Python path not found")
# Conda version: $(conda --version 2>/dev/null || echo "Conda version not available")
# Platform: $(uname -a 2>/dev/null || echo "Platform info not available")

# CUDA Information (if available)
EOF
    
    # CUDA info
    if command -v nvidia-smi &> /dev/null; then
        echo "# NVIDIA Driver version:" >> "requirements/${env_name}-system-info.txt"
        nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -1 >> "requirements/${env_name}-system-info.txt" 2>/dev/null || echo "# CUDA not available" >> "requirements/${env_name}-system-info.txt"
        echo "# CUDA version:" >> "requirements/${env_name}-system-info.txt"
        nvcc --version 2>/dev/null | grep "release" >> "requirements/${env_name}-system-info.txt" || echo "# NVCC not available" >> "requirements/${env_name}-system-info.txt"
    else
        echo "# NVIDIA/CUDA not available" >> "requirements/${env_name}-system-info.txt"
    fi
    
    # PyTorch build info (if installed)
    echo "  🔥 PyTorch build analysis..."
    python -c "
import sys
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'PyTorch CUDA version: {torch.version.cuda}')
        print(f'PyTorch cuDNN version: {torch.backends.cudnn.version()}')
    try:
        print(f'PyTorch build: {torch.__config__.show()}')
    except:
        print('PyTorch build config not available')
except ImportError:
    print('PyTorch not installed')
except Exception as e:
    print(f'Error getting PyTorch info: {e}')
" > "requirements/${env_name}-pytorch-info.txt" 2>/dev/null || echo "# PyTorch info not available" > "requirements/${env_name}-pytorch-info.txt"
    
    # Check for common source-built packages
    echo "  🔧 Source-built package detection..."
    cat > "requirements/${env_name}-source-analysis.txt" << EOF
# Source-built Package Analysis for $env_name
# Generated on: $(date)

EOF
    
    # Check for packages that are commonly built from source
    SOURCE_PACKAGES=("opencv" "cv2" "torch" "torchvision" "torchaudio" "tensorflow" "numpy" "scipy" "pillow" "matplotlib" "open3d" "mediapipe")
    
    for pkg in "${SOURCE_PACKAGES[@]}"; do
        python -c "
import sys
try:
    import importlib.util
    spec = importlib.util.find_spec('$pkg')
    if spec and spec.origin:
        print(f'$pkg location: {spec.origin}')
        
        # Try to get build info
        try:
            mod = importlib.import_module('$pkg')
            if hasattr(mod, '__version__'):
                print(f'$pkg version: {mod.__version__}')
            if hasattr(mod, '__file__'):
                print(f'$pkg file: {mod.__file__}')
            if hasattr(mod, '__path__'):
                print(f'$pkg path: {mod.__path__}')
                
            # Special cases for build info
            if '$pkg' == 'cv2':
                try:
                    print(f'OpenCV build info: {mod.getBuildInformation()}')
                except:
                    print('OpenCV build info not available')
            elif '$pkg' == 'torch':
                try:
                    print(f'PyTorch compiled with CUDA: {mod.cuda.is_available()}')
                    if mod.cuda.is_available():
                        print(f'PyTorch CUDA arch list: {mod.cuda.get_arch_list()}')
                except:
                    print('PyTorch CUDA info not available')
        except Exception as e:
            print(f'$pkg import error: {e}')
except ImportError:
    pass
except Exception as e:
    print(f'Error checking $pkg: {e}')
" >> "requirements/${env_name}-source-analysis.txt" 2>/dev/null
    done
    
    # Check pip show for detailed package info including installation method
    echo "  📖 Detailed package metadata..."
    pip list --format=freeze | cut -d'=' -f1 | while read package; do
        if [ ! -z "$package" ] && [ "$package" != "#" ] && [[ ! "$package" =~ ^- ]]; then
            pip show "$package" 2>/dev/null | grep -E "(Name|Version|Location|Requires|Required-by)" >> "requirements/${env_name}-detailed-packages.txt" || true
            echo "---" >> "requirements/${env_name}-detailed-packages.txt"
        fi
    done 2>/dev/null || echo "# Detailed package info not available" > "requirements/${env_name}-detailed-packages.txt"
    
    echo "  ✅ Environment $env_name analysis complete"
    return 0
}

# List of environments to analyze
ENVIRONMENTS=("aws" "whatsai2" "depth-pro")

echo "🔍 Checking available conda environments..."
echo "Available environments:"
conda env list

echo ""
echo "🔍 Starting extraction for environments: ${ENVIRONMENTS[*]}"

# Extract from each environment
for env_name in "${ENVIRONMENTS[@]}"; do
    echo ""
    echo "🔍 Extracting $env_name environment..."
    if extract_env_details "$env_name"; then
        echo "✅ Successfully extracted $env_name environment"
    else
        echo "⚠️  Failed to extract $env_name environment"
    fi
done

# Create a combined requirements file for Docker
echo "Creating combined requirements..."
cat > requirements/base-requirements.txt << EOF
# Base Python packages
fastapi
uvicorn[standard]
websockets
opencv-python-headless
pillow
numpy
torch
torchvision
transformers
accelerate
httpx
python-dotenv
asyncio-mqtt
pydantic
starlette

# Audio processing
soundfile
librosa

# Computer vision
mediapipe
ultralytics

# Point cloud processing
open3d

# Scientific computing
scipy
scikit-learn
matplotlib

# AWS SDK (if needed)
boto3

EOF

echo "Requirements extracted to requirements/ directory"
echo "Review and modify as needed before building Docker image"