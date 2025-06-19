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

# Function to check if a line is a local path that should be filtered
is_local_path() {
    local line="$1"
    
    # Check for file:// URLs
    if [[ "$line" =~ file:// ]]; then
        return 0  # Is local path
    fi
    
    # Check for absolute local paths (starts with /)
    if [[ "$line" =~ ^[^-].*@[[:space:]]*/[^[:space:]]+ ]]; then
        return 0  # Is local path
    fi
    
    # Check for relative local paths
    if [[ "$line" =~ ^[^-].*@[[:space:]]*\./[^[:space:]]+ ]]; then
        return 0  # Is local path
    fi
    
    # Check for editable installs pointing to local directories
    if [[ "$line" =~ ^-e[[:space:]]+/[^[:space:]]+ ]]; then
        return 0  # Is local path
    fi
    
    if [[ "$line" =~ ^-e[[:space:]]+\./[^[:space:]]+ ]]; then
        return 0  # Is local path
    fi
    
    return 1  # Not a local path
}

# Function to extract package name from various pip formats
extract_package_name() {
    local line="$1"
    
    # Handle editable git installs: -e git+https://...#egg=package_name
    if [[ "$line" =~ ^-e[[:space:]]+git\+.*#egg=([^[:space:]]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    
    # Handle git installs: package @ git+https://...
    if [[ "$line" =~ ^([^[:space:]@]+)[[:space:]]*@[[:space:]]*git\+ ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    
    # Handle standard pip installs: package==version
    if [[ "$line" =~ ^([^[:space:]@=]+)== ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    
    # Handle pip installs with @ version: package @ version
    if [[ "$line" =~ ^([^[:space:]@]+)[[:space:]]*@[[:space:]]*[^git] ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    
    # Fallback: extract first word/identifier
    if [[ "$line" =~ ^([a-zA-Z0-9_-]+) ]]; then
        echo "${BASH_REMATCH[1]}"
        return 0
    fi
    
    return 1
}

# Function to check if a dependency is git-based
is_git_dependency() {
    local line="$1"
    
    # Check for git+ URLs
    if [[ "$line" =~ git\+ ]]; then
        return 0  # Is git dependency
    fi
    
    # Check for github/gitlab URLs
    if [[ "$line" =~ (github\.com|gitlab\.com) ]]; then
        return 0  # Is git dependency
    fi
    
    return 1  # Not git dependency
}

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
    
    # Generate cleaned pip requirements instead of raw pip freeze
    echo "  🧹 Generating cleaned pip requirements..."
    
    # First get explicit conda package list for filtering
    conda list --explicit > "requirements/${env_name}-explicit.txt"
    local conda_packages=$(grep -o '[^/]*\.conda$\|[^/]*\.tar\.bz2$' "requirements/${env_name}-explicit.txt" | sed 's/-[0-9].*//' | sort | uniq)
    
    # Create cleaned pip requirements
    cat > "requirements/${env_name}-pip.txt" << EOF
# Cleaned pip requirements for $env_name environment
# Auto-generated by extract_requirements.sh
# Only includes PyPI packages not managed by conda, excluding local file paths
# Git dependencies are preserved as they can be handled by install_environments.sh

EOF
    
    # Process pip freeze output with improved filtering
    pip freeze | while IFS= read -r line; do
        # Skip empty lines and comments
        if [[ -z "$line" || "$line" =~ ^# ]]; then
            continue
        fi
        
        # Skip local paths but preserve git dependencies
        if is_local_path "$line"; then
            echo "  🚫 Filtering local path: $line" >&2
            continue
        fi
        
        # Extract package name using improved logic
        local package_name
        if ! package_name=$(extract_package_name "$line"); then
            echo "  ⚠️  Could not extract package name from: $line" >&2
            continue
        fi
        
        # Skip if package is managed by conda (but preserve git dependencies)
        if ! is_git_dependency "$line" && echo "$conda_packages" | grep -q "^${package_name}$"; then
            echo "  📦 Skipping conda-managed package: $package_name" >&2
            continue
        fi
        
        # # Skip packages that should prefer conda versions (unless they're git dependencies)
        # if ! is_git_dependency "$line"; then
        #     case "$package_name" in
        #         numpy|scipy|matplotlib|pillow|torch|torchvision|opencv-python|opencv-contrib-python|opencv-python-headless)
        #             echo "  🔄 Skipping conda-preferred package: $package_name" >&2
        #             continue
        #             ;;
        #     esac
        # fi

        local required_by=$(pip show "$package_name" 2>/dev/null | grep "Required-by:" | cut -d: -f2 | tr -d ' ')
        if [ ! -z "$required_by" ] && [ "$required_by" != "" ]; then
            echo "  📎 Skipping $package_name (required by: $required_by)"
            continue
        fi
        
        # Keep the package
        echo "$line" >> "requirements/${env_name}-pip.txt"
        
        # Log what we're keeping
        if is_git_dependency "$line"; then
            echo "  📚 Preserving git dependency: $package_name" >&2
        else
            echo "  ✅ Including pip package: $package_name" >&2
        fi
        
    done
    
    echo "  🔍 Detailed package analysis..."
    # Detailed conda package info (shows build strings and channels)
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
        nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 >> "requirements/${env_name}-system-info.txt" || echo "# NVIDIA driver info not available" >> "requirements/${env_name}-system-info.txt"
    else
        echo "# NVIDIA GPU not available" >> "requirements/${env_name}-system-info.txt"
    fi
    
    echo "  🔧 Source-built package analysis..."
    # Analyze packages for build information
    echo "# Source Analysis for $env_name environment" > "requirements/${env_name}-source-analysis.txt"
    echo "# Generated on: $(date)" >> "requirements/${env_name}-source-analysis.txt"
    echo "" >> "requirements/${env_name}-source-analysis.txt"
    
    # Key packages that might be source-built
    local key_packages=("cv2" "torch" "torchvision" "numpy" "scipy" "pillow" "matplotlib")
    
    for pkg in "${key_packages[@]}"; do
        python -c "
import sys
import importlib.util
try:
    spec = importlib.util.find_spec('$pkg')
    if spec is not None:
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
        
        # Show summary of cleaned pip requirements
        pip_count=$(grep -v '^#' "requirements/${env_name}-pip.txt" | grep -v '^$' | wc -l)
        git_count=$(grep -v '^#' "requirements/${env_name}-pip.txt" | grep -E "(git\+|@ git)" | wc -l)
        
        echo "  📊 Generated $pip_count cleaned pip packages ($git_count git dependencies)"
        
        if [ $pip_count -gt 0 ]; then
            echo "  🔍 Key pip packages:"
            grep -v '^#' "requirements/${env_name}-pip.txt" | grep -v '^$' | head -5 | sed 's/^/    /'
            if [ $pip_count -gt 5 ]; then
                echo "    ... and $((pip_count - 5)) more"
            fi
            
            if [ $git_count -gt 0 ]; then
                echo "  📚 Git dependencies found:"
                grep -E "(git\+|@ git)" "requirements/${env_name}-pip.txt" | sed 's/^/    /'
            fi
        fi
    else
        echo "⚠️  Failed to extract $env_name environment"
    fi
done

echo ""
echo "✅ Requirements extracted to requirements/ directory"
echo "📋 Files generated for each environment:"
echo "  - {env}-explicit.txt       (exact conda package URLs)"
echo "  - {env}-pip.txt            (cleaned PyPI packages + git dependencies)"
echo "  - {env}.yml                (conda environment file)"
echo "  - {env}-packages.json      (detailed package metadata)"
echo "  - {env}-git-packages.txt   (git and editable packages)"
echo "  - {env}-source-analysis.txt (source-built package info)"
echo ""
echo "🎯 Key improvements:"
echo "  - Better filtering of local file paths"
echo "  - Preserved git dependencies for install_environments.sh"
echo "  - Improved package name extraction"
echo "  - Enhanced logging for debugging"
echo ""
echo "🎉 Ready for Docker build with properly cleaned requirements!"