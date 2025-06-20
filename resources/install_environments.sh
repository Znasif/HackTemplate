#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <environment_name>"
    exit 1
fi

ENV_NAME=$1
PROJECT_DIR="/tmp/${ENV_NAME}"
TOML_FILE="${PROJECT_DIR}/pyproject.toml"

if [ ! -d "$PROJECT_DIR" ] || [ ! -f "$TOML_FILE" ]; then
    echo "Error: Project directory or pyproject.toml not found at ${PROJECT_DIR}"
    exit 1
fi

echo "--- Preparing to install environment: ${ENV_NAME} ---"

# --- CORRECTED PARSING LOGIC ---
# This is the robust fix. It uses awk to select only the text inside the
# double quotes, safely ignoring comments or other text on the line.
PYTHON_VERSION=$(grep 'requires-python' "$TOML_FILE" | awk -F'"' '{print $2}' | tr -d '>= ')

PROJECT_NAME=$(grep -E '^name\s*=' "$TOML_FILE" | head -n 1 | awk -F'"' '{print $2}')

if [ -z "$PYTHON_VERSION" ]; then
    echo "Error: Could not find a valid 'requires-python' value in ${TOML_FILE}"
    exit 1
fi

if [ -z "$PROJECT_NAME" ]; then
    echo "Error: Could not find a valid project 'name' value in ${TOML_FILE}"
    exit 1
fi

echo "Found project name: '${PROJECT_NAME}'"
echo "Found required Python version: '${PYTHON_VERSION}'"

echo "Creating Conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

echo "Installing dependencies from ${TOML_FILE} into '${ENV_NAME}' environment..."
conda run -n "$ENV_NAME" pip install "$PROJECT_DIR"

echo "--- Successfully created and provisioned environment: ${ENV_NAME} ---"