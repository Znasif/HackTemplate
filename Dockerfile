# Single-stage build - let conda handle all dependencies
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONPATH=/app
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y \
    # Essential tools
    wget \
    ca-certificates \
    # Git (required for git dependencies in pip requirements)
    git \
    # Build essentials that some pip packages might need
    build-essential \
    libxml2 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Clean up to minimize image size
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -ya

# Initialize conda
# Note: `conda init` modifies shell startup files like .bashrc, which are not
# sourced by default in non-interactive RUN/CMD shells. This is why scripts
# need `eval "$(conda shell.bash hook)"`.
RUN conda init bash

# --- Runtime setup ---
WORKDIR /app

# Copy application code and necessary scripts
COPY . /app/
# --- Environment Creation ---
# Copy resources and scripts needed for environment setup
COPY resources/ /tmp/
RUN git clone --recursive https://github.com/cvg/Hierarchical-Localization.git /app/Hierarchical-Localization
# Create conda environments
RUN chmod +x /tmp/install_environments.sh
RUN /tmp/install_environments.sh aws
RUN /tmp/install_environments.sh whatsai

RUN conda run -n whatsai pip install -e /app/Hierarchical-Localization
RUN conda run -n whatsai pip install huggingface_hub[cli]
RUN conda run -n whatsai huggingface-cli download microsoft/Florence-2-large \
    --include="*" \
    --resume-download

COPY --chmod=755 resources/start_server.sh /app/start_server.sh

# Create models directory (this is a good practice, creates the mount point)
RUN mkdir -p /app/models /app/audio_recordings /app/logs

RUN conda run -n whatsai conda run -n whatsai huggingface-cli download Ultralytics/YOLO11 yolo11n-seg.pt --local-dir /app/models --resume-download

# Expose port
EXPOSE 8000

# Start the application using the start script.
# This will be overridden by the docker-compose `command` if present.
CMD ["/app/start_server.sh"]