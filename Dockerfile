# Multi-stage build for handling source-built dependencies
# Stage 1: Build environment with all compilation tools
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH

# Install comprehensive build dependencies
RUN apt-get update && apt-get install -y \
    # Basic tools
    wget curl git unzip \
    # Build essentials
    build-essential cmake ninja-build \
    # Python development
    python3-dev python3-pip \
    # OpenCV dependencies
    libopencv-dev libopencv-contrib-dev \
    libgtk-3-dev libavcodec-dev libavformat-dev libswscale-dev \
    libv4l-dev libxvidcore-dev libx264-dev libjpeg-dev libpng-dev libtiff-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    # Audio processing
    ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0 \
    libasound2-dev portaudio19-dev libsndfile1-dev \
    # OpenGL and display
    libglu1-mesa-dev libgl1-mesa-dev libegl1-mesa-dev \
    libxrandr2 libxss1 libxcursor1 libxcomposite1 libxi6 libxtst6 \
    # Additional scientific computing dependencies
    libatlas-base-dev liblapack-dev libblas-dev \
    libhdf5-dev libprotobuf-dev protobuf-compiler \
    # Open3D dependencies
    libeigen3-dev libflann-dev libvtk9-dev \
    # Networking
    libssl-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -ya

# Initialize conda
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# Stage 2: Runtime environment
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    # Runtime essentials
    curl wget git \
    # Python runtime
    python3 python3-pip \
    # Required runtime libraries (without dev packages)
    libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d \
    libgtk-3-0 libavcodec58 libavformat58 libswscale5 \
    # Audio runtime
    ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0 \
    libasound2 portaudio19-dev libsndfile1 \
    # OpenGL runtime
    libglu1-mesa libgl1-mesa-glx libegl1-mesa \
    libxrandr2 libxss1 libxcursor1 libxcomposite1 libxi6 libxtst6 \
    # Scientific computing runtime
    libatlas3-base liblapack3 libblas3 \
    libhdf5-103-1 libprotobuf23 \
    # Open3D runtime
    libeigen3-dev libflann1.9 \
    # Networking
    libssl3 libffi8 \
    && rm -rf /var/lib/apt/lists/*

# Copy conda from builder stage
COPY --from=builder /opt/conda /opt/conda

# Initialize conda in runtime
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# --- Build-time steps: Create environments and install fallbacks ---
# Copy build-time scripts and resources to a temporary location
COPY resources/requirements/ /tmp/requirements/
COPY resources/install_environments.sh /tmp/install_environments.sh

# Run the build-time scripts
RUN chmod +x /tmp/install_environments.sh
RUN /tmp/install_environments.sh aws
RUN /tmp/install_environments.sh whatsai2
RUN /tmp/install_environments.sh depth-pro

# --- Runtime setup ---
# Set working directory for the final application
WORKDIR /app

# Copy application code and all runtime scripts to the WORKDIR
COPY . /app/
COPY resources/start_server.sh /app/start_server.sh
COPY resources/start_processor.py /app/start_processor.py

# Create models directory
RUN mkdir -p /app/models

# Make the main startup script executable
RUN chmod +x /app/start_server.sh

# Expose port
EXPOSE 8080

# Start the application
CMD ["/app/start_server.sh"]