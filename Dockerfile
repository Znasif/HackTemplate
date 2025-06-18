# Single-stage build - let conda handle all dependencies
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONPATH=/app:$PYTHONPATH
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install ONLY the absolute bare minimum that conda cannot provide
RUN apt-get update && apt-get install -y \
    # Essential tools (wget needed for miniconda download)
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -ya

# Initialize conda
RUN conda init bash && \
    echo "conda activate base" >> ~/.bashrc

# --- Environment Creation (conda handles all build dependencies) ---
# Copy requirements and scripts
COPY resources/requirements/ /tmp/requirements/
COPY resources/install_environments.sh /tmp/install_environments.sh

# Create conda environments - each handles its own build dependencies
RUN chmod +x /tmp/install_environments.sh
RUN /tmp/install_environments.sh aws
RUN /tmp/install_environments.sh whatsai2
RUN /tmp/install_environments.sh depth-pro

# --- Runtime setup ---
WORKDIR /app

# Copy application code
COPY . /app/
COPY resources/start_server.sh /app/start_server.sh
COPY resources/start_processor.py /app/start_processor.py

# Create models directory
RUN mkdir -p /app/models

# Make scripts executable
RUN chmod +x /app/start_server.sh

# Expose port
EXPOSE 8080

# Start the application
CMD ["/app/start_server.sh"]