# Define a build-time argument to switch between CPU and GPU installation
# Must be declared before the first FROM to be usable in FROM.
ARG RUNTIME=nvidia

# Select base image based on RUNTIME:
# - nvidia: full CUDA runtime (~1.5GB), required for GPU inference
# - cpu:    plain Ubuntu 24.04 (~80MB), sufficient for CPU-only inference
FROM nvidia/cuda:12.1.1-runtime-ubuntu24.04 AS base-nvidia
FROM ubuntu:24.04 AS base-cpu
FROM base-${RUNTIME} AS final

# Set environment variables for Python and Hugging Face
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
# Set the Hugging Face home directory to a path inside the container for better caching
ENV HF_HOME=/app/hf_cache

# Install system dependencies required for the application
# Ubuntu 24.04 ships Python 3.12 natively — no PPA needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    espeak-ng \
    python3.12 \
    python3.12-dev \
    python3-pip \
    && ln -sf /usr/bin/python3.12 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy requirements files first to leverage Docker's layer caching
COPY requirements.txt .
COPY requirements-nvidia.txt .

# Install Python dependencies
# --break-system-packages is required on Ubuntu 24.04 (PEP 668)
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# --- Conditionally Install GPU Dependencies ---
# If the RUNTIME argument is 'nvidia', install the specific GPU packages
# This mirrors the robust manual installation process.
RUN if [ "$RUNTIME" = "nvidia" ]; then \
    echo "RUNTIME=nvidia, installing GPU dependencies..."; \
    pip3 install --no-cache-dir --break-system-packages onnxruntime-gpu; \
    pip3 install --no-cache-dir --break-system-packages torch torchaudio --index-url https://download.pytorch.org/whl/cu121; \
    pip3 install --no-cache-dir --break-system-packages -r requirements-nvidia.txt; \
    else \
    echo "RUNTIME=cpu, skipping GPU dependencies."; \
    fi

# Copy the rest of the application code into the container
COPY . .

# Create required directories for the application data
RUN mkdir -p model_cache outputs logs hf_cache

# Expose the port the application will run on (aligned with docker-compose.yml)
EXPOSE 8005

# The command to run when the container starts
CMD ["python", "server.py"]
