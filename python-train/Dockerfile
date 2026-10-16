# syntax=docker/dockerfile:1

################################################################################
# Mini-Pic Diffusion Training for RunPod (CUDA 13)
#
# Base image: RunPod PyTorch CUDA 13 template (Ubuntu 24.04, PyTorch, CUDA 13)
# Alternative base: runpod/base:1.4.0-cuda1320-ubuntu2404 (from yumon-pet)
# Can be overridden via: --build-arg BASE_IMAGE=...
################################################################################
ARG BASE_IMAGE=runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MINI_PIC_DATA_ROOT=/app/data/DiffusionDB \
    TOKENIZER_PATH=/app/tokenizer.json \
    CHECKPOINT_DIR=/workspace/checkpoints

WORKDIR /app

# Ensure python3 and pip are present if an alternative minimal base image is used
RUN if ! command -v pip3 >/dev/null 2>&1; then \
        apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv ca-certificates curl && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the 64x64 augmented dataset (~1.03 GB, 10 JSON parts + ~130k PNGs)
# Layered before code so code edits do not invalidate the 1GB dataset layer cache
ARG DATA_SRC=data/DiffusionDB/images-64-augmented
COPY ${DATA_SRC} ./data/DiffusionDB/images-64-augmented

# Copy tokenizer
COPY tokenizer.json .

# Copy training scripts
COPY *.py ./

# Preflight verification during docker build (CPU-safe) to catch any missing data early
RUN python3 -c "\
from dataloader import DiffusionDataset; \
ds = DiffusionDataset('data/DiffusionDB/images-64-augmented', 'data/DiffusionDB/images-64-augmented', 'tokenizer.json', max_samples=4); \
assert len(ds) >= 1, 'Dataset failed to load samples'; \
print(f'Build preflight passed: loaded {len(ds)} test samples successfully.')"

# Checkpoints land on /workspace/checkpoints when a RunPod network volume is mounted.
# Default command starts training immediately upon pod launch.
CMD ["python3", "train.py"]
