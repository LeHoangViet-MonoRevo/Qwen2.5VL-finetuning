FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10, pip, FFmpeg, and build tools
RUN apt update && apt install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    ffmpeg \
    git \
    sudo \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN python3.10 -m pip install --upgrade pip


# Install build dependencies first
RUN pip install --no-cache-dir packaging wheel setuptools cmake ninja \
    torch==2.7.1 \
    torchvision==0.22.1\
    torchaudio==2.7.1

# Install Python packages
RUN pip install --no-cache-dir \
    decord==0.6.0 \
    autoawq==0.2.9 \
    torchcodec==0.5 \
    deepspeed==0.17.3 \
    transformers==4.51.3 \
    flash_attn==2.8.2

# Create a developer user (optional but recommended)
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Set working directory (this will be mounted)
WORKDIR /workspace

# Switch to non-root user
USER $USERNAME

# Default command
CMD ["bash"]

