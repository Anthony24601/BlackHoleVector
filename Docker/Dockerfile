FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and other essential tools
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and CUDA
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies for Meta LLaMA
RUN pip3 install transformers sentencepiece

# Copy your Meta LLaMA model files if you have any
COPY ./meta-llama /workspace
COPY ./nano-llama31 /workspace

WORKDIR /workspace
