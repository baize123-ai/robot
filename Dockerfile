FROM python:3.9-slim

# Install system deps for MuJoCo rendering and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    libgl1-mesa-dev \
    libosmesa6 \
    libglfw3-dev \
    libglew-dev \
    xvfb \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /workspace

# Copy requirements and install python deps
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt
# Ensure MuJoCo python bindings and viewer are available
RUN pip install mujoco mujoco-viewer imageio

# Copy project files (if building from local context)
COPY . /workspace
WORKDIR /workspace

# Helpful entrypoint: run a shell by default. Use xvfb-run for headless rendering when needed.
CMD ["bash"]
