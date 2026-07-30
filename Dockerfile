FROM python:3.9-slim

# Install system deps for pybullet and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    ca-certificates \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python deps
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip
RUN pip install -r /workspace/requirements.txt

# Copy project files (if building from local context)
COPY . /workspace
WORKDIR /workspace

CMD ["bash"]
