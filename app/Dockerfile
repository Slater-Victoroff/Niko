## Use an official PyTorch image that includes CUDA; adjust the tag if you need a
## different CUDA version (11.8 is a common choice). The host must have the
## NVIDIA container toolkit installed for GPU access.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps for torch and common packages (add curl for the-well downloader)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

CMD ["bash"]
