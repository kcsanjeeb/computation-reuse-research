# Dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 \
    numpy tqdm boto3

WORKDIR /workspace
COPY train.py /workspace/train.py

# Default command for local test; Katib will override with args
CMD ["python", "train.py", "--use-fakedata"]
