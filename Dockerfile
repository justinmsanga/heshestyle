FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CUDA-enabled wheels -- the host GPU's driver is provided by RunPod, this
# just needs a matching CUDA runtime baked into the torch wheel itself.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Bakes the model weights into the image at build time (not on first
# request) -- this is what keeps cold starts in the few-second range
# instead of however long a fresh Hugging Face download takes.
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
    AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
    AutoModel.from_pretrained('facebook/dinov2-base')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
