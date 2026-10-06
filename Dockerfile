FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CUDA-enabled wheels -- the host GPU's driver is provided by RunPod, this
# just needs a matching CUDA runtime baked into the torch wheel itself.
#
# cu128, not cu121: this worker's GPU pool (AMPERE_24) mixes older Ampere
# cards with newer Blackwell ones (confirmed live -- a real job landed on
# an RTX PRO 6000 Blackwell MIG slice). Blackwell's compute capability
# (sm_100/sm_120) only has compiled kernels in wheels built against CUDA
# 12.8+; a cu121 wheel runs fine on Ampere but fails on Blackwell with
# "no kernel image is available for execution on the device". cu128
# wheels are fat binaries covering both generations' compute
# capabilities, so this one build works across the whole pool.
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Bakes the model weights into the image at build time (not on first
# request) -- this is what keeps cold starts in the few-second range
# instead of however long a fresh Hugging Face download takes.
RUN python -c "from transformers import AutoImageProcessor, AutoModel; \
    AutoImageProcessor.from_pretrained('facebook/dinov2-base'); \
    AutoModel.from_pretrained('facebook/dinov2-base')"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
