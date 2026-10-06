"""
RunPod serverless worker: image -> DINOv2 embedding vector.

Input:  {"input": {"image_url": "https://..."}}
Output: {"embedding": [768 floats], "dim": 768}

Used by HeSheStyle's "find similar styles" feature -- every barber-posted
trend photo gets embedded once (backfill + on new posts), a customer's
uploaded photo gets embedded the same way at search time, and Postgres
(pgvector) finds the closest matches by cosine distance. DINOv2 was chosen
over CLIP specifically for this: it has no text-alignment objective, so it
preserves fine-grained visual detail (exact braid/fade shape) that CLIP's
text-matching training tends to wash out -- see the 70% vs 15% gap on
fine-grained visual classification benchmarks that motivated this pick.
"""

import base64
import io

import requests
import runpod
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-base"

# Loaded once per worker (not per request) -- this is what makes a warm
# worker fast; a cold worker pays this cost once before its first job.
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()


def load_image(job_input: dict) -> Image.Image:
    if job_input.get("image_url"):
        resp = requests.get(job_input["image_url"], timeout=20)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    if job_input.get("image_base64"):
        raw = base64.b64decode(job_input["image_base64"])
        return Image.open(io.BytesIO(raw)).convert("RGB")
    raise ValueError("Provide either image_url or image_base64.")


def handler(event):
    job_input = event.get("input", {})
    try:
        image = load_image(job_input)
    except Exception as e:
        return {"error": f"Couldn't load the image: {e}"}

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token embedding -- DINOv2's standard whole-image representation.
        embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)
        # L2-normalize so Postgres can compare embeddings with plain cosine
        # distance (pgvector's <=> operator) without re-normalizing later.
        embedding = embedding / embedding.norm()

    return {"embedding": embedding.cpu().tolist(), "dim": embedding.shape[0]}


runpod.serverless.start({"handler": handler})
