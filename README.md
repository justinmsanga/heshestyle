# dinov2-embedding-worker

RunPod serverless worker that turns an image into a DINOv2 embedding vector, for HeSheStyle's "find similar styles" search.

## Input

```json
{ "input": { "image_url": "https://..." } }
```

or

```json
{ "input": { "image_base64": "..." } }
```

## Output

```json
{ "embedding": [0.0123, -0.0456, ...], "dim": 768 }
```

## Deploying

Connect this repo to a new RunPod serverless endpoint (Console → Serverless → New Endpoint → GitHub Repo). RunPod builds the Docker image server-side from the `Dockerfile` in this repo — no local Docker needed.

Recommended GPU: 16-24GB VRAM is ample (DINOv2-base is small); see the RunPod GPU catalog for current pricing/availability.
