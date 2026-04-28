# Free.ai Video Upscaler

Self-hosted Real-ESRGAN x4 video upscaler with temporal smoothing — a free,
GPU-backed alternative to Topaz Video Upscaler.

Powers the [Free.ai video upscaler tool](https://free.ai/video/upscale/).

[![PyPI](https://img.shields.io/pypi/v/free-video-upscaler)](https://pypi.org/project/free-video-upscaler/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

## What it does

Takes a video, upscales every frame 4× with Real-ESRGAN, applies a temporal
smoothing pass to eliminate frame-to-frame flicker, then re-encodes preserving
the source audio. The output is competitive with commercial video upscalers
on per-frame detail and temporal coherence.

## Why temporal smoothing matters

Frame-by-frame neural upscalers (the cheap way) introduce high-frequency
flicker because each frame is upscaled independently — a slight texture
detail in pixel (12,34) might be reconstructed differently by the model on
two consecutive frames, producing a shimmer. Topaz solves this by training a
video-aware model with temporal context.

We solve it cheaper: a 3-tap weighted average over consecutive output frames
(0.15 × prev + 0.7 × current + 0.15 × next) kills the high-freq flicker
without smearing motion. A scene-cut detector (per-pixel mean-abs-diff > 60)
skips smoothing when adjacent frames are too different — preserves crisp
edits, scene transitions, and very fast motion.

## Don't want to self-host? Use our hosted API

If you don't have a GPU or just want to try the upscaler without setting
anything up, hit our hosted endpoint — same code, our infra:

```bash
curl -X POST https://api.free.ai/v1/video/upscale/ \
  -H "Authorization: Bearer sk-free-..." \
  -F "file=@my-video.mp4" \
  -F "model=realesrgan" \
  -F "scale=2"
# Returns: {"video_url": "https://gpu4.free.ai/static/outputs/<job>.mp4", ...}
```

- Get an API key at [free.ai/api/](https://free.ai/api/)
- Free pool covers small clips daily; longer clips deduct from your token balance
- Same temporal-smoothed pipeline as this repo, just running on our GPUs
- Premium video-aware upscalers (Topaz, etc.) also available via the same
  endpoint with `model=premium/topaz/upscale/video`

For high-volume integrations, [contact us](https://free.ai/contact/) — we
do volume pricing for partners.

## Install (self-hosted)

```bash
# 1. Install torch with the right CUDA version for your driver.
#    Common choice for CUDA 12.x systems:
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. Install the upscaler:
pip install free-video-upscaler

# 3. Patch basicsr's torchvision import (newer torchvision removed
#    `functional_tensor` — known basicsr issue, no fix released yet):
python -c "import basicsr.data.degradations as _d, re, pathlib; \
    p = pathlib.Path(_d.__file__); \
    p.write_text(p.read_text().replace('torchvision.transforms.functional_tensor', \
                                       'torchvision.transforms.functional'))"

# 4. Download model weights:
mkdir -p ~/.realesrgan/weights && cd ~/.realesrgan/weights
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth
```

## CLI

```bash
free-upscaler --input my-video.mp4 --output upscaled.mp4 --scale 2
```

```
free-upscaler [-h] --input INPUT --output OUTPUT
              [--scale {2,4}] [--model {x4plus,anime}]
              [--tile TILE] [--no-temporal] [--fp32]
              [--weights WEIGHTS] [--quiet] [--version]
```

## Python API

```python
from free_video_upscaler import upscale_video

upscale_video(
    "in.mp4", "out.mp4",
    scale=2,             # 2 or 4
    model="x4plus",      # "x4plus" or "anime"
    temporal_smooth=True,
    progress=lambda i, n: print(f"{i}/{n}"),
)
```

## How it compares

| Feature | This package | Topaz Video AI | Naive Real-ESRGAN |
|---|---|---|---|
| Per-frame upscale | Real-ESRGAN x4plus | Proteus / Iris | Real-ESRGAN x4plus |
| Temporal coherence | 3-tap smoothing | Video-aware net | None (flicker) |
| Anime / cel mode | x4plus_anime_6B | Artemis HQ | Anime variant |
| Audio preserved | Yes (`-c:a copy`) | Yes | Manual |
| Cost | $0 (your GPU) | $299/yr | $0 |
| License | Apache 2.0 | Proprietary | BSD 3-Clause |

## Scaling notes

- Real-ESRGAN x4plus uses ~6 GB VRAM at 1080p with `tile=256`. Drop tile to
  128 for 8 GB cards, or pass `tile=0` for no tiling on big GPUs.
- Anime variant runs ~4× faster (6-block vs 23-block RRDBNet) and uses less
  VRAM. Use it for line-art / cel content.
- For 60 s+ clips on a single GPU, the package extracts + upscales + re-
  encodes synchronously; expect 10–20 minutes for a 1080p 60 s clip on a
  consumer GPU. Run it as a background job for anything longer.

## License

Apache-2.0. Real-ESRGAN model weights are themselves Apache-2.0 (see
[Tencent ARC's Real-ESRGAN repo](https://github.com/xinntao/Real-ESRGAN)).
