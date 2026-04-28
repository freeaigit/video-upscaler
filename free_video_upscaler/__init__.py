"""Free.ai video upscaler — Real-ESRGAN x4 with temporal smoothing.

Public API:
    from free_video_upscaler import upscale_video

    upscale_video("input.mp4", "out.mp4", scale=2, model="x4plus")

CLI:
    free-upscaler --input input.mp4 --output out.mp4

Repo: https://github.com/freeaigit/video-upscaler
"""
from free_video_upscaler.core import upscale_video

__version__ = "0.1.0"
__all__ = ["upscale_video", "__version__"]
