"""Core upscaling pipeline: extract → Real-ESRGAN per frame → temporal
smoothing → re-encode preserving source audio.

Why subprocess-free here: the OSS package runs in whatever venv the
user has set up; we expect them to have torch + realesrgan installed
correctly per their own GPU. (Free.ai's own deployment uses a dedicated
venv-faceswap subprocess on top of this — that's an internal detail.)
"""
import os
import shutil
import subprocess
import sys
import tempfile

DEFAULT_X4PLUS_WEIGHTS = os.path.expanduser(
    "~/.realesrgan/weights/RealESRGAN_x4plus.pth"
)
DEFAULT_ANIME_WEIGHTS = os.path.expanduser(
    "~/.realesrgan/weights/RealESRGAN_x4plus_anime_6B.pth"
)


def _probe_fps(video):
    """ffprobe → r_frame_rate as float; fallback 30 fps."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video],
            capture_output=True, text=True, timeout=30,
        )
        s = (r.stdout or "").strip() or "30/1"
        num, den = s.split("/")
        return float(num) / float(den) if float(den) else 30.0
    except Exception:
        return 30.0


def _extract_frames(video, frames_dir):
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", video, "-vsync", "0",
         os.path.join(frames_dir, "f_%06d.png")],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "ffmpeg extract: " + ((proc.stderr or "").splitlines() or [""])[-1][:300]
        )
    return len([f for f in os.listdir(frames_dir) if f.endswith(".png")])


def _load_upsampler(weights, num_block, tile, fp16):
    """Return a RealESRGANer; falls back to CPU on GPU init failure."""
    import torch
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=num_block, num_grow_ch=32, scale=4,
    )
    try:
        return RealESRGANer(
            scale=4, model_path=weights, model=model,
            tile=tile, tile_pad=10, pre_pad=0,
            half=fp16 and torch.cuda.is_available(),
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    except (RuntimeError, AssertionError) as e:
        sys.stderr.write(f"GPU init failed ({e}); falling back to CPU\n")
        return RealESRGANer(
            scale=4, model_path=weights, model=model,
            tile=tile, tile_pad=10, pre_pad=0,
            half=False, device="cpu",
        )


def _upscale_frames(in_dir, out_dir, upsampler, tile_size, progress=None):
    """Per-frame Real-ESRGAN upscale. Drops tile by half on first OOM
    and retries before giving up."""
    import cv2
    import torch

    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".png"))
    n = len(files)
    for i, fname in enumerate(files):
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)
        img = cv2.imread(in_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        try:
            output, _ = upsampler.enhance(img, outscale=4)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                upsampler.tile = max(64, (tile_size or 256) // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                output, _ = upsampler.enhance(img, outscale=4)
            else:
                raise
        cv2.imwrite(out_path, output)
        if progress and i % 30 == 0:
            progress(i + 1, n)


def _temporal_smooth(frames_dir):
    """3-tap weighted blend over consecutive frames. Skips on scene
    cuts so transitions stay crisp.

    The blend weights (0.15 / 0.7 / 0.15) are tuned so adjacent frames
    contribute ~30% combined — enough to kill the high-freq flicker
    that's the #1 complaint with naive Real-ESRGAN frame-by-frame
    video upscale, but not so much that fast motion smears.

    Scene-cut threshold: per-pixel mean-abs-diff > 60 (8-bit space)
    means motion is too fast to blend safely; skip those frames.
    """
    import cv2
    import numpy as np

    files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".png"))
    if len(files) < 3:
        return
    paths = [os.path.join(frames_dir, f) for f in files]
    prev = cv2.imread(paths[0]).astype(np.float32)
    cur = cv2.imread(paths[1]).astype(np.float32)
    for i in range(1, len(paths) - 1):
        nxt = cv2.imread(paths[i + 1]).astype(np.float32)
        if (
            np.abs(cur - prev).mean() > 60
            or np.abs(cur - nxt).mean() > 60
        ):
            prev, cur = cur, nxt
            continue
        smoothed = (0.15 * prev + 0.70 * cur + 0.15 * nxt).clip(0, 255).astype(np.uint8)
        cv2.imwrite(paths[i], smoothed)
        prev = cur
        cur = nxt


def _encode(frames_dir, output, fps, target_scale, source_audio):
    """libx264 encode + audio stream-copy from source."""
    vf = "" if target_scale == 4 else "scale=iw/2:ih/2:flags=lanczos"
    cmd = [
        "ffmpeg", "-y", "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "f_%06d.png"),
        "-i", source_audio,
        "-map", "0:v:0", "-map", "1:a:0?",
    ]
    if vf:
        cmd += ["-vf", vf]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-crf", "18", "-preset", "veryfast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        output,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not os.path.exists(output):
        raise RuntimeError(
            "ffmpeg encode: " + ((proc.stderr or "").splitlines() or [""])[-1][:300]
        )


def upscale_video(
    input_video,
    output_video,
    scale=2,
    model="x4plus",
    tile=256,
    fp16=True,
    temporal_smooth=True,
    weights=None,
    progress=None,
):
    """Upscale a video file. Returns the output path on success.

    Args:
        input_video: path to source video.
        output_video: path to write the upscaled mp4.
        scale: 2 or 4 (network always emits 4×; we downsample for 2×).
        model: "x4plus" (photoreal, 23-block RRDBNet) or "anime"
            (6-block, 4× faster, sharper on cel art).
        tile: frame tile size for VRAM-tight setups; 0 = no tile.
        fp16: half-precision GPU inference (default True).
        temporal_smooth: 3-tap temporal blur to kill flicker.
        weights: override the default weights path.
        progress: optional callable(frame_idx, total) for progress.
    """
    if not os.path.exists(input_video):
        raise FileNotFoundError(input_video)
    if scale not in (2, 4):
        raise ValueError(f"scale must be 2 or 4, got {scale}")

    if model == "anime":
        weights = weights or DEFAULT_ANIME_WEIGHTS
        num_block = 6
    else:
        weights = weights or DEFAULT_X4PLUS_WEIGHTS
        num_block = 23
    if not os.path.exists(weights):
        raise FileNotFoundError(
            f"Real-ESRGAN weights not found at {weights}. "
            "Download with:\n"
            "  mkdir -p ~/.realesrgan/weights && cd ~/.realesrgan/weights\n"
            "  wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        )

    work = tempfile.mkdtemp(prefix="free_upscaler_")
    in_dir = os.path.join(work, "in")
    out_dir = os.path.join(work, "out")
    os.makedirs(in_dir)
    os.makedirs(out_dir)
    try:
        fps = _probe_fps(input_video)
        n = _extract_frames(input_video, in_dir)
        if n == 0:
            raise RuntimeError("ffmpeg extracted 0 frames (corrupt video?)")
        upsampler = _load_upsampler(weights, num_block, tile, fp16)
        _upscale_frames(in_dir, out_dir, upsampler, tile, progress)
        if temporal_smooth and n >= 3:
            _temporal_smooth(out_dir)
        _encode(out_dir, output_video, fps, scale, input_video)
        return output_video
    finally:
        shutil.rmtree(work, ignore_errors=True)
