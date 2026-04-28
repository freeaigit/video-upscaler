"""`free-upscaler` CLI entry point."""
import argparse
import json
import sys
import traceback

from free_video_upscaler import __version__, upscale_video


def main():
    ap = argparse.ArgumentParser(
        prog="free-upscaler",
        description="Free.ai video upscaler — Real-ESRGAN x4 with temporal smoothing.",
    )
    ap.add_argument("--input", required=True, help="input video path")
    ap.add_argument("--output", required=True, help="output video path")
    ap.add_argument(
        "--scale", type=int, default=2, choices=[2, 4],
        help="final scale factor (network outputs 4×; we downsample for 2×)",
    )
    ap.add_argument(
        "--model", default="x4plus", choices=["x4plus", "anime"],
        help="x4plus = photoreal RRDBNet 23-block; anime = 6-block, faster",
    )
    ap.add_argument("--tile", type=int, default=256, help="frame tile size")
    ap.add_argument(
        "--no-temporal", action="store_true",
        help="disable the 3-tap temporal smoothing pass",
    )
    ap.add_argument(
        "--fp32", action="store_true",
        help="disable FP16 inference (slightly higher quality, slower)",
    )
    ap.add_argument("--weights", default=None, help="override default weights path")
    ap.add_argument("--quiet", action="store_true", help="suppress progress output")
    ap.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    args = ap.parse_args()

    def progress(i, n):
        if not args.quiet:
            sys.stderr.write(f"frame {i}/{n}\n")
            sys.stderr.flush()

    try:
        out = upscale_video(
            args.input, args.output,
            scale=args.scale, model=args.model, tile=args.tile,
            fp16=not args.fp32,
            temporal_smooth=not args.no_temporal,
            weights=args.weights,
            progress=progress,
        )
        print(json.dumps({"ok": True, "output": out}))
        return 0
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
