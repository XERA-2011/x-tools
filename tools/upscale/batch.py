"""
批量高清重置入口

支持:
  - 批量 Real-ESRGAN 超分
  - 批量 FFmpeg 传统放大
"""
from pathlib import Path


from config import INPUT_DIR, UPSCALE_FACTOR, ensure_dirs
from tools.common import scan_videos, batch_process, print_summary, logger
from tools.upscale.ffmpeg_scale import upscale_video_ffmpeg


def _batch_ffmpeg_worker(video_path: Path, **kwargs) -> dict:
    """批量 FFmpeg 放大 worker"""
    return upscale_video_ffmpeg(video_path, **kwargs)


def _batch_realesrgan_worker(video_path: Path, **kwargs) -> dict:
    """批量 Real-ESRGAN 超分 worker"""
    from tools.upscale.realesrgan import upscale_video_realesrgan
    return upscale_video_realesrgan(video_path, **kwargs)


def batch_upscale_ffmpeg(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    scale: int | None = UPSCALE_FACTOR,
    width: int | None = None,
    height: int | None = None,
    algorithm: str = "lanczos",
    crf: int = 18,
    **kwargs,
) -> list[dict]:
    """
    批量 FFmpeg 放大

    Args:
        input_dir: 输入目录
        scale: 放大倍数, 与 width/height 二选一
        width: 目标宽度
        height: 目标高度
        algorithm: 插值算法
        crf: 输出质量
    """
    ensure_dirs()
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)

    if width and height:
        desc = f"批量放大 (FFmpeg → {width}x{height})"
    else:
        desc = f"批量放大 (FFmpeg {scale}x)"

    results = batch_process(
        videos,
        _batch_ffmpeg_worker,
        desc=desc,
        scale=scale,
        width=width,
        height=height,
        algorithm=algorithm,
        crf=crf,
        **kwargs,
    )
    print_summary(results)
    return results


def batch_upscale_realesrgan(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    scale: int | None = None,
    device: str | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    **kwargs,
) -> list[dict]:
    """
    批量 Real-ESRGAN 超分

    Args:
        input_dir: 输入目录
        scale: 放大倍数 (2 or 4), 与 target_width/target_height 二选一
        device: 推理设备
        target_width: 目标宽度 (自动选择最佳 AI 倍数)
        target_height: 目标高度
    """
    ensure_dirs()
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)

    if target_width and target_height:
        desc = f"批量超分 (Real-ESRGAN → {target_width}x{target_height})"
    else:
        desc = f"批量超分 (Real-ESRGAN {scale or UPSCALE_FACTOR}x)"

    results = batch_process(
        videos,
        _batch_realesrgan_worker,
        desc=desc,
        scale=scale,
        device=device,
        target_width=target_width,
        target_height=target_height,
        **kwargs,
    )
    print_summary(results)
    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量高清重置")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入视频目录")

    sub = parser.add_subparsers(dest="engine", required=True)

    # FFmpeg
    ffmpeg_p = sub.add_parser("ffmpeg", help="使用 FFmpeg 传统放大")
    ffmpeg_p.add_argument("-s", "--scale", type=int, default=UPSCALE_FACTOR, help="放大倍数")
    ffmpeg_p.add_argument("--algorithm", default="lanczos", choices=["lanczos", "bicubic", "bilinear", "spline"])
    ffmpeg_p.add_argument("--crf", type=int, default=18, help="输出质量")

    # Real-ESRGAN
    esrgan_p = sub.add_parser("realesrgan", help="使用 Real-ESRGAN AI 超分")
    esrgan_p.add_argument("-s", "--scale", type=int, choices=[2, 4], default=UPSCALE_FACTOR, help="放大倍数")
    esrgan_p.add_argument("--device", choices=["mps", "cpu", "cuda"], help="推理设备")

    args = parser.parse_args()

    if args.engine == "ffmpeg":
        batch_upscale_ffmpeg(input_dir=args.input_dir, scale=args.scale, algorithm=args.algorithm, crf=args.crf)
    elif args.engine == "realesrgan":
        batch_upscale_realesrgan(input_dir=args.input_dir, scale=args.scale, device=args.device)
