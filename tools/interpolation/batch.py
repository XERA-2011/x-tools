"""
批量插帧入口

支持:
  - 批量 RIFE AI 插帧
  - 批量 FFmpeg 运动补偿插帧
"""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import INPUT_DIR, INTERPOLATION_TARGET_FPS, ensure_dirs
from tools.common import scan_videos, batch_process, print_summary, logger
from tools.interpolation.ffmpeg_minterp import interpolate_video_ffmpeg


def _batch_ffmpeg_worker(video_path: Path, **kwargs) -> dict:
    """批量 FFmpeg 插帧 worker"""
    return interpolate_video_ffmpeg(video_path, **kwargs)


def _batch_rife_worker(video_path: Path, **kwargs) -> dict:
    """批量 RIFE 插帧 worker"""
    from tools.interpolation.rife import interpolate_video_rife
    return interpolate_video_rife(video_path, **kwargs)


def batch_interpolate_ffmpeg(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    target_fps: float = INTERPOLATION_TARGET_FPS,
    mode: str = "mci",
) -> list[dict]:
    """
    批量 FFmpeg 插帧

    Args:
        input_dir: 输入目录
        target_fps: 目标帧率
        mode: 插帧模式
    """
    ensure_dirs()
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)
    results = batch_process(
        videos,
        _batch_ffmpeg_worker,
        desc=f"批量插帧 (FFmpeg → {target_fps:.0f}fps)",
        target_fps=target_fps,
        mode=mode,
    )
    print_summary(results)
    return results


def batch_interpolate_rife(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    multiplier: int | None = None,
    target_fps: float | None = None,
) -> list[dict]:
    """
    批量 RIFE 插帧

    Args:
        input_dir: 输入目录
        multiplier: 帧率倍数
        target_fps: 目标帧率
    """
    ensure_dirs()
    ensure_dirs()
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)
    results = batch_process(
        videos,
        _batch_rife_worker,
        desc="批量插帧 (RIFE)",
        multiplier=multiplier,
        target_fps=target_fps,
    )
    print_summary(results)
    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量插帧")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入视频目录")

    sub = parser.add_subparsers(dest="engine", required=True)

    # FFmpeg
    ffmpeg_p = sub.add_parser("ffmpeg", help="使用 FFmpeg 运动补偿")
    ffmpeg_p.add_argument("-t", "--target-fps", type=float, default=INTERPOLATION_TARGET_FPS, help="目标帧率")
    ffmpeg_p.add_argument("--mode", choices=["mci", "blend", "dup"], default="mci")

    # RIFE
    rife_p = sub.add_parser("rife", help="使用 RIFE AI 插帧")
    rife_fps_group = rife_p.add_mutually_exclusive_group()
    rife_fps_group.add_argument("-m", "--multiplier", type=int, help="帧率倍数 (2/4/8)")
    rife_fps_group.add_argument("-t", "--target-fps", type=float, help="目标帧率")

    args = parser.parse_args()

    if args.engine == "ffmpeg":
        batch_interpolate_ffmpeg(input_dir=args.input_dir, target_fps=args.target_fps, mode=args.mode)
    elif args.engine == "rife":
        batch_interpolate_rife(
            input_dir=args.input_dir,
            multiplier=getattr(args, "multiplier", None),
            target_fps=getattr(args, "target_fps", None),
        )
