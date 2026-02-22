"""
批量内容截取入口

支持批量操作:
  - 批量截取视频片段 (相同时间范围)
  - 批量提取关键帧
  - 批量按间隔提取帧
"""
from pathlib import Path


from config import INPUT_DIR, ensure_dirs
from tools.common import scan_videos, batch_process, print_summary, logger
from tools.extract.clip_extractor import extract_clip
from tools.extract.keyframe_extractor import (
    extract_keyframes,
    extract_frames_interval,
    extract_frames_scene_change,
)


def batch_extract_clips(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    start: str = "0",
    end: str | None = None,
    duration: str | None = None,
    reencode: bool = False,
    **kwargs,
) -> list[dict]:
    """
    批量截取视频片段 — 对 input 目录下所有视频应用相同的截取参数

    Args:
        input_dir: 输入目录 (默认: config.INPUT_DIR)
        start: 开始时间
        end: 结束时间
        duration: 持续时长
        reencode: 是否重新编码
    """
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)
    results = batch_process(
        videos,
        extract_clip,
        desc="批量截取片段",
        start=start, end=end, duration=duration,
        reencode=reencode,
        **kwargs,
    )
    print_summary(results)
    return results


def batch_extract_keyframes(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    **kwargs,
) -> list[dict]:
    """批量提取关键帧"""
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)
    results = batch_process(
        videos,
        extract_keyframes,
        desc="批量提取关键帧",
        **kwargs,
    )
    print_summary(results)
    return results


def batch_extract_interval(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    interval: float = 10.0,
    **kwargs,
) -> list[dict]:
    """批量按间隔提取帧"""
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)
    results = batch_process(
        videos,
        extract_frames_interval,
        desc=f"批量提取帧 (间隔 {interval}s)",
        interval=interval,
        **kwargs,
    )
    print_summary(results)
    return results


def batch_extract_scenes(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    threshold: float = 0.3,
    **kwargs,
) -> list[dict]:
    """批量按场景切换提取帧"""
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)
    results = batch_process(
        videos,
        extract_frames_scene_change,
        desc="批量提取场景帧",
        threshold=threshold,
        **kwargs,
    )
    print_summary(results)
    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量内容截取")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入视频目录")

    sub = parser.add_subparsers(dest="mode", required=True)

    # 批量截取片段
    clip_p = sub.add_parser("clip", help="批量截取片段")
    clip_p.add_argument("-s", "--start", required=True, help="开始时间")
    clip_p.add_argument("-e", "--end", help="结束时间")
    clip_p.add_argument("-d", "--duration", help="持续时长")
    clip_p.add_argument("--reencode", action="store_true", help="重新编码")

    # 批量关键帧
    sub.add_parser("keyframes", help="批量提取关键帧")

    # 批量间隔帧
    int_p = sub.add_parser("interval", help="批量按间隔提取帧")
    int_p.add_argument("--seconds", type=float, default=1.0, help="间隔 (秒)")

    # 批量场景帧
    scene_p = sub.add_parser("scene", help="批量按场景切换提取帧")
    scene_p.add_argument("--threshold", type=float, default=0.3, help="阈值 (0-1)")

    args = parser.parse_args()
    ensure_dirs()

    if args.mode == "clip":
        batch_extract_clips(input_dir=args.input_dir, start=args.start, end=args.end, duration=args.duration, reencode=args.reencode)
    elif args.mode == "keyframes":
        batch_extract_keyframes(input_dir=args.input_dir)
    elif args.mode == "interval":
        batch_extract_interval(input_dir=args.input_dir, interval=args.seconds)
    elif args.mode == "scene":
        batch_extract_scenes(input_dir=args.input_dir, threshold=args.threshold)
