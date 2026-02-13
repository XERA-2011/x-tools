"""
关键帧提取模块

功能:
  - 提取视频 I-帧 (关键帧) 为图片
  - 按固定间隔提取帧
  - 按场景切换提取帧
"""
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    FFMPEG_BIN, OUTPUT_EXTRACT,
    KEYFRAME_IMAGE_FORMAT, KEYFRAME_IMAGE_QUALITY,
)
from tools.common import get_video_info, logger


def extract_keyframes(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    image_format: str = KEYFRAME_IMAGE_FORMAT,
    quality: int = KEYFRAME_IMAGE_QUALITY,
) -> dict:
    """
    提取视频的 I-帧 (关键帧) 为图片

    Args:
        video_path: 输入视频路径
        output_dir: 输出目录, 为 None 时自动生成
        image_format: 图片格式 (jpg/png)
        quality: 图片质量 (1-100, 仅 jpg 有效)

    Returns:
        dict: {"output_dir": str, "count": int}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 创建输出目录
    if output_dir is None:
        output_dir = OUTPUT_EXTRACT / f"{video_path.stem}_keyframes"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件名模板
    output_pattern = str(output_dir / f"frame_%04d.{image_format}")

    # 构建 FFmpeg 命令 — 只提取 I-帧
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", "select=eq(pict_type\\,I)",
        "-vsync", "vfr",
    ]

    # JPG 质量设置
    if image_format == "jpg":
        cmd += ["-qscale:v", str(max(1, min(31, 32 - int(quality * 31 / 100))))]

    cmd += [output_pattern]

    logger.info(f"提取关键帧: {video_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    # 统计输出文件数
    frames = list(output_dir.glob(f"frame_*.{image_format}"))
    logger.info(f"✅ 提取了 {len(frames)} 个关键帧 → {output_dir}")

    return {"output_dir": str(output_dir), "count": len(frames)}


def extract_frames_interval(
    video_path: str | Path,
    interval: float = 1.0,
    output_dir: str | Path | None = None,
    image_format: str = KEYFRAME_IMAGE_FORMAT,
    quality: int = KEYFRAME_IMAGE_QUALITY,
) -> dict:
    """
    按固定时间间隔提取帧

    Args:
        video_path: 输入视频路径
        interval: 提取间隔 (秒), 如 1.0 表示每秒一帧
        output_dir: 输出目录
        image_format: 图片格式
        quality: 图片质量

    Returns:
        dict: {"output_dir": str, "count": int}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if output_dir is None:
        output_dir = OUTPUT_EXTRACT / f"{video_path.stem}_frames_{interval}s"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / f"frame_%04d.{image_format}")

    # fps=1/interval, 例如 interval=2 → fps=0.5 (每 2 秒一帧)
    fps_filter = f"fps=1/{interval}"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", fps_filter,
    ]

    if image_format == "jpg":
        cmd += ["-qscale:v", str(max(1, min(31, 32 - int(quality * 31 / 100))))]

    cmd += [output_pattern]

    logger.info(f"按间隔提取帧: {video_path.name} (每 {interval}s)")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    frames = list(output_dir.glob(f"frame_*.{image_format}"))
    logger.info(f"✅ 提取了 {len(frames)} 帧 → {output_dir}")

    return {"output_dir": str(output_dir), "count": len(frames)}


def extract_frames_scene_change(
    video_path: str | Path,
    threshold: float = 0.3,
    output_dir: str | Path | None = None,
    image_format: str = KEYFRAME_IMAGE_FORMAT,
    quality: int = KEYFRAME_IMAGE_QUALITY,
) -> dict:
    """
    按场景切换提取帧 (检测画面变化)

    Args:
        video_path: 输入视频路径
        threshold: 场景变化阈值 (0-1, 越小越敏感, 推荐 0.3-0.5)
        output_dir: 输出目录
        image_format: 图片格式
        quality: 图片质量

    Returns:
        dict: {"output_dir": str, "count": int}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if output_dir is None:
        output_dir = OUTPUT_EXTRACT / f"{video_path.stem}_scenes"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_pattern = str(output_dir / f"scene_%04d.{image_format}")

    # 使用 scene 检测滤镜
    scene_filter = f"select=gt(scene\\,{threshold})"

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", scene_filter,
        "-vsync", "vfr",
    ]

    if image_format == "jpg":
        cmd += ["-qscale:v", str(max(1, min(31, 32 - int(quality * 31 / 100))))]

    cmd += [output_pattern]

    logger.info(f"按场景变化提取帧: {video_path.name} (阈值={threshold})")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    frames = list(output_dir.glob(f"scene_*.{image_format}"))
    logger.info(f"✅ 检测到 {len(frames)} 个场景帧 → {output_dir}")

    return {"output_dir": str(output_dir), "count": len(frames)}


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="关键帧/帧提取")
    parser.add_argument("input", help="输入视频路径")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--keyframes", action="store_true", help="提取 I-帧 (关键帧)")
    mode_group.add_argument("--interval", type=float, help="按间隔提取 (秒)")
    mode_group.add_argument("--scene", type=float, nargs="?", const=0.3, help="按场景切换提取 (阈值, 默认 0.3)")

    parser.add_argument("-o", "--output-dir", help="输出目录")
    parser.add_argument("--format", default=KEYFRAME_IMAGE_FORMAT, help=f"图片格式 (默认: {KEYFRAME_IMAGE_FORMAT})")
    parser.add_argument("--quality", type=int, default=KEYFRAME_IMAGE_QUALITY, help=f"图片质量 (默认: {KEYFRAME_IMAGE_QUALITY})")

    args = parser.parse_args()

    if args.keyframes:
        result = extract_keyframes(args.input, args.output_dir, args.format, args.quality)
    elif args.interval:
        result = extract_frames_interval(args.input, args.interval, args.output_dir, args.format, args.quality)
    elif args.scene is not None:
        result = extract_frames_scene_change(args.input, args.scene, args.output_dir, args.format, args.quality)

    print(f"\n✅ 完成: 提取了 {result['count']} 个文件 → {result['output_dir']}")
