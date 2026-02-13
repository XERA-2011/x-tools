#!/usr/bin/env python3
"""
x-tools 演示脚本

用法:
    python scripts/run_demo.py <视频文件路径>

将对指定视频演示所有可用的 extract 功能
"""
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config import ensure_dirs
from tools.common import get_video_info, logger


def demo_extract(video_path: str):
    """演示内容截取功能"""
    from tools.extract.clip_extractor import extract_clip
    from tools.extract.keyframe_extractor import (
        extract_keyframes,
        extract_frames_interval,
    )

    video = Path(video_path)
    if not video.is_file():
        print(f"❌ 文件不存在: {video_path}")
        sys.exit(1)

    ensure_dirs()

    # 1. 显示视频信息
    print("\n" + "=" * 60)
    print("📋 视频信息")
    print("=" * 60)
    info = get_video_info(video)
    if info:
        print(f"  时长:     {info['duration']:.1f} 秒")
        print(f"  分辨率:   {info['width']}x{info['height']}")
        print(f"  帧率:     {info['fps']} fps")
        print(f"  编码:     {info['codec']}")
        print(f"  码率:     {info['bitrate'] / 1000:.0f} kbps")
    else:
        print("  ⚠️ 无法读取视频信息 (请确认 ffprobe 已安装)")
        return

    # 2. 截取前 5 秒
    print("\n" + "=" * 60)
    print("✂️  测试: 截取前 5 秒")
    print("=" * 60)
    try:
        clip_duration = min(5, info["duration"])
        result = extract_clip(video, start="0", duration=str(clip_duration))
        print(f"  → {result['output']}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    # 3. 提取关键帧
    print("\n" + "=" * 60)
    print("🖼️  测试: 提取关键帧 (I-帧)")
    print("=" * 60)
    try:
        result = extract_keyframes(video)
        print(f"  → {result['count']} 个关键帧 → {result['output_dir']}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    # 4. 按间隔提取
    print("\n" + "=" * 60)
    print("⏱️  测试: 每 2 秒提取一帧")
    print("=" * 60)
    try:
        result = extract_frames_interval(video, interval=2.0)
        print(f"  → {result['count']} 帧 → {result['output_dir']}")
    except Exception as e:
        print(f"  ❌ 失败: {e}")

    print("\n" + "=" * 60)
    print("🎉 演示完成! 输出文件在 output/extract/ 目录下")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python scripts/run_demo.py <视频文件路径>")
        print("示例: python scripts/run_demo.py input/test.mp4")
        sys.exit(1)

    demo_extract(sys.argv[1])
