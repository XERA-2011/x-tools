"""
媒体信息查看模块

功能:
  - 使用 ffprobe 获取详细的媒体文件信息
  - Rich 表格展示: 分辨率、帧率、编码器、码率、音频信息等
  - 批量查看多个文件的信息摘要

使用方式:
  python tools/mediainfo/probe.py video.mp4
  python tools/mediainfo/probe.py video1.mp4 video2.mkv image.png
"""
import json
import subprocess
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import FFPROBE_BIN
from tools.common import logger


console = Console()


# ============================================================
# 清晰度等级判定
# ============================================================
RESOLUTION_TIERS = [
    (7680, 4320, "8K Ultra HD"),
    (3840, 2160, "4K Ultra HD"),
    (2560, 1440, "2K QHD"),
    (1920, 1080, "1080p Full HD"),
    (1280,  720, "720p HD"),
    ( 854,  480, "480p SD"),
    ( 640,  360, "360p"),
    (   0,    0, "低于 360p"),
]


def classify_resolution(width: int, height: int) -> str:
    """根据分辨率判断清晰度等级"""
    pixels = max(width, height)  # 取长边, 兼容竖屏
    for tier_w, tier_h, label in RESOLUTION_TIERS:
        if pixels >= max(tier_w, tier_h):
            return label
    return "未知"


def classify_fps(fps: float) -> str:
    """帧率评价"""
    if fps >= 120:
        return "超高帧率"
    elif fps >= 60:
        return "高帧率"
    elif fps >= 29:
        return "标准"
    elif fps >= 23:
        return "电影帧率"
    elif fps > 0:
        return "低帧率"
    return "未知"


def format_duration(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS"""
    if seconds <= 0:
        return "未知"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_bitrate(bps: int) -> str:
    """码率格式化 (bps → Kbps/Mbps)"""
    if bps <= 0:
        return "未知"
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.1f} Mbps"
    return f"{bps / 1_000:.0f} Kbps"


def format_filesize(size_bytes: int) -> str:
    """文件大小格式化"""
    if size_bytes <= 0:
        return "未知"
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.2f} GB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    return f"{size_bytes / 1024:.0f} KB"


# ============================================================
# 详细媒体信息获取
# ============================================================
def get_detailed_info(file_path: str | Path) -> dict:
    """
    使用 ffprobe 获取详细的媒体文件信息

    Returns:
        dict: 包含 video/audio/format 子字典的完整信息
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    cmd = [
        FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(file_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"无法读取文件信息: {file_path} — {e}")
        return {}

    streams = data.get("streams", [])
    fmt = data.get("format", {})

    # 视频流
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    # 音频流
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    info = {
        "filename": file_path.name,
        "filepath": str(file_path),
        "filesize": file_path.stat().st_size,
        "container": fmt.get("format_long_name", fmt.get("format_name", "未知")),
        "duration": float(fmt.get("duration", 0)),
        "bitrate": int(fmt.get("bit_rate", 0)),
        "streams_count": len(streams),
    }

    # 视频信息
    if video_stream:
        fps_str = video_stream.get("r_frame_rate", "0/1")
        try:
            num, den = map(int, fps_str.split("/"))
            fps = num / den if den else 0
        except (ValueError, ZeroDivisionError):
            fps = 0

        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))

        info["video"] = {
            "codec": video_stream.get("codec_name", "未知"),
            "codec_long": video_stream.get("codec_long_name", ""),
            "profile": video_stream.get("profile", ""),
            "width": width,
            "height": height,
            "resolution_tier": classify_resolution(width, height),
            "fps": round(fps, 2),
            "fps_tier": classify_fps(fps),
            "bitrate": int(video_stream.get("bit_rate", 0)),
            "pix_fmt": video_stream.get("pix_fmt", "未知"),
            "frames": int(video_stream.get("nb_frames", 0)),
        }

    # 音频信息
    if audio_stream:
        info["audio"] = {
            "codec": audio_stream.get("codec_name", "未知"),
            "codec_long": audio_stream.get("codec_long_name", ""),
            "sample_rate": int(audio_stream.get("sample_rate", 0)),
            "channels": int(audio_stream.get("channels", 0)),
            "channel_layout": audio_stream.get("channel_layout", ""),
            "bitrate": int(audio_stream.get("bit_rate", 0)),
        }

    return info


# ============================================================
# Rich 表格展示
# ============================================================
def display_info(info: dict):
    """使用 Rich 表格展示媒体信息"""
    if not info:
        console.print("[red]❌ 无法读取文件信息[/red]")
        return

    # 基础信息表
    table = Table(
        title=f"📋 {info['filename']}",
        show_header=False,
        border_style="cyan",
        title_style="bold cyan",
        padding=(0, 2),
    )
    table.add_column("属性", style="bold", width=14)
    table.add_column("值", min_width=30)

    table.add_row("📁 文件大小", format_filesize(info["filesize"]))
    table.add_row("📦 容器格式", info["container"])
    table.add_row("⏱️  时长", format_duration(info["duration"]))
    table.add_row("📊 总码率", format_bitrate(info["bitrate"]))
    table.add_row("🔢 流数量", str(info["streams_count"]))

    # 视频流信息
    video = info.get("video")
    if video:
        table.add_section()
        res = f"{video['width']}×{video['height']}"
        tier = video["resolution_tier"]
        table.add_row("🎬 分辨率", f"{res}  ({tier})")
        table.add_row("🎞️  帧率", f"{video['fps']} FPS  ({video['fps_tier']})")
        codec_desc = video["codec"].upper()
        if video["profile"]:
            codec_desc += f" ({video['profile']})"
        table.add_row("🔧 视频编码", codec_desc)
        if video["bitrate"]:
            table.add_row("📊 视频码率", format_bitrate(video["bitrate"]))
        table.add_row("🎨 像素格式", video["pix_fmt"])
        if video["frames"]:
            table.add_row("🖼️  总帧数", f"{video['frames']:,}")

    # 音频流信息
    audio = info.get("audio")
    if audio:
        table.add_section()
        table.add_row("🔊 音频编码", audio["codec"].upper())
        if audio["sample_rate"]:
            table.add_row("🎵 采样率", f"{audio['sample_rate']:,} Hz")
        channels = audio["channels"]
        layout = audio.get("channel_layout", "")
        ch_desc = f"{channels} 声道"
        if layout:
            ch_desc += f" ({layout})"
        table.add_row("📢 声道", ch_desc)
        if audio["bitrate"]:
            table.add_row("📊 音频码率", format_bitrate(audio["bitrate"]))
    elif video:
        table.add_section()
        table.add_row("🔇 音频", "无音频流")

    console.print()
    console.print(table)
    console.print()


def display_batch_summary(files: list[Path]):
    """批量展示多个文件的摘要表格"""
    table = Table(
        title="📋 媒体文件信息汇总",
        border_style="cyan",
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("文件名", style="bold", max_width=30)
    table.add_column("大小", width=10)
    table.add_column("分辨率", width=14)
    table.add_column("清晰度", width=14)
    table.add_column("帧率", width=10)
    table.add_column("编码", width=10)
    table.add_column("时长", width=8)
    table.add_column("音频", width=8)

    for i, f in enumerate(files, 1):
        info = get_detailed_info(f)
        if not info:
            table.add_row(str(i), f.name, "❌", "—", "—", "—", "—", "—", "—")
            continue

        video = info.get("video", {})
        audio = info.get("audio")

        res = f"{video.get('width', '?')}×{video.get('height', '?')}" if video else "图片"
        tier = video.get("resolution_tier", "—") if video else "—"
        fps = f"{video.get('fps', 0)} FPS" if video else "—"
        codec = video.get("codec", "—").upper() if video else "—"
        duration = format_duration(info["duration"])
        audio_str = audio["codec"].upper() if audio else "无"

        table.add_row(
            str(i), f.name, format_filesize(info["filesize"]),
            res, tier, fps, codec, duration, audio_str,
        )

    console.print()
    console.print(table)
    console.print()


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="查看媒体文件详细信息")
    parser.add_argument("files", nargs="+", help="输入文件路径 (支持多个)")
    parser.add_argument("--summary", action="store_true", help="摘要模式 (多文件对比)")

    args = parser.parse_args()
    paths = [Path(f) for f in args.files]

    if args.summary or len(paths) > 1:
        display_batch_summary(paths)
    else:
        info = get_detailed_info(paths[0])
        display_info(info)
