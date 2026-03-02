"""
FFmpeg 格式转换模块

功能:
  - 视频格式转换 (MKV/MOV/AVI → MP4 等)
  - 提取音频 (视频 → MP3/AAC/WAV/FLAC)
  - 去除音频 (仅保留视频流)
  - 快速封装 (-c copy, 无损换容器)

使用方式:
  python tools/convert/ffmpeg_convert.py video.mp4 -f mkv
  python tools/convert/ffmpeg_convert.py video.mp4 -f mp3 --audio-only
  python tools/convert/ffmpeg_convert.py video.mkv -f mp4 --copy
"""
import subprocess
from pathlib import Path

from config import FFMPEG_BIN, OUTPUT_CONVERT
from tools.common import get_video_info, generate_output_name, logger


# 支持的输出格式及其默认编码器
VIDEO_FORMATS = {
    "mp4":  {"vcodec": "libx264", "acodec": "aac"},
    "mkv":  {"vcodec": "libx264", "acodec": "aac"},
    "avi":  {"vcodec": "libx264", "acodec": "mp3"},
    "mov":  {"vcodec": "libx264", "acodec": "aac"},
    "webm": {"vcodec": "libvpx-vp9", "acodec": "libopus"},
    "ts":   {"vcodec": "libx264", "acodec": "aac"},
}

AUDIO_FORMATS = {
    "mp3":  {"acodec": "libmp3lame", "ext": "mp3"},
    "aac":  {"acodec": "aac", "ext": "m4a"},
    "wav":  {"acodec": "pcm_s16le", "ext": "wav"},
    "flac": {"acodec": "flac", "ext": "flac"},
}

ALL_FORMATS = set(VIDEO_FORMATS) | set(AUDIO_FORMATS)


def convert_media(
    input_path: str | Path,
    output_path: str | Path | None = None,
    target_format: str = "mp4",
    video_codec: str | None = None,
    audio_codec: str | None = None,
    crf: int = 18,
    audio_bitrate: str = "192k",
    copy_streams: bool = False,
    strip_audio: bool = False,
) -> dict:
    """
    使用 FFmpeg 转换媒体格式

    Args:
        input_path: 输入文件路径
        output_path: 输出路径 (默认自动生成)
        target_format: 目标格式 (mp4/mkv/avi/mov/webm/mp3/aac/wav/flac)
        video_codec: 视频编码器 (None = 使用格式默认)
        audio_codec: 音频编码器 (None = 使用格式默认)
        crf: 视频质量 (0-51, 越小越好, 推荐 18)
        audio_bitrate: 音频码率
        copy_streams: True = 无损转封装 (-c copy)
        strip_audio: True = 去除音频轨道

    Returns:
        dict: {"output": str, "input_format": str, "output_format": str, ...}
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    target_format = target_format.lower().lstrip(".")
    if target_format not in ALL_FORMATS:
        raise ValueError(f"不支持的格式: {target_format} (支持: {', '.join(sorted(ALL_FORMATS))})")

    is_audio_output = target_format in AUDIO_FORMATS

    # 获取输入文件信息
    info = get_video_info(input_path)
    input_format = input_path.suffix.lstrip(".")

    # 确定输出后缀
    if is_audio_output:
        out_ext = f".{AUDIO_FORMATS[target_format]['ext']}"
    else:
        out_ext = f".{target_format}"

    # 输出路径
    OUTPUT_CONVERT.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_name = generate_output_name(input_path.stem, out_ext, tag="convert")
        output_path = OUTPUT_CONVERT / output_name
    output_path = Path(output_path)

    # 构建 FFmpeg 命令
    cmd = [FFMPEG_BIN, "-y", "-i", str(input_path)]

    if copy_streams and not is_audio_output:
        # 快速封装: 不重新编码
        cmd += ["-c", "copy"]
        mode_desc = "快速封装"
    elif is_audio_output:
        # 提取音频
        acodec = audio_codec or AUDIO_FORMATS[target_format]["acodec"]
        cmd += ["-vn", "-c:a", acodec]
        if acodec not in ("pcm_s16le", "flac"):  # 无损格式不需要码率
            cmd += ["-b:a", audio_bitrate]
        mode_desc = "提取音频"
    elif strip_audio:
        # 去除音频
        vcodec = video_codec or VIDEO_FORMATS.get(target_format, {}).get("vcodec", "libx264")
        cmd += ["-an", "-c:v", vcodec, "-crf", str(crf), "-preset", "medium"]
        mode_desc = "去除音频"
    else:
        # 视频格式转换
        fmt_defaults = VIDEO_FORMATS.get(target_format, {"vcodec": "libx264", "acodec": "aac"})
        vcodec = video_codec or fmt_defaults["vcodec"]
        acodec = audio_codec or fmt_defaults["acodec"]
        cmd += [
            "-c:v", vcodec, "-crf", str(crf), "-preset", "medium",
            "-c:a", acodec, "-b:a", audio_bitrate,
        ]
        mode_desc = "格式转换"

    # MP4 优化: moov atom 前置
    if target_format == "mp4":
        cmd += ["-movflags", "+faststart"]

    cmd.append(str(output_path))

    # 日志
    size_info = f"{info.get('width', '?')}x{info.get('height', '?')}" if info else "未知"
    logger.info(f"{mode_desc}: {input_path.name} ({input_format} → {target_format}, {size_info})")

    # 执行
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    # 确认输出
    if not output_path.is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ {mode_desc}完成: {output_path.name} ({output_size_mb:.1f} MB)")

    return {
        "output": str(output_path),
        "input_format": input_format,
        "output_format": target_format,
        "mode": mode_desc,
        "size_mb": round(output_size_mb, 2),
    }


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FFmpeg 格式转换")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument(
        "-f", "--format", dest="target_format", default="mp4",
        help=f"目标格式 (默认: mp4, 支持: {', '.join(sorted(ALL_FORMATS))})",
    )
    parser.add_argument("--video-codec", help="视频编码器 (如 libx264, libx265)")
    parser.add_argument("--audio-codec", help="音频编码器 (如 aac, libmp3lame)")
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF (默认: 18)")
    parser.add_argument("--audio-bitrate", default="192k", help="音频码率 (默认: 192k)")
    parser.add_argument("--copy", action="store_true", help="快速封装 (不重新编码)")
    parser.add_argument("--audio-only", action="store_true", help="仅提取音频")
    parser.add_argument("--strip-audio", action="store_true", help="去除音频轨道")

    args = parser.parse_args()

    # 自动推断: 如果目标格式是音频格式, 自动切到 audio-only 逻辑
    result = convert_media(
        input_path=args.input,
        output_path=args.output,
        target_format=args.target_format,
        video_codec=args.video_codec,
        audio_codec=args.audio_codec,
        crf=args.crf,
        audio_bitrate=args.audio_bitrate,
        copy_streams=args.copy,
        strip_audio=args.strip_audio,
    )
    print(f"\n✅ {result['mode']}完成: {result['input_format']} → {result['output_format']}")
    print(f"   输出: {result['output']} ({result['size_mb']} MB)")
