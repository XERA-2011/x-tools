"""
批量格式转换入口

支持:
  - 批量视频格式转换 (MKV/MOV/AVI → MP4 等)
  - 批量提取音频 (视频 → MP3/AAC/WAV/FLAC)
  - 批量去除音频
  - 批量快速封装 (-c copy)
"""
from pathlib import Path

from config import INPUT_DIR, ensure_dirs
from tools.common import scan_videos, scan_media, batch_process, print_summary, logger
from tools.convert.ffmpeg_convert import convert_media, ALL_FORMATS


def batch_convert(
    input_dir: str | Path | None = None,
    files: list[Path] | None = None,
    target_format: str = "mp4",
    video_codec: str | None = None,
    audio_codec: str | None = None,
    crf: int = 18,
    audio_bitrate: str = "192k",
    copy_streams: bool = False,
    strip_audio: bool = False,
    **kwargs,
) -> list[dict]:
    """
    批量格式转换

    Args:
        input_dir: 输入目录
        files: 文件列表 (优先于 input_dir)
        target_format: 目标格式
        video_codec: 视频编码器
        audio_codec: 音频编码器
        crf: 视频质量
        audio_bitrate: 音频码率
        copy_streams: 快速封装模式
        strip_audio: 去除音频模式
    """
    if files is None:
        videos, images = scan_media(input_dir or INPUT_DIR)
        files = videos + images

    if copy_streams:
        desc = f"批量快速封装 (→ .{target_format})"
    elif strip_audio:
        desc = f"批量去除音频 (→ .{target_format})"
    else:
        desc = f"批量格式转换 (→ .{target_format})"

    results = batch_process(
        files,
        convert_media,
        desc=desc,
        target_format=target_format,
        video_codec=video_codec,
        audio_codec=audio_codec,
        crf=crf,
        audio_bitrate=audio_bitrate,
        copy_streams=copy_streams,
        strip_audio=strip_audio,
        **kwargs,
    )
    print_summary(results)
    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量格式转换")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入目录")
    parser.add_argument(
        "-f", "--format", dest="target_format", default="mp4",
        help=f"目标格式 (支持: {', '.join(sorted(ALL_FORMATS))})",
    )
    parser.add_argument("--video-codec", help="视频编码器")
    parser.add_argument("--audio-codec", help="音频编码器")
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF")
    parser.add_argument("--audio-bitrate", default="192k", help="音频码率")
    parser.add_argument("--copy", action="store_true", help="快速封装模式")
    parser.add_argument("--strip-audio", action="store_true", help="去除音频")

    args = parser.parse_args()
    ensure_dirs()

    batch_convert(
        input_dir=args.input_dir,
        target_format=args.target_format,
        video_codec=args.video_codec,
        audio_codec=args.audio_codec,
        crf=args.crf,
        audio_bitrate=args.audio_bitrate,
        copy_streams=args.copy,
        strip_audio=args.strip_audio,
    )
