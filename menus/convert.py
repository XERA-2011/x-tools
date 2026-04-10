"""格式转换菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from menus._prompts import confirm_action, maybe_show_file_list


def menu_convert(media: list[Path]):
    """格式转换菜单"""
    from tools.convert.batch import batch_convert
    from tools.convert.ffmpeg_convert import AUDIO_FORMATS, VIDEO_FORMATS

    mode = inquirer.select(
        message="选择转换模式:",
        choices=[
            Choice("transcode", "🎬 视频格式转换 (MKV/MOV/AVI ↔ MP4 等)"),
            Choice("audio", "🎵 提取音频 (视频 → MP3/AAC/WAV/FLAC)"),
            Choice("strip", "🔇 去除音频 (仅保留视频流)"),
            Choice("remux", "⚡ 快速封装 (无损换容器, 极快)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if mode == "back":
        return

    video_fmts = sorted(VIDEO_FORMATS.keys())
    audio_fmts = sorted(AUDIO_FORMATS.keys())

    if mode == "audio":
        target_format = inquirer.select(
            message="目标音频格式:",
            choices=[Choice(f, f".{f}") for f in audio_fmts],
            default="mp3",
        ).execute()
    elif mode in ("transcode", "strip", "remux"):
        target_format = inquirer.select(
            message="目标视频格式:",
            choices=[Choice(f, f".{f}") for f in video_fmts],
            default="mp4",
        ).execute()

    # 转码模式可选编码器
    video_codec = None
    if mode == "transcode":
        codec_choice = inquirer.select(
            message="视频编码器:",
            choices=[
                Choice(None, "🔧 默认 (根据格式自动选择)"),
                Choice("libx264", "H.264 (兼容性最好)"),
                Choice("libx265", "H.265/HEVC (更好压缩, 部分平台不支持)"),
            ],
            default=None,
        ).execute()
        video_codec = codec_choice

    maybe_show_file_list(media)

    if confirm_action(f"确认转换 {len(media)} 个文件 → .{target_format}?"):
        batch_convert(
            files=media,
            target_format=target_format,
            video_codec=video_codec,
            copy_streams=(mode == "remux"),
            strip_audio=(mode == "strip"),
        )
