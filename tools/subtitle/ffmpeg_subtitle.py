"""
FFmpeg 字幕烧录模块

功能:
  - 将 SRT 字幕文件烧录进视频 (硬字幕)
  - 可自定义字体、大小、颜色、位置、描边等样式
  - 基于 FFmpeg subtitles/ass 滤镜
"""
import subprocess
from pathlib import Path

from config import FFMPEG_BIN, OUTPUT_SUBTITLE
from tools.common import get_video_info, logger, generate_output_name

# 字幕样式预设 (ASS 格式 Style 行)
SUBTITLE_STYLES = {
    "default": {
        "name": "📝 默认 (白字黑边)",
        "ass_style": "Style: Default,Arial,22,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1",
    },
    "large": {
        "name": "🔤 大字 (醒目)",
        "ass_style": "Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,3,2,2,10,10,25,1",
    },
    "cinema": {
        "name": "🎬 影院风 (黄字)",
        "ass_style": "Style: Default,Arial,24,&H0000FFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1",
    },
    "minimal": {
        "name": "✨ 简约 (小字无阴影)",
        "ass_style": "Style: Default,Arial,18,&H00FFFFFF,&H000000FF,&H00333333,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,10,10,20,1",
    },
}


def _srt_to_ass(srt_path: Path, ass_path: Path, style_line: str):
    """将 SRT 转为 ASS 格式 (内嵌样式)"""
    import re

    # ASS 文件头
    header = f"""[Script Info]
Title: x-tools subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
{style_line}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # 解析 SRT
    srt_text = srt_path.read_text(encoding="utf-8")
    blocks = re.split(r"\n\n+", srt_text.strip())

    events = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # 解析时间: 00:00:01,500 --> 00:00:04,000
        time_match = re.search(r"(\d{2}:\d{2}:\d{2}),(\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}),(\d{3})", lines[1])
        if not time_match:
            continue

        start = f"{time_match.group(1)}.{time_match.group(2)[:2]}"
        end = f"{time_match.group(3)}.{time_match.group(4)[:2]}"
        text = "\\N".join(lines[2:])  # ASS 换行用 \N

        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    ass_content = header + "\n".join(events) + "\n"
    ass_path.write_text(ass_content, encoding="utf-8")


def burn_subtitles(
    video_path: str | Path,
    subtitle_path: str | Path,
    output_path: str | Path | None = None,
    style: str = "default",
    crf: int = 18,
) -> dict:
    """
    将字幕烧录进视频

    Args:
        video_path: 视频文件路径
        subtitle_path: SRT 字幕文件路径
        output_path: 输出路径 (默认自动生成)
        style: 字幕样式预设名称
        crf: 视频质量

    Returns:
        dict: {"output": str, "size_mb": float}
    """
    video_path = Path(video_path)
    subtitle_path = Path(subtitle_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not subtitle_path.is_file():
        raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")

    # 输出路径
    OUTPUT_SUBTITLE.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_name = generate_output_name(video_path.stem, ".mp4", tag="sub")
        output_path = OUTPUT_SUBTITLE / output_name
    output_path = Path(output_path)

    # 获取样式
    style_line = SUBTITLE_STYLES.get(style, SUBTITLE_STYLES["default"])["ass_style"]

    # SRT → ASS (内嵌样式), 放到临时目录
    import shutil
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    tmp_ass = Path(tmp_dir) / "sub.ass"
    _srt_to_ass(subtitle_path, tmp_ass, style_line)

    # 必须转义: \ 和 : (尤其是 Windows 下的盘符和路径, Mac/Linux 偶尔也会被误判)
    # 对于 subprocess + FFmpeg filter string，最佳方案是直接用绝对路径转义后不带引号
    ass_escaped = str(tmp_ass.resolve()).replace("\\", "/").replace(":", "\\:")
    vf = f"ass={ass_escaped}"

    # 获取原视频码率
    orig_info = get_video_info(str(video_path))
    orig_bitrate = orig_info.get("bitrate", 0)
    if orig_bitrate > 0:
        encode_opts = ["-b:v", str(orig_bitrate)]
    else:
        encode_opts = ["-crf", str(crf)]

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", *encode_opts, "-preset", "fast",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(f"烧录字幕: {video_path.name}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not output_path.is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ 字幕烧录完成: {output_path.name} ({size_mb:.1f} MB)")

    return {
        "output": str(output_path),
        "size_mb": round(size_mb, 2),
    }
