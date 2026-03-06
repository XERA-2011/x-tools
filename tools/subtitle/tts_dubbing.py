"""
基于 SRT 字幕的 AI 视频配音模块 (Text-to-Speech)

功能:
  - 解析 SRT 字幕的时间轴和文字
  - 使用 edge-tts 将字幕转化为自然的 AI 语音
  - 使用 FFmpeg 将语音片段拼接并混入原视频
"""
import asyncio
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from config import FFMPEG_BIN, OUTPUT_SUBTITLE
from tools.common import logger, generate_output_name, get_video_info

# Edge-TTS 音色预设 (部分中英文高质量音色)
TTS_VOICES = {
    "xiaoxiao": {"name": "👧 晓晓 (温柔女声)", "voice": "zh-CN-XiaoxiaoNeural"},
    "yunxi": {"name": "👦 云希 (活力男声)", "voice": "zh-CN-YunxiNeural"},
    "yunjian": {"name": "👨 云健 (影视男声)", "voice": "zh-CN-YunjianNeural"},
    "xiaoyi": {"name": "👩 晓伊 (卡通女声)", "voice": "zh-CN-XiaoyiNeural"},
    "yunxia": {"name": "👦 云夏 (正太男声)", "voice": "zh-CN-YunxiaNeural"},
}


def _parse_srt_time(time_str: str) -> float:
    """将 SRT 时间 00:00:01,500 解析为秒数 (1.5)"""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", time_str.strip())
    if not match:
        return 0.0
    h, m, s, ms = map(int, match.groups())
    return h * 3600 + m * 60 + s + ms / 1000.0


def _parse_srt_file(srt_path: Path) -> list:
    """解析 SRT 文件，返回字幕列表 [{"start": 1.5, "end": 4.0, "text": "你好"}]"""
    srt_text = srt_path.read_text(encoding="utf-8").strip()
    blocks = re.split(r"\n\n+", srt_text)

    subtitles = []
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        time_match = re.search(r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})", lines[1])
        if not time_match:
            continue

        start = _parse_srt_time(time_match.group(1))
        end = _parse_srt_time(time_match.group(2))
        text = " ".join(lines[2:]).strip()

        if text:
            subtitles.append({
                "start": start,
                "end": end,
                "text": text
            })
    return subtitles


async def _generate_audio_segment(text: str, voice: str, output_path: Path):
    """使用 edge-tts 生成单句音频"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


async def _process_all_subtitles(subtitles: list, voice: str, tmp_dir: Path) -> list:
    """批量生成所有字幕的音频文件，并返回对应的文件路径和时间点"""
    tasks = []
    audio_segments = []

    for i, sub in enumerate(subtitles):
        audio_path = tmp_dir / f"seg_{i:04d}.mp3"
        audio_segments.append({
            "idx": i,
            "start": sub["start"],
            "path": audio_path,
            "text": sub["text"]
        })
        tasks.append(_generate_audio_segment(sub["text"], voice, audio_path))

    # 并发生成音频
    await asyncio.gather(*tasks)
    return audio_segments


def _merge_audio_to_video(video_path: Path, audio_segments: list, output_path: Path):
    """使用 FFmpeg 的 adelay 将所有生成的单句音频对齐并混音，然后与原视频合并"""
    if not audio_segments:
        # 如果没有音频，直接复制视频
        shutil.copy(video_path, output_path)
        return

    # 构建复杂的 FFmpeg filter_complex 字符串
    inputs = ["-i", str(video_path)]
    filter_chains = []

    # 遍历每个生成的语音片段
    # audio_segments 形如: [{"path": Path("..."), "start": 1.5}]
    for i, seg in enumerate(audio_segments):
        inputs.extend(["-i", str(seg["path"])])
        
        # 将开始时间(秒)转换为毫秒，用于 adelay
        delay_ms = int(seg["start"] * 1000)
        # 滤镜示例: [1:a]adelay=1500|1500[a1];
        # ffmpeg 中第一路是视频输入(idx=0)，生成的音频从 idx=1 开始
        filter_chains.append(f"[{i+1}:a]adelay={delay_ms}|{delay_ms}[a{i+1}]")

    # 混合所有的语音(和原音)
    # 检查原视频是否有音频
    orig_info = get_video_info(str(video_path))
    has_audio = orig_info.get("has_audio", False)
    
    amix_inputs = ""
    amix_count = len(audio_segments)
    
    if has_audio:
        # 如果有原音，直接使用原音的音频轨道进行混合
        amix_inputs += "[0:a]"
        amix_count += 1

    for i in range(len(audio_segments)):
        amix_inputs += f"[a{i+1}]"

    # amix=inputs=N:duration=longest:dropout_transition=0:normalize=0
    # 彻底关闭 amix 自带的动态音量标准化 (normalize=0)，防止随着片段播放完毕，存留音轨的音量突然被成倍放大
    filter_chains.append(f"{amix_inputs}amix=inputs={amix_count}:duration=longest:dropout_transition=0:normalize=0[outa]")

    filter_complex = ";".join(filter_chains)

    cmd = [
        FFMPEG_BIN, "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "0:v",          # 原视频轨
        "-map", "[outa]",       # 合并后的新音频轨
        "-c:v", "copy",         # 视频直接复制，极快
        "-c:a", "aac", "-b:a", "192k", 
        str(output_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 配音混音失败:\n{result.stderr[-500:]}")


def dub_video_with_tts(
    video_path: str | Path,
    srt_path: str | Path,
    output_path: str | Path | None = None,
    voice_key: str = "xiaoxiao",
) -> dict:
    """
    给视频添加基于字幕生成的 AI 配音
    """
    video_path = Path(video_path)
    srt_path = Path(srt_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not srt_path.is_file():
        raise FileNotFoundError(f"字幕文件不存在: {srt_path}")

    # 输出路径
    OUTPUT_SUBTITLE.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_name = generate_output_name(video_path.stem, ".mp4", tag="tts")
        output_path = OUTPUT_SUBTITLE / output_name
    output_path = Path(output_path)

    voice = TTS_VOICES.get(voice_key, TTS_VOICES["xiaoxiao"])["voice"]

    # 1. 解析 SRT
    subtitles = _parse_srt_file(srt_path)
    if not subtitles:
        logger.warning("字幕文件为空或无法解析。")
        shutil.copy(video_path, output_path)
        return {"output": str(output_path), "size_mb": 0}

    logger.info(f"解析到 {len(subtitles)} 条字幕，开始生成 AI 配音 ({voice_key})...")
    
    tmp_dir = Path(tempfile.mkdtemp(prefix="x_tools_tts_"))
    try:
        # 2. 批量生成音频片段
        audio_segments = asyncio.run(_process_all_subtitles(subtitles, voice, tmp_dir))
        
        # 过滤掉生成失败的
        audio_segments = [seg for seg in audio_segments if seg["path"].exists() and seg["path"].stat().st_size > 0]
        
        logger.info("音频生成完毕，开始混入视频...")
        
        # 3. 如果原视频没音频，直接混音会失败？get_video_info 已经在内部处理
        _merge_audio_to_video(video_path, audio_segments, output_path)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if not output_path.is_file():
        raise RuntimeError("配音合成失败，未生成输出文件。")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ 配音视频合成完成: {output_path.name} ({size_mb:.1f} MB)")

    return {
        "output": str(output_path),
        "size_mb": round(size_mb, 2)
    }
