"""
为单个视频添加背景音乐

使用方式:
  python tools/bgm/ffmpeg_bgm.py video.mp4 -m music/bgm.mp3
"""
from pathlib import Path

import subprocess

from config import FFMPEG_BIN, OUTPUT_BGM
from tools.common import generate_output_name, get_video_info, logger
from tools.ffmpeg.ffmpeg_core import (
    build_base_filters,
    build_concat_graph,
    compute_total_duration,
)


def add_bgm_to_video(
    video_path: str | Path,
    music_path: str | Path,
    output_path: str | Path | None = None,
    music_volume: float = 0.3,
    keep_original_audio: bool = False,
    crf: int = 18,
    trim_start: float = 0.0,
    trim_end: float = 0.0,
    audio_fade_in: float = 0.0,
    audio_fade_out: float = 0.0,
) -> dict:
    """
    为单个视频添加背景音乐

    Args:
        video_path: 视频文件路径
        music_path: 背景音乐路径
        output_path: 输出路径 (默认自动生成)
        music_volume: 背景音乐音量 (0.0~1.0, 默认 0.3)
        keep_original_audio: 是否保留原视频声音 (True=混合, False=替换)
        crf: 视频质量
        trim_start: 裁剪开头秒数 (默认 0.0)
        trim_end: 裁剪结尾秒数 (默认 0.0)
        audio_fade_in: 音频淡入时长 (默认 0.0)
        audio_fade_out: 音频淡出时长 (默认 0.0)

    Returns:
        dict: {"output": str, "duration": float, ...}
    """
    if output_path is None:
        OUTPUT_BGM.mkdir(parents=True, exist_ok=True)
        output_name = generate_output_name("bgm", ".mp4", tag="added")
        output_path = OUTPUT_BGM / output_name

    video_path = Path(video_path)
    music_path = Path(music_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not music_path.is_file():
        raise FileNotFoundError(f"音乐文件不存在: {music_path}")

    # 以视频分辨率为基准
    first_info = get_video_info(str(video_path))
    target_w = first_info.get("width", 1920)
    target_h = first_info.get("height", 1080)
    target_w = target_w // 2 * 2
    target_h = target_h // 2 * 2

    include_audio = keep_original_audio
    filter_parts, durations, _raw_durations = build_base_filters(
        video_paths=[video_path],
        target_w=target_w,
        target_h=target_h,
        trim_start=trim_start,
        trim_end=trim_end,
        audio_fade_in=audio_fade_in,
        audio_fade_out=audio_fade_out,
        include_audio=include_audio,
    )
    filter_parts = build_concat_graph(
        filter_parts=filter_parts,
        n=1,
        durations=durations,
        xfade_type=None,
        transition_duration=0.0,
        include_audio=include_audio,
    )

    total_duration = compute_total_duration(
        durations=durations,
        xfade_type=None,
        transition_duration=0.0,
        n=1,
    )

    # 构建 FFmpeg 命令
    cmd = [FFMPEG_BIN, "-y", "-i", str(video_path), "-i", str(music_path)]

    # 背景音乐处理
    bgm_filters = [f"volume={music_volume}"]
    if total_duration > audio_fade_in + audio_fade_out:
        if audio_fade_in > 0:
            bgm_filters.append(f"afade=t=in:st=0:d={audio_fade_in}:curve=tri")
        if audio_fade_out > 0:
            bgm_filters.append(
                f"afade=t=out:st={total_duration - audio_fade_out:.3f}:d={audio_fade_out}:curve=tri"
            )
    bgm_filter_str = ",".join(bgm_filters)

    if keep_original_audio:
        filter_parts.append(f"[1:a]{bgm_filter_str}[bgm]")
        filter_parts.append("[outa][bgm]amix=inputs=2:duration=first:dropout_transition=2[final_a]")
        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += ["-map", "[outv]", "-map", "[final_a]"]
    else:
        filter_parts.append(f"[1:a]{bgm_filter_str}[bgm_a]")
        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += ["-map", "[outv]", "-map", "[bgm_a]"]

    cmd += [
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]

    logger.info("开始添加背景音乐...")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-800:]}")

    if not Path(output_path).is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    out_info = get_video_info(str(output_path))
    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)

    logger.info(
        f"✅ 添加音乐完成: {Path(output_path).name} "
        f"({out_info.get('width', 0)}x{out_info.get('height', 0)}, "
        f"{out_info.get('duration', 0):.1f}s, {output_size_mb:.1f} MB)"
    )

    return {
        "output": str(output_path),
        "video_count": 1,
        "duration": out_info.get("duration", 0),
        "size_mb": round(output_size_mb, 2),
        "has_bgm": True,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="为单个视频添加背景音乐")
    parser.add_argument("video", help="要处理的视频文件")
    parser.add_argument("-m", "--music", required=True, help="背景音乐路径")
    parser.add_argument("--music-volume", type=float, default=0.3, help="音乐音量 (0.0~1.0)")
    parser.add_argument("--keep-audio", action="store_true", help="保留原视频声音")
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF")

    args = parser.parse_args()

    result = add_bgm_to_video(
        video_path=args.video,
        music_path=args.music,
        music_volume=args.music_volume,
        keep_original_audio=args.keep_audio,
        crf=args.crf,
    )
    print(f"\n✅ 添加音乐完成 → {result['duration']:.1f}s ({result['size_mb']} MB)")
    print(f"   输出: {result['output']}")
