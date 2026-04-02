"""
视频拼接模块

功能:
  - 将多个视频按顺序拼接为一个
  - 自动统一分辨率和编码格式
  - 可选配背景音乐 (替换或混合原声)
  - 支持从 music/ 目录选择内置音乐

使用方式:
  python tools/concat/ffmpeg_concat.py video1.mp4 video2.mp4
  python tools/concat/ffmpeg_concat.py video1.mp4 video2.mp4 -m music/bgm.mp3
"""
import subprocess
import tempfile
from pathlib import Path

from config import FFMPEG_BIN, FFPROBE_BIN, MUSIC_DIR, OUTPUT_CONCAT
from tools.common import generate_output_name, get_video_info, logger
from tools.ffmpeg.ffmpeg_core import (
    build_base_filters,
    build_concat_graph,
    compute_total_duration,
)

# 支持的音频格式
AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg", ".wma"}


def get_available_music() -> list[Path]:
    """获取 music/ 目录下所有可用音乐文件"""
    if not MUSIC_DIR.is_dir():
        return []
    files = []
    for ext in AUDIO_EXTENSIONS:
        files.extend(MUSIC_DIR.glob(f"*{ext}"))
    return sorted(files, key=lambda f: f.name)


# 过渡效果预设
TRANSITION_PRESETS = {
    "none": {"name": "⏩ 无过渡 (直接拼接)", "xfade": None},
    "fade": {"name": "🌑 淡入淡出", "xfade": "fade"},
}


def concat_videos(
    video_paths: list[str | Path],
    output_path: str | Path | None = None,
    music_path: str | Path | None = None,
    music_volume: float = 0.3,
    keep_original_audio: bool = False,
    crf: int = 18,
    transition: str = "none",
    transition_duration: float = 1.0,
    audio_fade_in: float = 0.0,
    audio_fade_out: float = 0.0,
    mute_audio: bool = False,
) -> dict:
    """
    拼接多个视频为一个

    Args:
        video_paths: 视频文件路径列表 (按顺序拼接)
        output_path: 输出路径 (默认自动生成)
        music_path: 背景音乐路径 (None=不加音乐)
        music_volume: 背景音乐音量 (0.0~1.0, 默认 0.3)
        keep_original_audio: 是否保留原视频声音 (True=混合, False=替换)
        crf: 视频质量
        audio_fade_in: 首尾音频淡入时长 (秒)
        audio_fade_out: 首尾音频淡出时长 (秒)
        mute_audio: 是否拼接后输出静音视频

    Returns:
        dict: {"output": str, "duration": float, ...}
    """
    video_paths = [Path(p) for p in video_paths]

    # 验证文件存在
    for vp in video_paths:
        if not vp.is_file():
            raise FileNotFoundError(f"视频文件不存在: {vp}")

    # 如果只有 1 个视频，必须添加背景音乐才有意义
    if len(video_paths) < 2 and music_path is None:
        raise ValueError("至少需要 2 个视频才能拼接，或者为单个视频添加背景音乐")

    if music_path is not None:
        music_path = Path(music_path)
        if not music_path.is_file():
            raise FileNotFoundError(f"音乐文件不存在: {music_path}")

    # 获取第一个视频的信息作为基准
    first_info = get_video_info(str(video_paths[0]))
    target_w = first_info.get("width", 1920)
    target_h = first_info.get("height", 1080)

    # 确保偶数
    target_w = target_w // 2 * 2
    target_h = target_h // 2 * 2

    fade_msg = ""
    if audio_fade_in > 0 or audio_fade_out > 0:
        fade_msg = f" (音频过渡: 入 {audio_fade_in}s / 出 {audio_fade_out}s)"
    
    if len(video_paths) == 1:
        logger.info(f"为单个视频添加背景音乐 → {target_w}x{target_h}{fade_msg}")
    else:
        logger.info(f"拼接 {len(video_paths)} 个视频 → {target_w}x{target_h}{fade_msg}")

    # 输出路径
    OUTPUT_CONCAT.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        if len(video_paths) == 1:
            output_name = generate_output_name("concat", ".mp4", tag="with_bgm")
        else:
            output_name = generate_output_name("concat", ".mp4", tag=f"{len(video_paths)}in1")
        output_path = OUTPUT_CONCAT / output_name
    output_path = Path(output_path)

    # 构建 FFmpeg concat filter 命令
    n = len(video_paths)

    cmd = [FFMPEG_BIN, "-y"]

    # 输入所有视频
    for vp in video_paths:
        cmd += ["-i", str(vp)]

    # 如果有背景音乐, 也作为输入
    music_input_idx = None
    if music_path is not None:
        music_input_idx = n
        cmd += ["-i", str(music_path)]

    # 构建 filter_complex
    xfade_type = TRANSITION_PRESETS.get(transition, {}).get("xfade")
    td = transition_duration
    if mute_audio:
        include_audio = False
    else:
        include_audio = (music_path is None or keep_original_audio)
    filter_parts, durations, _raw_durations = build_base_filters(
        video_paths=video_paths,
        target_w=target_w,
        target_h=target_h,
        trim_start=0.0,
        trim_end=0.0,
        audio_fade_in=audio_fade_in,
        audio_fade_out=audio_fade_out,
        include_audio=include_audio,
    )
    filter_parts = build_concat_graph(
        filter_parts=filter_parts,
        n=n,
        durations=durations,
        xfade_type=xfade_type,
        transition_duration=td,
        include_audio=include_audio,
    )

    # 背景音乐处理
    if music_path is not None:
        # 计算总视频时长 (用于背景音乐淡入淡出)
        total_duration = compute_total_duration(
            durations=durations,
            xfade_type=xfade_type,
            transition_duration=td,
            n=n,
        )
        
        # 构建背景音乐 filter (音量 + 淡入淡出)
        bgm_filters = [f"volume={music_volume}"]
        
        # 为背景音乐添加淡入淡出效果
        if total_duration > audio_fade_in + audio_fade_out:
            if audio_fade_in > 0:
                # 使用 tri (三角形) 曲线，更平滑自然
                bgm_filters.append(f"afade=t=in:st=0:d={audio_fade_in}:curve=tri")
            if audio_fade_out > 0:
                bgm_filters.append(f"afade=t=out:st={total_duration - audio_fade_out:.3f}:d={audio_fade_out}:curve=tri")
        
        bgm_filter_str = ",".join(bgm_filters)
        
        if keep_original_audio:
            # 混合: 原声 + BGM
            filter_parts.append(f"[{music_input_idx}:a]{bgm_filter_str}[bgm]")
            filter_parts.append("[outa][bgm]amix=inputs=2:duration=first:dropout_transition=2[final_a]")
            cmd += ["-filter_complex", ";".join(filter_parts)]
            cmd += ["-map", "[outv]", "-map", "[final_a]"]
        else:
            # 替换: 仅用 BGM
            filter_parts.append(f"[{music_input_idx}:a]{bgm_filter_str}[bgm_a]")
            cmd += ["-filter_complex", ";".join(filter_parts)]
            cmd += ["-map", "[outv]", "-map", "[bgm_a]"]
    else:
        # 无 BGM: 用拼接后的原声
        cmd += ["-filter_complex", ";".join(filter_parts)]
        if include_audio:
            cmd += ["-map", "[outv]", "-map", "[outa]"]
        else:
            cmd += ["-map", "[outv]"]

    # 编码参数
    cmd += [
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
    ]
    if include_audio or music_path is not None:
        cmd += ["-c:a", "aac", "-b:a", "192k"]
        
    cmd += [
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]

    logger.info(f"开始拼接...")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-800:]}")

    if not output_path.is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    # 获取输出信息
    out_info = get_video_info(str(output_path))
    output_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(
        f"✅ {'添加音乐完成' if len(video_paths) == 1 else '拼接完成'}: {output_path.name} "
        f"({out_info.get('width', 0)}x{out_info.get('height', 0)}, "
        f"{out_info.get('duration', 0):.1f}s, {output_size_mb:.1f} MB)"
    )

    return {
        "output": str(output_path),
        "video_count": len(video_paths),
        "duration": out_info.get("duration", 0),
        "size_mb": round(output_size_mb, 2),
        "has_bgm": music_path is not None,
    }



# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频拼接")
    parser.add_argument("videos", nargs="+", help="要拼接的视频文件")
    parser.add_argument("-m", "--music", help="背景音乐路径")
    parser.add_argument("--music-volume", type=float, default=0.3, help="音乐音量 (0.0~1.0)")
    parser.add_argument("--keep-audio", action="store_true", help="保留原视频声音")
    parser.add_argument("--mute", action="store_true", help="拼接后消音")
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF")

    args = parser.parse_args()

    result = concat_videos(
        video_paths=args.videos,
        music_path=args.music,
        music_volume=args.music_volume,
        keep_original_audio=args.keep_audio,
        crf=args.crf,
        mute_audio=args.mute,
    )
    print(f"\n✅ 拼接完成: {result['video_count']} 个视频 → {result['duration']:.1f}s ({result['size_mb']} MB)")
    print(f"   输出: {result['output']}")
