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

from config import FFMPEG_BIN, FFPROBE_BIN, OUTPUT_CONCAT, MUSIC_DIR
from tools.common import get_video_info, generate_output_name, logger

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


def concat_videos(
    video_paths: list[str | Path],
    output_path: str | Path | None = None,
    music_path: str | Path | None = None,
    music_volume: float = 0.3,
    keep_original_audio: bool = False,
    crf: int = 18,
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

    Returns:
        dict: {"output": str, "duration": float, ...}
    """
    video_paths = [Path(p) for p in video_paths]

    # 验证文件存在
    for vp in video_paths:
        if not vp.is_file():
            raise FileNotFoundError(f"视频文件不存在: {vp}")

    if len(video_paths) < 2:
        raise ValueError("至少需要 2 个视频才能拼接")

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

    logger.info(f"拼接 {len(video_paths)} 个视频 → {target_w}x{target_h}")

    # 输出路径
    OUTPUT_CONCAT.mkdir(parents=True, exist_ok=True)
    if output_path is None:
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
    # 策略: 始终只 concat 视频流 (避免部分视频无音频导致错误)
    # 音频单独处理: BGM 替换 / BGM+原声混合 / 仅原声拼接
    filter_parts = []

    for i in range(n):
        # scale + pad 确保所有视频尺寸一致, setsar=1 统一像素比
        filter_parts.append(
            f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1[v{i}]"
        )

    # concat 仅视频流
    video_streams = "".join(f"[v{i}]" for i in range(n))
    filter_parts.append(f"{video_streams}concat=n={n}:v=1:a=0[outv]")

    if music_path is not None and keep_original_audio:
        # 混合模式: 拼接原声 + BGM
        # 拼接各视频音频流
        audio_concat_parts = "".join(f"[{i}:a]" for i in range(n))
        filter_parts.append(f"{audio_concat_parts}concat=n={n}:v=0:a=1[orig_a]")
        filter_parts.append(f"[{music_input_idx}:a]volume={music_volume}[bgm]")
        filter_parts.append("[orig_a][bgm]amix=inputs=2:duration=first:dropout_transition=2[outa]")
        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += ["-map", "[outv]", "-map", "[outa]"]
    elif music_path is not None:
        # 替换模式: 仅用 BGM
        filter_parts.append(f"[{music_input_idx}:a]volume={music_volume}[outa]")
        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += ["-map", "[outv]", "-map", "[outa]"]
    else:
        # 无 BGM: 尝试拼接原声, 失败则静音输出
        audio_concat_parts = "".join(f"[{i}:a]" for i in range(n))
        filter_parts.append(f"{audio_concat_parts}concat=n={n}:v=0:a=1[outa]")
        cmd += ["-filter_complex", ";".join(filter_parts)]
        cmd += ["-map", "[outv]", "-map", "[outa]"]

    # 编码参数
    cmd += [
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-c:a", "aac", "-b:a", "192k",
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
        f"✅ 拼接完成: {output_path.name} "
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
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF")

    args = parser.parse_args()

    result = concat_videos(
        video_paths=args.videos,
        music_path=args.music,
        music_volume=args.music_volume,
        keep_original_audio=args.keep_audio,
        crf=args.crf,
    )
    print(f"\n✅ 拼接完成: {result['video_count']} 个视频 → {result['duration']:.1f}s ({result['size_mb']} MB)")
    print(f"   输出: {result['output']}")
