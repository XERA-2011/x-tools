"""
歌词 MV 生成主控
整合解析、节拍、渲染和导出网络
"""
import os
import shutil
import tempfile
import struct
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import FFMPEG_BIN, OUTPUT_MV
from tools.common import generate_output_name, logger, get_video_info

from tools.mv.parser import load_lyrics, from_whisper_segments
from tools.mv.beat_detector import detect_beats
from tools.mv.renderer import LyricRenderer

def generate_mv(
    music_path: Path,
    lyrics_path: Path | None = None,
    whisper_segments: list | None = None,
    output_path: Path | None = None,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    highlight_color: str = "#FF3333",
    glow: bool = True,
    bilingual: bool = False,
    bg_image_path: Path | None = None,
) -> dict:
    """
    生成歌词 MV 主函数
    """
    music_path = Path(music_path)
    if not music_path.is_file():
        raise FileNotFoundError(f"音乐文件不存在: {music_path}")

    # 1. 准备配置和输出
    OUTPUT_MV.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        tag = f"{width}x{height}"
        output_name = generate_output_name(music_path.stem, ".mp4", tag=tag)
        output_path = OUTPUT_MV / output_name
    output_path = Path(output_path)

    # 2. 节拍分析
    logger.info("🎬 [1/4] 分析音乐节拍...")
    beat_info = detect_beats(music_path)

    # 3. 歌词处理
    logger.info("🎵 [2/4] 加载歌词...")
    if whisper_segments:
        lyrics = from_whisper_segments(whisper_segments)
    elif lyrics_path and lyrics_path.is_file():
        lyrics = load_lyrics(lyrics_path, beat_info.duration)
    else:
        # Fallback to empty
        lyrics = []

    # 4. 渲染帧序列
    logger.info(f"🎨 [3/4] 渲染图像帧 ({fps} FPS)...")
    renderer = LyricRenderer(
        width=width,
        height=height,
        fps=fps,
        highlight_color=highlight_color,
        glow=glow,
        bg_image_path=bg_image_path,
    )
    
    total_frames = int(beat_info.duration * fps)
    
    # 使用临时文件流直接喂给 FFmpeg, 避免生成百万个小文件
    # 创建一个 pipe
    import subprocess
    cmd = [
        FFMPEG_BIN, "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{width}x{height}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-", # 从 stdin 读
        "-i", str(music_path), # 输入音频
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", # 取最短的时长
        str(output_path)
    ]
    
    logger.info("🚀 [4/4] FFmpeg 合成视频...")
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 渲染循环 (可进一步用多线程优化, 这里为了演示使用单线程 + direct pipe)
    # 进度显示
    from rich.progress import Progress
    with Progress() as progress:
        task = progress.add_task("渲染并编码帧...", total=total_frames)
        try:
            for frame_idx in range(total_frames):
                time = frame_idx / fps
                img = renderer.render_frame(time, lyrics, beat_info)
                # 写入管道
                process.stdin.write(img.tobytes())
                progress.advance(task)
        except IOError as e:    
            logger.error(f"Pipe broken: {e}")
        finally:
            if process.stdin:
                process.stdin.close()
                
    # 等待 FFmpeg 完成 (可能在提取 stderr 时打印错误)
    process.wait()
    if process.returncode != 0:
        stderr_log = process.stderr.read().decode('utf-8', errors='replace')
        logger.error(f"FFmpeg Error:\n{stderr_log[-500:]}")
        raise RuntimeError("视频合成失败")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ 生成完毕: {output_path.name} ({size_mb:.1f} MB)")
    
    return {
        "output": str(output_path),
        "duration": beat_info.duration,
        "frames": total_frames,
        "size_mb": round(size_mb, 2),
    }

