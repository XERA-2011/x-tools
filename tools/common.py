"""
公共工具模块 — 批量调度、日志、进度条、视频信息获取
"""
import json
import logging
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from config import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, FFPROBE_BIN, FFMPEG_BIN

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("x-tools")


# ============================================================
# 视频信息
# ============================================================
def get_video_info(video_path: str | Path) -> dict:
    """
    使用 ffprobe 获取视频信息 (时长、分辨率、帧率等)

    Returns:
        dict: {
            "duration": float,   # 秒
            "width": int,
            "height": int,
            "fps": float,
            "codec": str,
            "bitrate": int,      # bps
        }
    """
    video_path = str(video_path)
    cmd = [
        FFPROBE_BIN,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"无法读取视频信息: {video_path} — {e}")
        return {}

    # 提取视频流信息
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        {},
    )
    fmt = data.get("format", {})

    # 解析帧率 (如 "30000/1001")
    fps_str = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, fps_str.split("/"))
        fps = num / den if den else 0
    except (ValueError, ZeroDivisionError):
        fps = 0

    return {
        "duration": float(fmt.get("duration", 0)),
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": round(fps, 2),
        "codec": video_stream.get("codec_name", "unknown"),
        "bitrate": int(fmt.get("bit_rate", 0)),
    }


# ============================================================
# 文件扫描
# ============================================================
def scan_videos(directory: str | Path) -> list[Path]:
    """
    扫描目录下所有支持的视频文件 (非递归)

    Returns:
        list[Path]: 排序后的视频文件列表
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.error(f"目录不存在: {directory}")
        return []

    videos = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )
    logger.info(f"扫描到 {len(videos)} 个视频文件: {directory}")
    return videos


def scan_images(directory: str | Path) -> list[Path]:
    """
    扫描目录下所有支持的图片文件 (非递归)

    Returns:
        list[Path]: 排序后的图片文件列表
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.error(f"目录不存在: {directory}")
        return []

    images = sorted(
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info(f"扫描到 {len(images)} 个图片文件: {directory}")
    return images


def scan_media(directory: str | Path) -> tuple[list[Path], list[Path]]:
    """
    扫描目录下所有支持的媒体文件 (视频 + 图片)

    Returns:
        tuple: (videos, images)
    """
    return scan_videos(directory), scan_images(directory)


# ============================================================
# 批量执行器
# ============================================================
def batch_process(
    videos: list[Path],
    process_fn: Callable,
    desc: str = "处理中",
    max_workers: int = 1,
    **kwargs,
) -> list[dict]:
    """
    批量处理视频文件

    Args:
        videos: 视频文件列表
        process_fn: 处理函数, 签名为 process_fn(video_path: Path, **kwargs) -> dict
        desc: 进度条描述
        max_workers: 并行工作进程数 (默认 1，串行执行)
        **kwargs: 传递给 process_fn 的额外参数

    Returns:
        list[dict]: 每个视频的处理结果
    """
    results = []

    if not videos:
        logger.warning("没有需要处理的视频文件")
        return results

    if max_workers <= 1:
        # 串行执行 — 带进度条
        for video in tqdm(videos, desc=desc, unit="个"):
            try:
                result = process_fn(video, **kwargs)
                results.append({"file": str(video), "status": "success", **result})
            except Exception as e:
                logger.error(f"处理失败: {video.name} — {e}")
                results.append({"file": str(video), "status": "error", "error": str(e)})
    else:
        # 并行执行
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_fn, v, **kwargs): v for v in videos
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc=desc, unit="个"
            ):
                video = futures[future]
                try:
                    result = future.result()
                    results.append({"file": str(video), "status": "success", **result})
                except Exception as e:
                    logger.error(f"处理失败: {video.name} — {e}")
                    results.append({"file": str(video), "status": "error", "error": str(e)})

    # 汇总
    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    logger.info(f"批量处理完成: ✅ {success} 成功, ❌ {failed} 失败")

    return results


def print_summary(results: list[dict]):
    """打印批量处理结果摘要"""
    print("\n" + "=" * 60)
    print("📊 处理结果摘要")
    print("=" * 60)

    for r in results:
        name = Path(r["file"]).name
        if r["status"] == "success":
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name} — {r.get('error', '未知错误')}")

    success = sum(1 for r in results if r["status"] == "success")
    print(f"\n合计: {success}/{len(results)} 成功")
    print("=" * 60)


# ============================================================
# 音频合并 (公共)
# ============================================================
def merge_audio(original_video: Path, processed_video: str, output_path: Path):
    """
    将原视频的音频合并到处理后的视频中

    Args:
        original_video: 原始视频路径 (取音频)
        processed_video: 处理后的视频路径 (取视频流, 无音频)
        output_path: 最终输出路径
    """
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(processed_video),    # 修复后的视频 (无音频)
        "-i", str(original_video),     # 原视频 (取音频)
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",              # 用处理后的视频流
        "-map", "1:a:0?",             # 用原视频的音频流 (可选, 原视频可能无音频)
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # 如果混合失败 (比如原视频无音频), 直接用 libx264 重编码视频
        logger.warning("音频混合失败, 仅输出视频")
        cmd_fallback = [
            FFMPEG_BIN, "-y",
            "-i", str(processed_video),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            str(output_path),
        ]
        subprocess.run(cmd_fallback, capture_output=True, text=True, check=True)
