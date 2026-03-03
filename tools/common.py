"""
公共工具模块 — 批量调度、日志、进度条、视频信息获取
"""
import json
import logging
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Callable, Generator

import cv2
import numpy as np
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.console import Console

from config import VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, FFPROBE_BIN, FFMPEG_BIN, OUTPUT_DIR

# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("x-tools")
console = Console()


# ============================================================
# 辅助函数: 文件名生成
# ============================================================
def generate_output_name(stem: str, suffix: str, tag: str = "") -> str:
    """
    生成带时间戳的输出文件名, 避免覆盖
    格式: {stem}_{tag}_{MMDD_HHMMSS}{suffix}
    """
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    tag_part = f"_{tag}" if tag else ""
    return f"{stem}{tag_part}_{timestamp}{suffix}"


# ============================================================
# 视频信息
# ============================================================
def get_video_info(video_path: str | Path) -> dict:
    """
    使用 ffprobe 获取视频信息 (时长、分辨率、帧率等)
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
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', check=True)
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
def scan_videos(directory: str | Path, recursive: bool = False) -> list[Path]:
    """
    扫描目录下所有支持的视频文件
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.error(f"目录不存在: {directory}")
        return []

    pattern = "**/*" if recursive else "*"
    videos = sorted(
        f for f in directory.glob(pattern)
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )
    logger.info(f"扫描到 {len(videos)} 个视频文件: {directory} (递归: {recursive})")
    return videos


def scan_images(directory: str | Path, recursive: bool = False) -> list[Path]:
    """
    扫描目录下所有支持的图片文件
    """
    directory = Path(directory)
    if not directory.is_dir():
        logger.error(f"目录不存在: {directory}")
        return []

    pattern = "**/*" if recursive else "*"
    images = sorted(
        f for f in directory.glob(pattern)
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    logger.info(f"扫描到 {len(images)} 个图片文件: {directory} (递归: {recursive})")
    return images


def scan_media(directory: str | Path, recursive: bool = False) -> tuple[list[Path], list[Path]]:
    """
    扫描目录下所有支持的媒体文件 (视频 + 图片)
    """
    return scan_videos(directory, recursive), scan_images(directory, recursive)


# ============================================================
# 批量执行器 (Rich Progress)
# ============================================================
def batch_process(
    videos: list[Path],
    process_fn: Callable,
    desc: str = "处理中",
    max_workers: int = 1,
    dry_run: bool = False,
    **kwargs,
) -> list[dict]:
    """
    批量处理视频文件
    """
    results = []
    
    if not videos:
        logger.warning("没有需要处理的文件")
        return results

    if dry_run:
        print(f"\nExample dry run for {len(videos)} files:")
        for v in videos[:5]:
            print(f"  - {v.name}")
        if len(videos) > 5:
            print(f"  ... and {len(videos) - 5} more")
        return []

    # 记录总开始时间
    start_time_all = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task_id = progress.add_task(desc, total=len(videos))

        if max_workers <= 1:
            # 串行执行
            for video in videos:
                start_time = time.time()
                try:
                    result = process_fn(video, **kwargs)
                    elapsed = time.time() - start_time
                    results.append({"file": str(video), "status": "success", "elapsed": elapsed, **result})
                except Exception as e:
                    elapsed = time.time() - start_time
                    logger.error(f"处理失败: {video.name} — {e}")
                    results.append({"file": str(video), "status": "error", "error": str(e), "elapsed": elapsed})
                finally:
                    progress.advance(task_id)
        else:
            # 并行执行
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_fn, v, **kwargs): v for v in videos
                }
                for future in as_completed(futures):
                    video = futures[future]
                    start_time = time.time() # 注意:这是近似时间,并行下不准确
                    try:
                        result = future.result()
                        # 这里无法准确获取单个任务执行时间,暂用0
                        results.append({"file": str(video), "status": "success", "elapsed": 0.0, **result})
                    except Exception as e:
                        logger.error(f"处理失败: {video.name} — {e}")
                        results.append({"file": str(video), "status": "error", "error": str(e), "elapsed": 0.0})
                    finally:
                        progress.advance(task_id)

    total_elapsed = time.time() - start_time_all
    
    # 汇总
    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    logger.info(f"批量处理完成: ✅ {success} 成功, ❌ {failed} 失败, 总耗时: {total_elapsed:.1f}s")

    return results


def print_summary(results: list[dict]):
    """打印批量处理结果摘要"""
    if not results:
        return

    print("\n" + "=" * 60)
    print("📊 处理结果摘要")
    print("=" * 60)

    total_time = 0.0
    for r in results:
        name = Path(r["file"]).name
        elapsed = r.get("elapsed", 0.0)
        total_time += elapsed
        if r["status"] == "success":
            time_str = f"({elapsed:.1f}s)" if elapsed > 0 else ""
            print(f"  ✅ {name} {time_str}")
        else:
            print(f"  ❌ {name} — {r.get('error', '未知错误')}")

    success = sum(1 for r in results if r["status"] == "success")
    avg_time = (total_time / len(results)) if results else 0
    
    print("-" * 60)
    print(f"合计: {success}/{len(results)} 成功")
    if total_time > 0:
        print(f"平均耗时: {avg_time:.1f}s / 文件")
    print("=" * 60)


# ============================================================
# VideoProcessor 上下文管理器 (抽象基类)
# ============================================================
class VideoFrameProcessor:
    """
    通用视频逐帧处理上下文管理器
    
    Usage:
        with VideoFrameProcessor(video_path, output_path) as vp:
             for frame in vp.frames():
                 processed_frame = do_something(frame)
                 vp.write(processed_frame)
    """
    def __init__(self, input_path: str | Path, output_path: str | Path, enable_merge_audio: bool = True):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.enable_merge_audio = enable_merge_audio
        self.temp_path = None
        self.cap = None
        self.writer = None
        self.fps = 0.0
        self.width = 0
        self.height = 0
        self.total_frames = 0
        self.frames_processed = 0

    def __enter__(self):
        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频: {self.input_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 创建临时文件 (同目录下的隐藏文件)
        import tempfile
        # 使用 tempfile 生成临时文件，但在 output_path 同级目录，避免跨盘符移动慢
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # 用 .tmp 后缀
        self.temp_path = self.output_path.with_name(f".tmp_{self.output_path.name}")
        
        return self

    def init_writer(self, width: int = 0, height: int = 0, fps: float = 0.0):
        """初始化写入器 (可选, 默认使用原视频参数)"""
        w = width if width > 0 else self.width
        h = height if height > 0 else self.height
        f = fps if fps > 0 else self.fps
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.temp_path), fourcc, f, (w, h))

    def frames(self, desc: str = "Processing") -> Generator[np.ndarray, None, None]:
        """生成器：逐帧读取并显示进度条"""
        if not self.writer:
            self.init_writer() # 默认初始化

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as p:
            task = p.add_task(desc, total=self.total_frames)
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                yield frame
                p.advance(task)
                self.frames_processed += 1

    def write(self, frame: np.ndarray):
        """写入一帧"""
        if self.writer:
            self.writer.write(frame)

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.cap:
                self.cap.release()
            if self.writer:
                self.writer.release()
            
            if exc_type is None and self.frames_processed > 0:
                # 正常结束
                if self.enable_merge_audio:
                    merge_audio(self.input_path, self.temp_path, self.output_path)
                else:
                    import shutil
                    if self.output_path.exists():
                         self.output_path.unlink()
                    shutil.move(str(self.temp_path), str(self.output_path))
        finally:
            # 无论正常还是异常结束，强制清理临时文件
            if getattr(self, "temp_path", None) and self.temp_path.exists():
                try:
                    self.temp_path.unlink()
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {self.temp_path} - {e}")


# ============================================================
# 音频合并 (公共)
# ============================================================
def merge_audio(original_video: Path, processed_video: Path, output_path: Path):
    """
    将原视频的音频合并到处理后的视频中。
    OpenCV VideoWriter 使用 mp4v (MPEG-4 Part 2) 编码临时文件，
    必须通过 FFmpeg 重编码为 H.264 (libx264)，否则输出的 MP4 在
    微信视频号等平台导入时会因编解码器不兼容而失败。
    """
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(processed_video),    # 修复后的视频 (无音频, mp4v)
        "-i", str(original_video),     # 原视频 (取音频)
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",  # 重编码为 H.264，确保兼容性
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",              # 用处理后的视频流
        "-map", "1:a:0?",             # 用原视频的音频流 (可选, 原视频可能无音频)
        "-shortest",
        "-movflags", "+faststart",     # 优化 MP4 流式播放 (moov atom 前置)
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        # 音频混合失败 (原视频无音频流等), 降级为仅输出 H.264 视频
        logger.warning("音频混合失败, 降级为纯视频输出")
        cmd_fallback = [
            FFMPEG_BIN, "-y",
            "-i", str(processed_video),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            "-movflags", "+faststart",
            str(output_path),
        ]
        result2 = subprocess.run(cmd_fallback, capture_output=True, text=True, encoding='utf-8', errors='replace')
        if result2.returncode != 0:
            logger.error(f"FFmpeg 重编码失败: {result2.stderr}")
            import shutil
            shutil.copy(processed_video, output_path)


# ============================================================
# 公共工具函数
# ============================================================
def parse_region(s: str) -> tuple[int, int, int, int]:
    """
    解析区域字符串 'x1,y1,x2,y2'

    Args:
        s: 格式为 "x1,y1,x2,y2" 的字符串

    Returns:
        (x1, y1, x2, y2) 整数元组

    Raises:
        ValueError: 格式不正确时
    """
    parts = [int(x.strip()) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("区域格式: x1,y1,x2,y2")
    return tuple(parts)


def calc_overlay_position(
    base_width: int,
    base_height: int,
    overlay_width: int,
    overlay_height: int,
    position: str | tuple[int, int],
    margin: int,
) -> tuple[int, int]:
    """
    根据位置名称计算叠加层坐标 (水印文字 / Logo 通用)

    Args:
        base_width, base_height: 底图尺寸
        overlay_width, overlay_height: 叠加层尺寸
        position: 位置 ("bottom-right" / "top-left" / ... / (x, y))
        margin: 边距

    Returns:
        (x, y) 坐标
    """
    if isinstance(position, (list, tuple)):
        return int(position[0]), int(position[1])

    positions = {
        "bottom-right": (base_width - overlay_width - margin, base_height - overlay_height - margin),
        "bottom-left": (margin, base_height - overlay_height - margin),
        "bottom-center": ((base_width - overlay_width) // 2, base_height * 6 // 7 - overlay_height // 2),
        "bottom-center-5": ((base_width - overlay_width) // 2, base_height * 4 // 5 - overlay_height // 2),
        "bottom-center-10": ((base_width - overlay_width) // 2, base_height * 9 // 10 - overlay_height // 2),
        "bottom-center-16": ((base_width - overlay_width) // 2, base_height * 15 // 16 - overlay_height // 2),
        "top-right": (base_width - overlay_width - margin, margin),
        "top-left": (margin, margin),
        "center": ((base_width - overlay_width) // 2, (base_height - overlay_height) // 2),
    }
    return positions.get(position, positions["bottom-right"])


def orient_resolution(
    src_width: int, src_height: int,
    target_width: int, target_height: int,
) -> tuple[int, int]:
    """
    竖屏视频自动翻转目标分辨率 (如 1920x1080 → 1080x1920)

    Returns:
        (target_width, target_height) 调整后的目标分辨率
    """
    if src_height > src_width and target_width > target_height:
        logger.info(f"检测到竖屏视频, 自动调整目标分辨率为 {target_height}x{target_width}")
        return target_height, target_width
    return target_width, target_height


def load_or_create_mask(
    width: int,
    height: int,
    regions: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    feather: int = 3,
    ref_width: int = 0,
    ref_height: int = 0,
) -> np.ndarray:
    """
    加载 mask 文件或根据区域列表创建 mask (公共逻辑)

    必须指定 regions 或 mask_path 其中之一

    Args:
        width, height: 视频尺寸
        regions: 水印区域列表 [(x1,y1,x2,y2), ...]
        mask_path: mask 图片路径
        feather: 边缘羽化
        ref_width, ref_height: ROI 坐标的参考分辨率 (0=不缩放)

    Returns:
        二值 mask (uint8, 白色=水印区域)
    """
    from tools.watermark.opencv_inpaint import create_mask_from_regions, _scale_regions

    # 按参考分辨率缩放坐标
    if regions and ref_width > 0 and ref_height > 0:
        regions = _scale_regions(regions, ref_width, ref_height, width, height)

    if mask_path:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取 mask: {mask_path}")
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = create_mask_from_regions(width, height, regions, feather)

    return mask
