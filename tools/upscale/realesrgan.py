"""
Real-ESRGAN 视频超分辨率模块

方法说明:
  - 使用 realesrgan-ncnn-vulkan 二进制工具逐帧超分
  - 无需 Python AI 依赖 (PyTorch / basicsr), 通过 Vulkan GPU 加速
  - 支持 2x / 3x / 4x 放大
  - 适合: 低分辨率视频清晰化 (如 480p → 1080p, 720p → 4K)

前置条件:
  bin/realesrgan-ncnn-vulkan 可执行文件 (从 GitHub Releases 下载)

使用方式:
  python tools/upscale/realesrgan.py video.mp4
  python tools/upscale/realesrgan.py video.mp4 --scale 4
  python tools/upscale/realesrgan.py video.mp4 --target-width 1920 --target-height 1080
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


from config import OUTPUT_UPSCALE, UPSCALE_FACTOR, FFMPEG_BIN
from tools.common import logger, VideoFrameProcessor, generate_output_name, get_video_info


# realesrgan-ncnn-vulkan 二进制路径 (项目 bin/ 目录)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REALESRGAN_BIN = _PROJECT_ROOT / "bin" / "realesrgan-ncnn-vulkan"
REALESRGAN_MODELS = _PROJECT_ROOT / "bin" / "models"


def _check_realesrgan() -> bool:
    """检查 realesrgan-ncnn-vulkan 是否可用"""
    return REALESRGAN_BIN.is_file()


def _auto_select_scale(src_width: int, src_height: int, target_width: int, target_height: int) -> int:
    """
    根据源分辨率和目标分辨率, 自动选择最小 Real-ESRGAN 倍数
    选择原则: 超分后的分辨率必须 >= 目标分辨率
    """
    ratio_w = target_width / src_width
    ratio_h = target_height / src_height
    ratio = max(ratio_w, ratio_h)

    if ratio <= 2.0:
        return 2
    elif ratio <= 3.0:
        return 3
    elif ratio <= 4.0:
        return 4
    else:
        logger.warning(f"目标分辨率需要 {ratio:.1f}x 放大, 超出 Real-ESRGAN 4x 上限, 将使用 4x")
        return 4


def _ffmpeg_rescale(input_path: Path, output_path: Path, target_width: int, target_height: int):
    """
    使用 FFmpeg 精确缩放到目标分辨率 (用于超分后调整到精确尺寸)
    """
    # 确保尺寸为偶数 (h264 要求)
    target_width = target_width + (target_width % 2)
    target_height = target_height + (target_height % 2)

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(input_path),
        "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 缩放失败:\n{result.stderr[-500:]}")


def _run_realesrgan_on_frames(input_dir: Path, output_dir: Path, scale: int = 2, model: str = "realesr-animevideov3"):
    """
    对一个目录的图片帧批量执行 realesrgan-ncnn-vulkan 超分
    """
    cmd = [
        str(REALESRGAN_BIN),
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-s", str(scale),
        "-n", model,
        "-m", str(REALESRGAN_MODELS),
        "-f", "png",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"realesrgan-ncnn-vulkan 执行失败:\n{result.stderr[-500:]}")


def upscale_video_realesrgan(
    video_path: str | Path,
    output_path: str | Path | None = None,
    scale: int | None = None,
    device: str | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    model: str = "realesr-animevideov3",
) -> dict:
    """
    使用 Real-ESRGAN 对视频超分辨率放大

    流程: 视频 → 提取帧 → realesrgan-ncnn-vulkan 超分 → 重组视频 → [可选] FFmpeg 精确缩放

    Args:
        video_path: 输入视频路径
        output_path: 输出路径
        scale: 放大倍数 (2/3/4), 与 target_width/target_height 二选一
        device: (保留参数, ncnn-vulkan 自动选择 GPU)
        target_width: 目标宽度 (与 scale 二选一, 自动选择最佳倍数)
        target_height: 目标高度 (与 scale 二选一)
        model: ncnn 模型名称

    Returns:
        dict: {"output": str, "frames_processed": int,
               "original_size": str, "upscaled_size": str}
    """
    if not _check_realesrgan():
        raise RuntimeError(
            "❌ 未找到 realesrgan-ncnn-vulkan 二进制文件\n"
            f"   期望路径: {REALESRGAN_BIN}\n"
            "   请从 GitHub 下载: https://github.com/xinntao/Real-ESRGAN/releases"
        )

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 获取源视频信息
    info = get_video_info(video_path)
    src_width = info.get("width", 0)
    src_height = info.get("height", 0)
    fps = info.get("fps", 30)

    # 竖屏视频: 自动翻转目标分辨率 (1920x1080 → 1080x1920)
    if target_width and target_height and src_height > src_width:
        if target_width > target_height:
            target_width, target_height = target_height, target_width
            logger.info(f"检测到竖屏视频, 自动调整目标分辨率为 {target_width}x{target_height}")

    # 判断模式: 目标分辨率 vs 倍数放大
    need_rescale = False
    if target_width and target_height:
        scale = _auto_select_scale(src_width, src_height, target_width, target_height)
        upscaled_w = src_width * scale
        upscaled_h = src_height * scale
        need_rescale = (upscaled_w != target_width or upscaled_h != target_height)
        logger.info(f"目标分辨率: {target_width}x{target_height}, 自动选择 {scale}x 超分")
    elif scale is None:
        scale = UPSCALE_FACTOR

    if scale not in (2, 3, 4):
        raise ValueError("scale 仅支持 2, 3 或 4")

    # 准备输出路径
    OUTPUT_UPSCALE.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        if target_width and target_height:
            tag = f"_{target_width}x{target_height}"
        else:
            tag = f"_{scale}x"
        output_path = OUTPUT_UPSCALE / generate_output_name(video_path.stem, ".mp4", tag)
    output_path = Path(output_path)

    actual_output = output_path
    if need_rescale:
        ai_output = output_path.with_name(f".tmp_ai_{output_path.name}")
    else:
        ai_output = output_path

    original_size = f"{src_width}x{src_height}"

    # ==========================================
    # 流程: 提取帧 → 超分 → 重组视频
    # ==========================================
    with tempfile.TemporaryDirectory(prefix="xtools_esrgan_") as tmpdir:
        frames_dir = Path(tmpdir) / "frames"
        upscaled_dir = Path(tmpdir) / "upscaled"
        frames_dir.mkdir()
        upscaled_dir.mkdir()

        # Step 1: 提取所有帧为 PNG
        logger.info(f"提取视频帧: {video_path.name}")
        extract_cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(video_path),
            "-qscale:v", "1",
            "-qmin", "1",
            str(frames_dir / "frame_%08d.png"),
        ]
        result = subprocess.run(extract_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"帧提取失败:\n{result.stderr[-500:]}")

        frame_files = sorted(frames_dir.glob("frame_*.png"))
        total_frames = len(frame_files)
        logger.info(f"共提取 {total_frames} 帧, 开始 Real-ESRGAN {scale}x 超分...")

        # Step 2: 执行 realesrgan-ncnn-vulkan 超分 (整个目录)
        _run_realesrgan_on_frames(frames_dir, upscaled_dir, scale=scale, model=model)

        # Step 3: 重组为视频
        logger.info("重组超分帧为视频...")
        upscaled_files = sorted(upscaled_dir.glob("frame_*.png"))
        if not upscaled_files:
            raise RuntimeError("超分输出为空, 请检查 realesrgan-ncnn-vulkan 日志")

        reassemble_cmd = [
            FFMPEG_BIN, "-y",
            "-framerate", str(fps),
            "-i", str(upscaled_dir / "frame_%08d.png"),
            "-i", str(video_path),
            "-map", "0:v",     # 使用超分的视频帧
            "-map", "1:a?",    # 保留原始音频 (如果有)
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-c:a", "copy",
            "-pix_fmt", "yuv420p",
            str(ai_output),
        ]
        result = subprocess.run(reassemble_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"视频重组失败:\n{result.stderr[-500:]}")

    frames_processed = total_frames

    # 后缩放: AI 超分结果 → 精确目标分辨率
    if need_rescale:
        logger.info(f"精确缩放: {src_width * scale}x{src_height * scale} → {target_width}x{target_height}")
        _ffmpeg_rescale(ai_output, actual_output, target_width, target_height)
        ai_output.unlink(missing_ok=True)
        upscaled_size = f"{target_width}x{target_height}"
    else:
        upscaled_size = f"{src_width * scale}x{src_height * scale}"

    logger.info(f"✅ 超分完成: {actual_output.name} ({original_size} → {upscaled_size})")

    return {
        "output": str(actual_output),
        "frames_processed": frames_processed,
        "original_size": original_size,
        "upscaled_size": upscaled_size,
    }


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Real-ESRGAN 视频超分")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument(
        "-s", "--scale",
        type=int, choices=[2, 3, 4], default=None,
        help=f"放大倍数 (默认: {UPSCALE_FACTOR})",
    )
    parser.add_argument("--target-width", type=int, help="目标宽度")
    parser.add_argument("--target-height", type=int, help="目标高度")
    parser.add_argument(
        "--model", default="realesr-animevideov3",
        help="模型名称 (默认: realesr-animevideov3)",
    )

    args = parser.parse_args()

    if not _check_realesrgan():
        print(f"❌ 未找到 realesrgan-ncnn-vulkan: {REALESRGAN_BIN}")
        sys.exit(1)

    result = upscale_video_realesrgan(
        video_path=args.input,
        output_path=args.output,
        scale=args.scale,
        target_width=args.target_width,
        target_height=args.target_height,
        model=args.model,
    )
    print(f"\n✅ 超分完成: {result['original_size']} → {result['upscaled_size']}")
    print(f"   输出: {result['output']}")
