"""
FFmpeg 传统视频放大模块

方法说明:
  - 使用 FFmpeg scale 滤镜进行视频放大
  - 支持多种插值算法: lanczos (推荐), bicubic, bilinear
  - 速度快, 无需 GPU, 但效果不如 AI 超分
  - 适合: 快速放大、对质量要求不高的场景

使用方式:
  python tools/upscale/ffmpeg_scale.py video.mp4 --scale 2
  python tools/upscale/ffmpeg_scale.py video.mp4 --width 1920 --height 1080
"""
import subprocess
from pathlib import Path


from config import FFMPEG_BIN, OUTPUT_UPSCALE, UPSCALE_FACTOR
from tools.common import get_video_info, logger, orient_resolution


def upscale_video_ffmpeg(
    video_path: str | Path,
    output_path: str | Path | None = None,
    scale: int | None = UPSCALE_FACTOR,
    width: int | None = None,
    height: int | None = None,
    algorithm: str = "lanczos",
    crf: int = 18,
) -> dict:
    """
    使用 FFmpeg 放大视频

    Args:
        video_path: 输入视频路径
        output_path: 输出路径
        scale: 放大倍数 (与 width/height 二选一)
        width: 目标宽度 (与 scale 二选一, -1 表示按比例)
        height: 目标高度 (与 scale 二选一, -1 表示按比例)
        algorithm: 插值算法 ("lanczos" / "bicubic" / "bilinear" / "spline")
        crf: 输出质量 (0-51, 越小越好, 推荐 18)

    Returns:
        dict: {"output": str, "original_size": str, "upscaled_size": str}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 获取原始信息
    info = get_video_info(video_path)
    if not info:
        raise RuntimeError(f"无法读取视频信息: {video_path}")

    orig_w, orig_h = info["width"], info["height"]

    # 竖屏视频: 自动翻转目标分辨率
    if width and height:
        width, height = orient_resolution(orig_w, orig_h, width, height)

    # 计算目标尺寸
    if width and height:
        target_w, target_h = width, height
    elif width:
        target_w = width
        target_h = -2  # FFmpeg 自动保持比例 (必须为偶数)
    elif height:
        target_w = -2
        target_h = height
    elif scale:
        target_w = orig_w * scale
        target_h = orig_h * scale
    else:
        raise ValueError("必须指定 scale, width, 或 height")

    # 确保尺寸为偶数 (h264 要求)
    if isinstance(target_w, int) and target_w > 0:
        target_w = target_w + (target_w % 2)
    if isinstance(target_h, int) and target_h > 0:
        target_h = target_h + (target_h % 2)

    # 输出路径
    OUTPUT_UPSCALE.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        suffix = f"_{scale}x" if scale else f"_{target_w}x{target_h}"
        output_path = OUTPUT_UPSCALE / f"{video_path.stem}{suffix}_ffmpeg.mp4"
    output_path = Path(output_path)

    # scale 滤镜标志
    flags_map = {
        "lanczos": "lanczos",
        "bicubic": "bicubic",
        "bilinear": "bilinear",
        "spline": "spline16",
    }
    sws_flags = flags_map.get(algorithm, "lanczos")

    # 构建命令
    scale_filter = f"scale={target_w}:{target_h}:flags={sws_flags}"
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", scale_filter,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-c:a", "copy",
        str(output_path),
    ]

    actual_target = f"{target_w}x{target_h}" if target_w > 0 and target_h > 0 else f"scale={scale}x"
    logger.info(f"FFmpeg 放大: {video_path.name} ({orig_w}x{orig_h} → {actual_target}, {algorithm})")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    # 获取输出信息
    out_info = get_video_info(output_path)
    upscaled_size = f"{out_info.get('width', '?')}x{out_info.get('height', '?')}"

    logger.info(f"✅ 放大完成: {output_path.name} ({orig_w}x{orig_h} → {upscaled_size})")

    return {
        "output": str(output_path),
        "original_size": f"{orig_w}x{orig_h}",
        "upscaled_size": upscaled_size,
    }


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FFmpeg 视频放大")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出路径")

    size_group = parser.add_mutually_exclusive_group()
    size_group.add_argument(
        "-s", "--scale", type=int,
        help=f"放大倍数 (默认: {UPSCALE_FACTOR})",
    )
    size_group.add_argument("-W", "--width", type=int, help="目标宽度")
    size_group.add_argument("-H", "--height", type=int, help="目标高度")

    parser.add_argument(
        "--algorithm",
        choices=["lanczos", "bicubic", "bilinear", "spline"],
        default="lanczos",
        help="插值算法 (默认: lanczos)",
    )
    parser.add_argument("--crf", type=int, default=18, help="输出质量 CRF (默认: 18)")

    args = parser.parse_args()

    # 默认 2x
    scale = args.scale or (UPSCALE_FACTOR if not args.width and not args.height else None)

    result = upscale_video_ffmpeg(
        video_path=args.input,
        output_path=args.output,
        scale=scale,
        width=args.width,
        height=args.height,
        algorithm=args.algorithm,
        crf=args.crf,
    )
    print(f"\n✅ 放大完成: {result['original_size']} → {result['upscaled_size']}")
    print(f"   输出: {result['output']}")
