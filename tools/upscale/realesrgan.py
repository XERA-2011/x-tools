"""
Real-ESRGAN 视频超分辨率模块

方法说明:
  - 使用 Real-ESRGAN 模型对视频逐帧进行超分辨率放大
  - 支持 2x / 4x 放大, MPS 加速 (Mac M4)
  - 适合: 低分辨率视频清晰化 (如 480p → 1080p, 720p → 4K)

依赖:
  pip install realesrgan torch torchvision

使用方式:
  python tools/upscale/realesrgan.py video.mp4
  python tools/upscale/realesrgan.py video.mp4 --scale 4
"""
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


from config import OUTPUT_UPSCALE, UPSCALE_FACTOR, FFMPEG_BIN
from tools.common import logger, VideoFrameProcessor, generate_output_name, get_video_info


def _check_realesrgan():
    """检查 Real-ESRGAN 是否可用"""
    try:
        from realesrgan import RealESRGANer  # noqa: F401
        return True
    except ImportError:
        return False


def _get_device():
    """获取最佳推理设备"""
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def _init_upsampler(scale: int = 2, device: str | None = None, model_name: str = "RealESRGAN_x2plus"):
    """
    初始化 Real-ESRGAN upsampler

    Args:
        scale: 放大倍数 (2 or 4)
        device: 推理设备 ("mps" / "cuda" / "cpu"), None 自动选择
        model_name: 模型名称
    """
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        raise RuntimeError(
            "❌ 未安装 Real-ESRGAN，请运行:\n"
            "   pip install realesrgan torch torchvision basicsr\n"
            "   或使用 FFmpeg 方案: python tools/upscale/ffmpeg_scale.py"
        )

    if device is None:
        device = _get_device()

    # 根据放大倍数选择模型
    if scale == 4:
        model_name = "RealESRGAN_x4plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4
    else:
        model_name = "RealESRGAN_x2plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2

    # 模型权重会自动下载到 ~/.cache
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=None,  # 自动下载
        model=model,
        tile=0,           # 0=不分块 (MPS 上分块可能有问题)
        tile_pad=10,
        pre_pad=0,
        half=False,       # MPS 不支持 half precision
        device=device,
    )

    logger.info(f"Real-ESRGAN 模型已加载 ({model_name}, 设备: {device})")
    return upsampler


def _auto_select_scale(src_width: int, src_height: int, target_width: int, target_height: int) -> int:
    """
    根据源分辨率和目标分辨率, 自动选择最小 Real-ESRGAN 倍数 (2 或 4)
    选择原则: 超分后的分辨率必须 >= 目标分辨率
    """
    ratio_w = target_width / src_width
    ratio_h = target_height / src_height
    ratio = max(ratio_w, ratio_h)

    if ratio <= 2.0:
        return 2
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


def upscale_video_realesrgan(
    video_path: str | Path,
    output_path: str | Path | None = None,
    scale: int | None = None,
    device: str | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
) -> dict:
    """
    使用 Real-ESRGAN 对视频超分辨率放大

    Args:
        video_path: 输入视频路径
        output_path: 输出路径
        scale: 放大倍数 (2 or 4), 与 target_width/target_height 二选一
        device: 推理设备, None 自动选择
        target_width: 目标宽度 (与 scale 二选一, 自动选择最佳倍数)
        target_height: 目标高度 (与 scale 二选一)

    Returns:
        dict: {"output": str, "frames_processed": int,
               "original_size": str, "upscaled_size": str}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 获取源视频信息用于自动选择倍数
    info = get_video_info(video_path)
    src_width = info.get("width", 0)
    src_height = info.get("height", 0)

    # 判断模式: 目标分辨率 vs 倍数放大
    need_rescale = False
    if target_width and target_height:
        # 目标分辨率模式: 自动选择最小倍数
        scale = _auto_select_scale(src_width, src_height, target_width, target_height)
        upscaled_w = src_width * scale
        upscaled_h = src_height * scale
        # 只有当超分结果与目标不一致时才需要后缩放
        need_rescale = (upscaled_w != target_width or upscaled_h != target_height)
        logger.info(f"目标分辨率: {target_width}x{target_height}, 自动选择 {scale}x 超分")
    elif scale is None:
        scale = UPSCALE_FACTOR

    if scale not in (2, 4):
        raise ValueError("scale 仅支持 2 或 4")

    # 初始化模型
    upsampler = _init_upsampler(scale, device)

    OUTPUT_UPSCALE.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        if target_width and target_height:
            tag = f"_{target_width}x{target_height}"
        else:
            tag = f"_{scale}x"
        output_path = OUTPUT_UPSCALE / generate_output_name(video_path.stem, ".mp4", tag)
    output_path = Path(output_path)

    # 如果需要后缩放, 先输出到临时路径
    actual_output = output_path
    if need_rescale:
        ai_output = output_path.with_name(f".tmp_ai_{output_path.name}")
    else:
        ai_output = output_path

    logger.info(f"超分处理: {video_path.name} (scale={scale}x)")

    frames_processed = 0
    original_size = f"{src_width}x{src_height}"

    with VideoFrameProcessor(video_path, ai_output) as vp:
        new_width = vp.width * scale
        new_height = vp.height * scale

        # 初始化 writer
        vp.init_writer(width=new_width, height=new_height)
        
        desc = f"Real-ESRGAN {scale}x"
        for frame in vp.frames(desc=desc):
            # Real-ESRGAN 推理
            try:
                output, _ = upsampler.enhance(frame, outscale=scale)
                vp.write(output)
            except Exception as e:
                # 某些帧可能出错, 用 OpenCV resize 作为 fallback
                logger.warning(f"帧 {frames_processed} 推理失败, 使用 bicubic 回退: {e}")
                fallback = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                vp.write(fallback)
            
            frames_processed += 1

    # 后缩放: AI 超分结果 → 精确目标分辨率
    if need_rescale:
        logger.info(f"精确缩放: {new_width}x{new_height} → {target_width}x{target_height}")
        _ffmpeg_rescale(ai_output, actual_output, target_width, target_height)
        ai_output.unlink(missing_ok=True)  # 清理临时文件
        upscaled_size = f"{target_width}x{target_height}"
    else:
        upscaled_size = f"{new_width}x{new_height}"

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
        type=int, choices=[2, 4], default=UPSCALE_FACTOR,
        help=f"放大倍数 (默认: {UPSCALE_FACTOR})",
    )
    parser.add_argument(
        "--device",
        choices=["mps", "cpu", "cuda"],
        help="推理设备 (默认: 自动选择)",
    )

    args = parser.parse_args()

    if not _check_realesrgan():
        print("❌ 未安装 Real-ESRGAN，请运行: pip install realesrgan torch torchvision basicsr")
        sys.exit(1)

    result = upscale_video_realesrgan(
        video_path=args.input,
        output_path=args.output,
        scale=args.scale,
        device=args.device,
    )
    print(f"\n✅ 超分完成: {result['original_size']} → {result['upscaled_size']}")
    print(f"   输出: {result['output']}")
