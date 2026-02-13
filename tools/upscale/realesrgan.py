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
import tempfile
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FFMPEG_BIN, OUTPUT_UPSCALE, UPSCALE_FACTOR
from tools.common import get_video_info, logger


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


def upscale_video_realesrgan(
    video_path: str | Path,
    output_path: str | Path | None = None,
    scale: int = UPSCALE_FACTOR,
    device: str | None = None,
) -> dict:
    """
    使用 Real-ESRGAN 对视频超分辨率放大

    Args:
        video_path: 输入视频路径
        output_path: 输出路径
        scale: 放大倍数 (2 or 4)
        device: 推理设备, None 自动选择

    Returns:
        dict: {"output": str, "frames_processed": int,
               "original_size": str, "upscaled_size": str}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if scale not in (2, 4):
        raise ValueError("scale 仅支持 2 或 4")

    # 初始化模型
    upsampler = _init_upsampler(scale, device)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    new_width = width * scale
    new_height = height * scale

    # 输出路径
    OUTPUT_UPSCALE.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_UPSCALE / f"{video_path.stem}_{scale}x.mp4"
    output_path = Path(output_path)

    # 临时文件 (无音频)
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (new_width, new_height))

    logger.info(
        f"超分处理: {video_path.name} ({width}x{height} → {new_width}x{new_height}, "
        f"{total_frames} 帧, {scale}x)"
    )

    from tqdm import tqdm
    frames_done = 0

    for _ in tqdm(range(total_frames), desc=f"Real-ESRGAN {scale}x", unit="帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # Real-ESRGAN 推理
        try:
            output, _ = upsampler.enhance(frame, outscale=scale)
            writer.write(output)
        except Exception as e:
            # 某些帧可能出错, 用 OpenCV resize 作为 fallback
            logger.warning(f"帧 {frames_done} 推理失败, 使用 bicubic 回退: {e}")
            fallback = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            writer.write(fallback)

        frames_done += 1

    cap.release()
    writer.release()

    # 合并音频
    _merge_audio(video_path, temp_video_path, output_path)
    Path(temp_video_path).unlink(missing_ok=True)

    original_size = f"{width}x{height}"
    upscaled_size = f"{new_width}x{new_height}"
    logger.info(f"✅ 超分完成: {output_path.name} ({original_size} → {upscaled_size})")

    return {
        "output": str(output_path),
        "frames_processed": frames_done,
        "original_size": original_size,
        "upscaled_size": upscaled_size,
    }


def _merge_audio(original_video: Path, processed_video: str, output_path: Path):
    """将原视频音频合并到处理后的视频"""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(processed_video),
        "-i", str(original_video),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        cmd_fallback = [
            FFMPEG_BIN, "-y",
            "-i", str(processed_video),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            "-an",
            str(output_path),
        ]
        subprocess.run(cmd_fallback, capture_output=True, text=True, check=True)


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
