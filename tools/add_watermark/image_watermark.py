"""
图片 (Logo) 水印模块

方法说明:
  - 将 PNG/带透明通道的 Logo 叠加到图片或视频上
  - 支持按比例缩放 Logo 大小
  - 支持透明度和位置配置

使用方式:
  # 图片加 Logo 水印
  python tools/add_watermark/image_watermark.py image.jpg -w logo.png

  # 视频加 Logo 水印
  python tools/add_watermark/image_watermark.py video.mp4 -w logo.png --scale 0.2

依赖:
  pip install Pillow
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    FFMPEG_BIN, OUTPUT_ADD_WATERMARK, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
    ADD_WATERMARK_OPACITY, ADD_WATERMARK_POSITION,
    ADD_WATERMARK_MARGIN, ADD_WATERMARK_LOGO_SCALE,
    clamp,
)
from tools.common import logger, merge_audio


# ============================================================
# Logo 预处理
# ============================================================
def _prepare_logo(
    logo_path: str | Path,
    base_width: int,
    scale: float,
) -> Image.Image:
    """
    加载并缩放 Logo

    Args:
        logo_path: Logo 图片路径 (推荐 PNG 带透明通道)
        base_width: 底图宽度
        scale: Logo 占底图宽度的比例 (0.0~1.0)

    Returns:
        PIL.Image (RGBA): 缩放后的 Logo
    """
    logo = Image.open(logo_path).convert("RGBA")

    # 按比例缩放
    target_width = int(base_width * scale)
    ratio = target_width / logo.width
    target_height = int(logo.height * ratio)
    logo = logo.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return logo


def _calc_logo_position(
    base_width: int,
    base_height: int,
    logo_width: int,
    logo_height: int,
    position: str | tuple[int, int],
    margin: int,
) -> tuple[int, int]:
    """计算 Logo 位置"""
    if isinstance(position, (list, tuple)):
        return int(position[0]), int(position[1])

    positions = {
        "bottom-right": (base_width - logo_width - margin, base_height - logo_height - margin),
        "bottom-left": (margin, base_height - logo_height - margin),
        "top-right": (base_width - logo_width - margin, margin),
        "top-left": (margin, margin),
        "center": ((base_width - logo_width) // 2, (base_height - logo_height) // 2),
    }
    return positions.get(position, positions["bottom-right"])


# ============================================================
# 核心渲染
# ============================================================
def _render_logo_on_pil_image(
    pil_image: Image.Image,
    logo: Image.Image,
    opacity: float,
    position: str | tuple[int, int],
    margin: int,
) -> Image.Image:
    """
    在 PIL Image 上叠加 Logo 水印

    Args:
        pil_image: 底图
        logo: Logo (RGBA)
        opacity: 透明度 (0.0~1.0)
        position: 位置
        margin: 边距

    Returns:
        PIL.Image: 加了 Logo 的图片
    """
    base = pil_image.convert("RGBA")

    # 调整 Logo 透明度
    if opacity < 1.0:
        # 修改 Logo alpha 通道
        r, g, b, a = logo.split()
        a = a.point(lambda x: int(x * opacity))
        logo = Image.merge("RGBA", (r, g, b, a))

    # 计算位置
    x, y = _calc_logo_position(
        base.width, base.height,
        logo.width, logo.height,
        position, margin,
    )

    # 创建与底图同尺寸的透明层, 将 Logo 粘贴到对应位置
    logo_layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    logo_layer.paste(logo, (x, y))

    # 合成
    result = Image.alpha_composite(base, logo_layer)
    return result


# ============================================================
# 图片加 Logo 水印
# ============================================================
def add_image_watermark_image(
    image_path: str | Path,
    watermark_path: str | Path,
    output_path: str | Path | None = None,
    scale: float = ADD_WATERMARK_LOGO_SCALE,
    opacity: float = ADD_WATERMARK_OPACITY,
    position: str | tuple[int, int] = ADD_WATERMARK_POSITION,
    margin: int = ADD_WATERMARK_MARGIN,
) -> dict:
    """
    为图片添加 Logo 水印

    Args:
        image_path: 输入图片路径
        watermark_path: Logo 图片路径 (推荐 PNG)
        output_path: 输出路径
        scale: Logo 大小比例 (0.0~1.0)
        opacity: 透明度
        position: 位置
        margin: 边距

    Returns:
        dict: {"output": str}
    """
    image_path = Path(image_path)
    watermark_path = Path(watermark_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    if not watermark_path.is_file():
        raise FileNotFoundError(f"水印图片不存在: {watermark_path}")

    pil_image = Image.open(image_path)

    # 参数验证
    opacity = clamp(opacity, 0.0, 1.0, "opacity")
    scale = clamp(scale, 0.01, 1.0, "scale")

    logo = _prepare_logo(watermark_path, pil_image.width, scale)

    result_image = _render_logo_on_pil_image(pil_image, logo, opacity, position, margin)

    # 输出
    OUTPUT_ADD_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_ADD_WATERMARK / f"{image_path.stem}_wm{image_path.suffix}"
    output_path = Path(output_path)

    if output_path.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
        result_image = result_image.convert("RGB")
    result_image.save(output_path, quality=95)

    logger.info(f"✅ 图片 Logo 水印完成: {output_path.name}")
    return {"output": str(output_path)}


# ============================================================
# 视频加 Logo 水印
# ============================================================
def add_image_watermark_video(
    video_path: str | Path,
    watermark_path: str | Path,
    output_path: str | Path | None = None,
    scale: float = ADD_WATERMARK_LOGO_SCALE,
    opacity: float = ADD_WATERMARK_OPACITY,
    position: str | tuple[int, int] = ADD_WATERMARK_POSITION,
    margin: int = ADD_WATERMARK_MARGIN,
) -> dict:
    """
    为视频添加 Logo 水印

    Args:
        video_path: 输入视频路径
        watermark_path: Logo 图片路径
        其他参数同 add_image_watermark_image

    Returns:
        dict: {"output": str, "frames_processed": int}
    """
    video_path = Path(video_path)
    watermark_path = Path(watermark_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not watermark_path.is_file():
        raise FileNotFoundError(f"水印图片不存在: {watermark_path}")

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 预处理 Logo (只需做一次)
    # 参数验证
    opacity = clamp(opacity, 0.0, 1.0, "opacity")
    scale = clamp(scale, 0.01, 1.0, "scale")

    logo = _prepare_logo(watermark_path, width, scale)

    # 输出路径
    OUTPUT_ADD_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_ADD_WATERMARK / f"{video_path.stem}_wm.mp4"
    output_path = Path(output_path)

    # 临时文件
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    logger.info(f"视频 Logo 水印: {video_path.name} ({total_frames} 帧)")

    from tqdm import tqdm
    frames_done = 0

    for _ in tqdm(range(total_frames), desc="添加 Logo 水印", unit="帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB -> PIL
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        result_pil = _render_logo_on_pil_image(pil_image, logo, opacity, position, margin)

        # PIL -> BGR
        result_frame = cv2.cvtColor(np.array(result_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(result_frame)
        frames_done += 1

    cap.release()
    writer.release()

    # 合并音频
    merge_audio(video_path, temp_video_path, output_path)
    Path(temp_video_path).unlink(missing_ok=True)

    logger.info(f"✅ 视频 Logo 水印完成: {output_path.name} ({frames_done} 帧)")
    return {"output": str(output_path), "frames_processed": frames_done}


# ============================================================
# 统一入口
# ============================================================
def add_image_watermark(
    input_path: str | Path,
    watermark_path: str | Path,
    **kwargs,
) -> dict:
    """
    自动判断输入类型 (图片/视频), 添加 Logo 水印
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return add_image_watermark_image(input_path, watermark_path, **kwargs)
    elif suffix in VIDEO_EXTENSIONS:
        return add_image_watermark_video(input_path, watermark_path, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")




# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="图片 (Logo) 水印")
    parser.add_argument("input", help="输入文件路径 (图片或视频)")
    parser.add_argument("-w", "--watermark", required=True, help="水印 Logo 图片路径 (推荐 PNG)")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument("--scale", type=float, default=ADD_WATERMARK_LOGO_SCALE, help=f"Logo 大小比例 (默认: {ADD_WATERMARK_LOGO_SCALE})")
    parser.add_argument("--opacity", type=float, default=ADD_WATERMARK_OPACITY, help=f"透明度 0.0~1.0 (默认: {ADD_WATERMARK_OPACITY})")
    parser.add_argument(
        "--position",
        default=ADD_WATERMARK_POSITION,
        help=f"位置: bottom-right/bottom-left/top-right/top-left/center (默认: {ADD_WATERMARK_POSITION})",
    )
    parser.add_argument("--margin", type=int, default=ADD_WATERMARK_MARGIN, help=f"边距像素 (默认: {ADD_WATERMARK_MARGIN})")

    args = parser.parse_args()

    result = add_image_watermark(
        input_path=args.input,
        watermark_path=args.watermark,
        output_path=args.output,
        scale=args.scale,
        opacity=args.opacity,
        position=args.position,
        margin=args.margin,
    )
    print(f"\n✅ Logo 水印添加完成: {result['output']}")
