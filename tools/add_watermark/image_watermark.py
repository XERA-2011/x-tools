"""
图片 (Logo) 水印模块

方法说明:
  - 将 PNG/带透明通道的 Logo 叠加到图片或视频上
  - 视频处理优化: 预渲染 Logo 层 + NumPy Alpha Blending
  - 图片处理: 使用 Pillow 原始合成
  - 支持按比例缩放 Logo 大小、透明度、位置

使用方式:
  # 图片加 Logo 水印
  python tools/add_watermark/image_watermark.py image.jpg -w logo.png

  # 视频加 Logo 水印
  python tools/add_watermark/image_watermark.py video.mp4 -w logo.png --scale 0.2

依赖:
  pip install Pillow
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


from config import (
    OUTPUT_ADD_WATERMARK, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
    ADD_WATERMARK_OPACITY, ADD_WATERMARK_POSITION,
    ADD_WATERMARK_MARGIN, ADD_WATERMARK_LOGO_SCALE,
    clamp,
)
from tools.common import logger, VideoFrameProcessor, generate_output_name, calc_overlay_position


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
    """
    logo = Image.open(logo_path).convert("RGBA")

    # 按比例缩放
    target_width = int(base_width * scale)
    ratio = target_width / logo.width
    target_height = int(logo.height * ratio)
    logo = logo.resize((target_width, target_height), Image.Resampling.LANCZOS)

    return logo





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
    在 PIL Image 上叠加 Logo 水印 (用于单张图片)
    """
    base = pil_image.convert("RGBA")

    # 调整 Logo 透明度
    if opacity < 1.0:
        r, g, b, a = logo.split()
        a = a.point(lambda x: int(x * opacity))
        logo = Image.merge("RGBA", (r, g, b, a))

    # 计算位置
    x, y = calc_overlay_position(
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


def _create_logo_layer(
    width: int,
    height: int,
    watermark_path: str | Path,
    scale: float,
    opacity: float,
    position: str | tuple[int, int],
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    创建全尺寸 Logo 层 (用于视频加速)
    Returns: (bgr, alpha)
    """
    # 1. 准备 Logo (缩放)
    logo = _prepare_logo(watermark_path, width, scale)
    
    # 2. 调整透明度
    if opacity < 1.0:
        r, g, b, a = logo.split()
        a = a.point(lambda x: int(x * opacity))
        logo = Image.merge("RGBA", (r, g, b, a))
        
    # 3. 创建层
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    
    # 4. 计算位置并粘贴
    x, y = calc_overlay_position(
        width, height,
        logo.width, logo.height,
        position, margin,
    )
    layer.paste(logo, (x, y))
    
    # 5. 转 NumPy
    layer_np = np.array(layer)
    b, g, r = layer_np[:, :, 2], layer_np[:, :, 1], layer_np[:, :, 0]
    a = layer_np[:, :, 3]
    
    bgr = cv2.merge([b, g, r])
    alpha = a.astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=2)
    
    return bgr, alpha


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
        output_path = OUTPUT_ADD_WATERMARK / generate_output_name(image_path.stem, image_path.suffix, "_wm")
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
    """
    video_path = Path(video_path)
    watermark_path = Path(watermark_path)

    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if not watermark_path.is_file():
        raise FileNotFoundError(f"水印图片不存在: {watermark_path}")

    opacity = clamp(opacity, 0.0, 1.0, "opacity")
    scale = clamp(scale, 0.01, 1.0, "scale")

    OUTPUT_ADD_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_ADD_WATERMARK / generate_output_name(video_path.stem, ".mp4", "_wm")
    output_path = Path(output_path)

    logger.info(f"视频 Logo 水印: {video_path.name}")
    frames_processed = 0

    with VideoFrameProcessor(video_path, output_path) as vp:
        # 预渲染
        overlay_bgr, overlay_alpha = _create_logo_layer(
            vp.width, vp.height, watermark_path, scale, opacity, position, margin
        )
        
        for frame in vp.frames(desc="添加 Logo 水印"):
            frame_float = frame.astype(float)
            out = frame_float * (1.0 - overlay_alpha) + overlay_bgr.astype(float) * overlay_alpha
            vp.write(out.astype(np.uint8))
            frames_processed += 1

    logger.info(f"✅ 视频 Logo 水印完成: {output_path.name} ({frames_processed} 帧)")
    return {"output": str(output_path), "frames_processed": frames_processed}


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
