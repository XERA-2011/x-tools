"""
文字水印模块

方法说明:
  - 使用 Pillow 渲染 TrueType 字体, 支持中文
  - 视频处理优化: 预渲染水印层 + NumPy Alpha Blending (速度提升显著)
  - 图片处理: 使用 Pillow 原始合成
  - 可配置: 位置、字号、颜色、透明度

使用方式:
  # 图片加水印
  python tools/add_watermark/text_watermark.py image.jpg -t "© 2026 版权所有"

  # 视频加水印
  python tools/add_watermark/text_watermark.py video.mp4 -t "Sample Watermark"

依赖:
  pip install Pillow
"""
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


from config import (
    OUTPUT_ADD_WATERMARK, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
    ADD_WATERMARK_FONT_SIZE, ADD_WATERMARK_OPACITY,
    ADD_WATERMARK_COLOR, ADD_WATERMARK_POSITION, ADD_WATERMARK_MARGIN,
    clamp,
)
from tools.common import logger, VideoFrameProcessor, generate_output_name


# ============================================================
# 字体查找
# ============================================================
def _find_cjk_font() -> str:
    """
    查找系统中可用的中文字体路径

    Returns:
        str: 字体文件路径
    """
    system = platform.system()

    candidates = []
    if system == "Darwin":  # macOS
        candidates = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
        ]
    elif system == "Linux":
        candidates = [
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        ]
    elif system == "Windows":
        candidates = [
            "C:/Windows/Fonts/msyh.ttc",      # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",     # 黑体
            "C:/Windows/Fonts/simsun.ttc",     # 宋体
        ]

    for font_path in candidates:
        if Path(font_path).exists():
            return font_path

    # 回退: 使用 Pillow 默认字体 (不支持中文, 但不会崩溃)
    logger.warning("未找到中文字体, 将使用默认字体 (中文可能显示为方块)")
    return ""


def _get_font(font_path: str | None = None, font_size: int = 36) -> ImageFont.FreeTypeFont:
    """获取字体对象"""
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, font_size)

    cjk_font = _find_cjk_font()
    if cjk_font:
        return ImageFont.truetype(cjk_font, font_size)

    # 最终回退
    return ImageFont.load_default()


# ============================================================
# 位置计算
# ============================================================
def _calc_position(
    base_width: int,
    base_height: int,
    text_width: int,
    text_height: int,
    position: str | tuple[int, int],
    margin: int,
) -> tuple[int, int]:
    """
    根据位置名称计算水印坐标

    Args:
        base_width, base_height: 底图尺寸
        text_width, text_height: 水印文字尺寸
        position: 位置 ("bottom-right" / "top-left" / ... / (x, y))
        margin: 边距

    Returns:
        (x, y) 坐标
    """
    if isinstance(position, (list, tuple)):
        return int(position[0]), int(position[1])

    positions = {
        "bottom-right": (base_width - text_width - margin, base_height - text_height - margin),
        "bottom-left": (margin, base_height - text_height - margin),
        "top-right": (base_width - text_width - margin, margin),
        "top-left": (margin, margin),
        "center": ((base_width - text_width) // 2, (base_height - text_height) // 2),
    }
    return positions.get(position, positions["bottom-right"])


# ============================================================
# 核心渲染
# ============================================================
def _render_text_on_pil_image(
    pil_image: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    opacity: float,
    position: str | tuple[int, int],
    margin: int,
) -> Image.Image:
    """
    在 PIL Image 上渲染文字水印 (带透明度)
    用于处理单张图片
    """
    # 确保底图是 RGBA
    base = pil_image.convert("RGBA")

    # 创建透明文字层
    txt_layer = Image.new("RGBA", base.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)

    # 测量文字尺寸
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 计算位置
    x, y = _calc_position(base.size[0], base.size[1], text_width, text_height, position, margin)

    # 绘制带透明度的文字
    alpha = int(opacity * 255)
    draw.text((x, y), text, font=font, fill=(*color, alpha))

    # 合成
    result = Image.alpha_composite(base, txt_layer)
    return result


def _create_watermark_layer(
    width: int,
    height: int,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    opacity: float,
    position: str | tuple[int, int],
    margin: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    创建全尺寸的水印层 (用于视频合成优化)
    
    Returns:
        (bgr_layer, alpha_channel)
        bgr_layer: (H, W, 3) BGR
        alpha_channel: (H, W, 1) float 0.0~1.0
    """
    # PIL 绘制 RGBA 层
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x, y = _calc_position(width, height, text_width, text_height, position, margin)
    
    alpha_val = int(opacity * 255)
    draw.text((x, y), text, font=font, fill=(*color, alpha_val))
    
    # 转换为 NumPy 数组
    layer_np = np.array(layer) # RGBA, (H, W, 4)
    
    # 分离通道
    # PIL RGB -> OpenCV BGR
    # layer_np is R G B A
    # OpenCV expects B G R
    # alpha is A
    
    b, g, r = layer_np[:, :, 2], layer_np[:, :, 1], layer_np[:, :, 0]
    a = layer_np[:, :, 3]
    
    bgr = cv2.merge([b, g, r])
    alpha = a.astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=2) # (H, W, 1)
    
    return bgr, alpha


# ============================================================
# 图片加文字水印
# ============================================================
def add_text_watermark_image(
    image_path: str | Path,
    text: str,
    output_path: str | Path | None = None,
    font_path: str | None = None,
    font_size: int = ADD_WATERMARK_FONT_SIZE,
    color: tuple[int, int, int] = ADD_WATERMARK_COLOR,
    opacity: float = ADD_WATERMARK_OPACITY,
    position: str | tuple[int, int] = ADD_WATERMARK_POSITION,
    margin: int = ADD_WATERMARK_MARGIN,
) -> dict:
    """
    为图片添加文字水印
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    pil_image = Image.open(image_path)

    # 参数验证
    opacity = clamp(opacity, 0.0, 1.0, "opacity")
    font_size = max(1, int(font_size))

    font = _get_font(font_path, font_size)

    result_image = _render_text_on_pil_image(
        pil_image, text, font, color, opacity, position, margin,
    )

    # 输出
    OUTPUT_ADD_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_ADD_WATERMARK / generate_output_name(image_path.stem, image_path.suffix, "_wm")
    output_path = Path(output_path)

    # 保存 (转回 RGB, 因为 JPEG 不支持 RGBA)
    if output_path.suffix.lower() in (".jpg", ".jpeg", ".bmp"):
        result_image = result_image.convert("RGB")
    result_image.save(output_path, quality=95)

    logger.info(f"✅ 图片文字水印完成: {output_path.name}")
    return {"output": str(output_path)}


# ============================================================
# 视频加文字水印
# ============================================================
def add_text_watermark_video(
    video_path: str | Path,
    text: str,
    output_path: str | Path | None = None,
    font_path: str | None = None,
    font_size: int = ADD_WATERMARK_FONT_SIZE,
    color: tuple[int, int, int] = ADD_WATERMARK_COLOR,
    opacity: float = ADD_WATERMARK_OPACITY,
    position: str | tuple[int, int] = ADD_WATERMARK_POSITION,
    margin: int = ADD_WATERMARK_MARGIN,
) -> dict:
    """
    为视频添加文字水印
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    font = _get_font(font_path, font_size)
    opacity = clamp(opacity, 0.0, 1.0, "opacity")
    font_size = max(1, int(font_size))

    OUTPUT_ADD_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_ADD_WATERMARK / generate_output_name(video_path.stem, ".mp4", "_wm")
    output_path = Path(output_path)

    logger.info(f"视频文字水印: {video_path.name}")
    frames_processed = 0

    with VideoFrameProcessor(video_path, output_path) as vp:
        # 预渲染水印层 (性能优化)
        overlay_bgr, overlay_alpha = _create_watermark_layer(
            vp.width, vp.height, text, font, color, opacity, position, margin
        )
        
        # 优化: 只有当 alpha > 0 的地方才需要计算
        # 如果水印很小, 全图 blend 浪费算力
        # 但 NumPy 全图操作非常快 (vectorized), 相比 masking 复杂度可能更优
        # 且 overlay_alpha 大部分是 0. 
        # out = src * 1 + 0.
        
        # 注意: cv2 读出的 frame 是 uint8
        # 转换 float 计算避免溢出
        
        for frame in vp.frames(desc="添加文字水印"):
            frame_float = frame.astype(float)
            
            # Blend: src * (1 - alpha) + overlay * alpha
            out = frame_float * (1.0 - overlay_alpha) + overlay_bgr.astype(float) * overlay_alpha
            
            vp.write(out.astype(np.uint8))
            frames_processed += 1

    logger.info(f"✅ 视频文字水印完成: {output_path.name} ({frames_processed} 帧)")
    return {"output": str(output_path), "frames_processed": frames_processed}


# ============================================================
# 统一入口
# ============================================================
def add_text_watermark(
    input_path: str | Path,
    text: str,
    **kwargs,
) -> dict:
    """
    自动判断输入类型 (图片/视频), 添加文字水印
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return add_text_watermark_image(input_path, text, **kwargs)
    elif suffix in VIDEO_EXTENSIONS:
        return add_text_watermark_video(input_path, text, **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文字水印 (支持中文)")
    parser.add_argument("input", help="输入文件路径 (图片或视频)")
    parser.add_argument("-t", "--text", required=True, help="水印文字")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument("--font", help="字体文件路径 (.ttf/.ttc)")
    parser.add_argument("--font-size", type=int, default=ADD_WATERMARK_FONT_SIZE, help=f"字号 (默认: {ADD_WATERMARK_FONT_SIZE})")
    parser.add_argument("--opacity", type=float, default=ADD_WATERMARK_OPACITY, help=f"透明度 0.0~1.0 (默认: {ADD_WATERMARK_OPACITY})")
    parser.add_argument(
        "--position",
        default=ADD_WATERMARK_POSITION,
        help=f"位置: bottom-right/bottom-left/top-right/top-left/center (默认: {ADD_WATERMARK_POSITION})",
    )
    parser.add_argument("--margin", type=int, default=ADD_WATERMARK_MARGIN, help=f"边距像素 (默认: {ADD_WATERMARK_MARGIN})")

    # 颜色参数需要解析 hex 或 r,g,b. Config 默认是 tuple. CLI 暂时默认 white.
    # 此处简化 CLI 颜色支持
    parser.add_argument("--color", default="255,255,255", help="颜色 R,G,B (默认: 255,255,255)")

    args = parser.parse_args()

    color_tuple = tuple(map(int, args.color.split(",")))
    if len(color_tuple) != 3:
        parser.error("Color must be R,G,B")

    result = add_text_watermark(
        input_path=args.input,
        text=args.text,
        output_path=args.output,
        font_path=args.font,
        font_size=args.font_size,
        opacity=args.opacity,
        position=args.position,
        margin=args.margin,
        color=color_tuple,
    )
    print(f"\n✅ 水印添加完成: {result['output']}")
