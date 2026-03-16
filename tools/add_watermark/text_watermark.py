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
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import (
    ADD_WATERMARK_COLOR,
    ADD_WATERMARK_FONT_SIZE,
    ADD_WATERMARK_MARGIN,
    ADD_WATERMARK_OPACITY,
    ADD_WATERMARK_POSITION,
    ADD_WATERMARK_STROKE_WIDTH,
    IMAGE_EXTENSIONS,
    OUTPUT_ADD_WATERMARK,
    VIDEO_EXTENSIONS,
    clamp,
)
from tools.common import VideoFrameProcessor, calc_overlay_position, generate_output_name, logger

# 项目内置字体目录 (放在 bin/fonts/ 下，随项目分发)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_BUNDLED_FONTS_DIR = _PROJECT_ROOT / "bin" / "fonts"


# ============================================================
# 字体查找
# ============================================================

def _has_smp_chars(text: str) -> bool:
    """
    检测文本是否含有辅助多文种平面字符 (SMP, U+10000+)。
    𝕏𝔼ℝ𝔸 等 Unicode 数学字母数字符号均属于此范围，
    大多数 CJK 字体不包含这些字形。
    """
    return any(ord(c) > 0xFFFF for c in text)


def _find_bundled_unicode_font() -> str:
    """
    查找项目内置的 Unicode 数学字体 (bin/fonts/)。
    优先级: STIXTwoMath > Quivira > 其他 otf/ttf
    """
    preferred = [
        _BUNDLED_FONTS_DIR / "STIXTwoMath-Regular.otf",
        _BUNDLED_FONTS_DIR / "Quivira.ttf",
    ]
    for p in preferred:
        if p.exists():
            return str(p)
    # 扫描 bin/fonts/ 内任何可用字体
    if _BUNDLED_FONTS_DIR.exists():
        for p in _BUNDLED_FONTS_DIR.glob("*.otf"):
            return str(p)
        for p in _BUNDLED_FONTS_DIR.glob("*.ttf"):
            return str(p)
    return ""


def _find_cjk_font() -> str:
    """
    查找系统中可用的中文字体路径。

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


def _get_font(font_path: str | None = None, font_size: int = 36, text: str = "") -> ImageFont.FreeTypeFont:
    """
    获取字体对象。

    选择策略:
      1. 用户显式指定 font_path → 直接使用
      2. 文本含 SMP/数学字符 → 优先使用项目内置 Unicode 数学字体
      3. 否则 → 使用系统 CJK 字体 (支持中文)
      4. 最终回退 → Pillow 默认字体
    """
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, font_size)

    # 含数学/特殊 Unicode 字符 → 使用项目内置字体
    if text and _has_smp_chars(text):
        bundled = _find_bundled_unicode_font()
        if bundled:
            logger.debug(f"检测到 SMP 字符, 使用内置 Unicode 字体: {Path(bundled).name}")
            return ImageFont.truetype(bundled, font_size)
        else:
            logger.warning(
                "文本含 Unicode 数学/特殊字符 (如 𝕏𝔼ℝ𝔸), 但未找到项目内置字体。\n"
                f"请将字体文件放入: {_BUNDLED_FONTS_DIR}\n"
                "推荐字体: STIXTwoMath-Regular.otf (https://github.com/stipub/stixfonts)"
            )

    # 普通文本 → CJK 字体
    cjk_font = _find_cjk_font()
    if cjk_font:
        return ImageFont.truetype(cjk_font, font_size)

    return ImageFont.load_default()





# ============================================================
# 核心渲染
# ============================================================
def _calc_spaced_text_size(draw, text, font, stroke_width, letter_spacing):
    """计算拉开字间距后的文字总尺寸"""
    total_width = 0
    max_height = 0
    for i, char in enumerate(text):
        bbox = draw.textbbox((0, 0), char, font=font, stroke_width=stroke_width)
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1]
        total_width += char_w
        if i < len(text) - 1:
            total_width += letter_spacing
        max_height = max(max_height, char_h)
    return total_width, max_height


def _draw_spaced_text(draw, x, y, text, font, fill, stroke_width, letter_spacing):
    """逐字绘制, 每个字符之间添加额外间距"""
    cursor_x = x
    for char in text:
        bbox = draw.textbbox((0, 0), char, font=font, stroke_width=stroke_width)
        char_w = bbox[2] - bbox[0]
        draw.text((cursor_x, y), char, font=font, fill=fill,
                  stroke_width=stroke_width, stroke_fill=fill)
        cursor_x += char_w + letter_spacing


def _render_text_on_pil_image(
    pil_image: Image.Image,
    text: str,
    font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
    opacity: float,
    position: str | tuple[int, int],
    margin: int,
    stroke_width: int = ADD_WATERMARK_STROKE_WIDTH,
    blend_mode: str = "normal",
    letter_spacing: int = 0,
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

    alpha = int(opacity * 255)

    if letter_spacing > 0:
        # 拉开字间距: 逐字绘制
        text_width, text_height = _calc_spaced_text_size(draw, text, font, stroke_width, letter_spacing)
        x, y = calc_overlay_position(base.size[0], base.size[1], text_width, text_height, position, margin)
        _draw_spaced_text(draw, x, y, text, font, (*color, alpha), stroke_width, letter_spacing)
    else:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x, y = calc_overlay_position(base.size[0], base.size[1], text_width, text_height, position, margin)
        draw.text((x, y), text, font=font, fill=(*color, alpha),
                  stroke_width=stroke_width, stroke_fill=(*color, alpha))

    # 合成
    if blend_mode == "overlay":
        # 叠加 (双重增强): 亮处更亮暗处更暗
        base_arr = np.array(base).astype(float)
        txt_arr = np.array(txt_layer).astype(float)
        txt_alpha = txt_arr[:, :, 3:4] / 255.0
        txt_rgb = txt_arr[:, :, :3]
        base_rgb = base_arr[:, :, :3]

        low = 2 * base_rgb * txt_rgb / 255.0
        high = 255.0 - 2 * (255.0 - base_rgb) * (255.0 - txt_rgb) / 255.0
        mid = np.where(base_rgb <= 128, low, high)
        low2 = 2 * mid * txt_rgb / 255.0
        high2 = 255.0 - 2 * (255.0 - mid) * (255.0 - txt_rgb) / 255.0
        blended = np.where(mid <= 128, low2, high2)

        result_rgb = base_rgb * (1.0 - txt_alpha) + blended * txt_alpha
        result_arr = base_arr.copy()
        result_arr[:, :, :3] = np.clip(result_rgb, 0, 255)
        result = Image.fromarray(result_arr.astype(np.uint8), "RGBA")
    else:
        # normal: 标准透明叠加
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
    stroke_width: int = ADD_WATERMARK_STROKE_WIDTH,
    letter_spacing: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    创建全尺寸的水印层 (用于视频合成优化)
    
    Returns:
        (x, y, w, h, bgr_layer, alpha_channel)
        bgr_layer: (h, w, 3) BGR
        alpha_channel: (h, w, 1) float 0.0~1.0
    """
    # PIL 绘制 RGBA 层
    layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    
    alpha_val = int(opacity * 255)

    if letter_spacing > 0:
        text_width, text_height = _calc_spaced_text_size(draw, text, font, stroke_width, letter_spacing)
        x, y = calc_overlay_position(width, height, text_width, text_height, position, margin)
        _draw_spaced_text(draw, x, y, text, font, (*color, alpha_val), stroke_width, letter_spacing)
    else:
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x, y = calc_overlay_position(width, height, text_width, text_height, position, margin)
        draw.text((x, y), text, font=font, fill=(*color, alpha_val),
                  stroke_width=stroke_width, stroke_fill=(*color, alpha_val))
    
    # 转换为 NumPy 数组
    layer_np = np.array(layer) # RGBA, (H, W, 4)
    
    # 找到有文字内容的区域 (ROI), 避免全图算力浪费
    active_y, active_x = np.where(layer_np[:, :, 3] > 0)
    if len(active_y) == 0:
        return 0, 0, 1, 1, np.zeros((1, 1, 3), dtype=np.uint8), np.zeros((1, 1, 1), dtype=float)
        
    y_min, y_max = active_y.min(), active_y.max() + 1
    x_min, x_max = active_x.min(), active_x.max() + 1
    
    # 裁剪出 ROI
    roi_np = layer_np[y_min:y_max, x_min:x_max]
    
    # 分离通道
    b, g, r = roi_np[:, :, 2], roi_np[:, :, 1], roi_np[:, :, 0]
    a = roi_np[:, :, 3]
    
    bgr = cv2.merge([b, g, r])
    alpha = a.astype(float) / 255.0
    alpha = np.expand_dims(alpha, axis=2) # (H, W, 1)
    
    return x_min, y_min, x_max - x_min, y_max - y_min, bgr, alpha


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
    stroke_width: int = ADD_WATERMARK_STROKE_WIDTH,
    blend_mode: str = "normal",
    letter_spacing: int = 0,
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

    font = _get_font(font_path, font_size, text=text)

    result_image = _render_text_on_pil_image(
        pil_image, text, font, color, opacity, position, margin, stroke_width, blend_mode, letter_spacing,
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
    stroke_width: int = ADD_WATERMARK_STROKE_WIDTH,
    blend_mode: str = "normal",
    letter_spacing: int = 0,
) -> dict:
    """
    为视频添加文字水印
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    font = _get_font(font_path, font_size, text=text)
    opacity = clamp(opacity, 0.0, 1.0, "opacity")
    font_size = max(1, int(font_size))

    OUTPUT_ADD_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_ADD_WATERMARK / generate_output_name(video_path.stem, ".mp4", "_wm")
    output_path = Path(output_path)

    logger.info(f"视频文字水印: {video_path.name}")
    frames_processed = 0

    with VideoFrameProcessor(video_path, output_path) as vp:
        # 预渲染水印层 (性能优化: 提取 ROI)
        x, y, w, h, overlay_bgr, overlay_alpha = _create_watermark_layer(
            vp.width, vp.height, text, font, color, opacity, position, margin, stroke_width, letter_spacing
        )
        
        # 将 overlay 转为 float 放在循环外
        overlay_bgr_float = overlay_bgr.astype(float)
        
        for frame in vp.frames(desc="添加文字水印"):
            roi = frame[y:y+h, x:x+w]
            roi_f = roi.astype(float)

            if blend_mode == "overlay":
                low = 2 * roi_f * overlay_bgr_float / 255.0
                high = 255.0 - 2 * (255.0 - roi_f) * (255.0 - overlay_bgr_float) / 255.0
                mid = np.where(roi_f <= 128, low, high)
                low2 = 2 * mid * overlay_bgr_float / 255.0
                high2 = 255.0 - 2 * (255.0 - mid) * (255.0 - overlay_bgr_float) / 255.0
                blended = np.where(mid <= 128, low2, high2)
                out_roi = roi_f * (1.0 - overlay_alpha) + blended * overlay_alpha
            else:
                # normal
                out_roi = roi_f * (1.0 - overlay_alpha) + overlay_bgr_float * overlay_alpha

            # 放回原图
            frame[y:y+h, x:x+w] = np.clip(out_roi, 0, 255).astype(np.uint8)
            vp.write(frame)
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
