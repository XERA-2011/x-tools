"""
文字水印模块

方法说明:
  - 使用 Pillow 渲染 TrueType 字体, 支持中文文字水印
  - 支持图片和视频
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
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import (
    FFMPEG_BIN, OUTPUT_ADD_WATERMARK, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
    ADD_WATERMARK_FONT_SIZE, ADD_WATERMARK_OPACITY,
    ADD_WATERMARK_COLOR, ADD_WATERMARK_POSITION, ADD_WATERMARK_MARGIN,
)
from tools.common import logger


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

    Args:
        pil_image: 底图 (RGB/RGBA)
        text: 水印文字
        font: 字体对象
        color: 文字颜色 (R, G, B)
        opacity: 透明度 (0.0~1.0)
        position: 位置
        margin: 边距

    Returns:
        PIL.Image: 加了水印的图片
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

    Args:
        image_path: 输入图片路径
        text: 水印文字
        output_path: 输出路径, None 时自动生成
        font_path: 字体文件路径, None 时自动查找
        font_size: 字号
        color: 颜色 (R, G, B)
        opacity: 透明度 (0.0~1.0)
        position: 位置
        margin: 边距

    Returns:
        dict: {"output": str}
    """
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    font = _get_font(font_path, font_size)
    pil_image = Image.open(image_path)

    result_image = _render_text_on_pil_image(
        pil_image, text, font, color, opacity, position, margin,
    )

    # 输出
    OUTPUT_ADD_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_ADD_WATERMARK / f"{image_path.stem}_wm{image_path.suffix}"
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

    Args:
        video_path: 输入视频路径
        text: 水印文字
        其他参数同 add_text_watermark_image

    Returns:
        dict: {"output": str, "frames_processed": int}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    font = _get_font(font_path, font_size)

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 输出路径
    OUTPUT_ADD_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_ADD_WATERMARK / f"{video_path.stem}_wm.mp4"
    output_path = Path(output_path)

    # 临时文件 (无音频)
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    logger.info(f"视频文字水印: {video_path.name} ({total_frames} 帧)")

    from tqdm import tqdm
    frames_done = 0

    for _ in tqdm(range(total_frames), desc="添加文字水印", unit="帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # BGR -> RGB -> PIL
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 渲染水印
        result_pil = _render_text_on_pil_image(
            pil_image, text, font, color, opacity, position, margin,
        )

        # PIL -> RGB -> BGR -> cv2
        result_frame = cv2.cvtColor(np.array(result_pil.convert("RGB")), cv2.COLOR_RGB2BGR)
        writer.write(result_frame)
        frames_done += 1

    cap.release()
    writer.release()

    # 合并音频
    _merge_audio(video_path, temp_video_path, output_path)
    Path(temp_video_path).unlink(missing_ok=True)

    logger.info(f"✅ 视频文字水印完成: {output_path.name} ({frames_done} 帧)")
    return {"output": str(output_path), "frames_processed": frames_done}


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

    Args:
        input_path: 输入文件路径
        text: 水印文字
        **kwargs: 传递给具体处理函数的参数

    Returns:
        dict: 处理结果
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
# 音频合并 (复用去水印模块的模式)
# ============================================================
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
        logger.warning("音频混合失败, 仅输出视频")
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

    args = parser.parse_args()

    result = add_text_watermark(
        input_path=args.input,
        text=args.text,
        output_path=args.output,
        font_path=args.font,
        font_size=args.font_size,
        opacity=args.opacity,
        position=args.position,
        margin=args.margin,
    )
    print(f"\n✅ 水印添加完成: {result['output']}")
