"""
批量加水印入口

支持:
  - 批量文字水印 (图片 + 视频)
  - 批量 Logo 水印 (图片 + 视频)
"""
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import INPUT_DIR, ensure_dirs
from tools.common import scan_media, batch_process, print_summary, logger
from tools.add_watermark.text_watermark import add_text_watermark
from tools.add_watermark.image_watermark import add_image_watermark


def _batch_text_worker(file_path: Path, **kwargs) -> dict:
    """批量文字水印 worker"""
    return add_text_watermark(file_path, **kwargs)


def _batch_image_worker(file_path: Path, **kwargs) -> dict:
    """批量 Logo 水印 worker"""
    return add_image_watermark(file_path, **kwargs)


def batch_add_text_watermark(
    input_dir: str | Path | None = None,
    files: list[Path] | None = None,
    text: str = "",
    **kwargs,
) -> list[dict]:
    """
    批量添加文字水印

    Args:
        input_dir: 输入目录
        files: 文件列表 (优先使用)
        text: 水印文字
        **kwargs: 传递给 add_text_watermark 的参数
    """
    ensure_dirs()
    if files is None:
        videos, images = scan_media(input_dir or INPUT_DIR)
        files = images + videos

    results = batch_process(
        files,
        _batch_text_worker,
        desc="批量添加文字水印",
        text=text,
        **kwargs,
    )
    print_summary(results)
    return results


def batch_add_image_watermark(
    input_dir: str | Path | None = None,
    files: list[Path] | None = None,
    watermark_path: str | Path = "",
    **kwargs,
) -> list[dict]:
    """
    批量添加 Logo 水印

    Args:
        input_dir: 输入目录
        files: 文件列表 (优先使用)
        watermark_path: Logo 图片路径
        **kwargs: 传递给 add_image_watermark 的参数
    """
    ensure_dirs()
    if files is None:
        videos, images = scan_media(input_dir or INPUT_DIR)
        files = images + videos

    results = batch_process(
        files,
        _batch_image_worker,
        desc="批量添加 Logo 水印",
        watermark_path=watermark_path,
        **kwargs,
    )
    print_summary(results)
    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量加水印")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入目录")

    sub = parser.add_subparsers(dest="type", required=True)

    # 文字水印
    text_p = sub.add_parser("text", help="文字水印")
    text_p.add_argument("-t", "--text", required=True, help="水印文字")
    text_p.add_argument("--font", help="字体文件路径")
    text_p.add_argument("--font-size", type=int, default=36)
    text_p.add_argument("--opacity", type=float, default=0.7)
    text_p.add_argument("--position", default="bottom-right")
    text_p.add_argument("--margin", type=int, default=20)

    # Logo 水印
    img_p = sub.add_parser("image", help="Logo 水印")
    img_p.add_argument("-w", "--watermark", required=True, help="Logo 图片路径")
    img_p.add_argument("--scale", type=float, default=0.15)
    img_p.add_argument("--opacity", type=float, default=0.7)
    img_p.add_argument("--position", default="bottom-right")
    img_p.add_argument("--margin", type=int, default=20)

    args = parser.parse_args()

    if args.type == "text":
        batch_add_text_watermark(
            input_dir=args.input_dir,
            text=args.text,
            font_path=args.font,
            font_size=args.font_size,
            opacity=args.opacity,
            position=args.position,
            margin=args.margin,
        )
    elif args.type == "image":
        batch_add_image_watermark(
            input_dir=args.input_dir,
            watermark_path=args.watermark,
            scale=args.scale,
            opacity=args.opacity,
            position=args.position,
            margin=args.margin,
        )
