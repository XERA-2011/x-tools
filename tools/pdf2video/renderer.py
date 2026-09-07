"""
PDF 页面高清渲染模块

功能:
  - 基于 pypdfium2 将 PDF 每页高保真渲染为高清 PNG 图片 (1080P/2K)
"""
from collections.abc import Callable
from pathlib import Path

from tools.common import logger


def render_pdf_to_images(
    pdf_path: Path | str,
    output_dir: Path | str,
    scale: float = 2.0,
    progress_callback: Callable | None = None,
) -> list[Path]:
    """
    使用 pypdfium2 将 PDF 每页渲染为高清 PNG

    Args:
        pdf_path: PDF 路径
        output_dir: 图片输出目录
        scale: 渲染缩放因子 (scale=2.0 渲染效果约等于 150~200 DPI，适合 1080P/2K)
        progress_callback: 每页渲染完成的回调函数 callback(current_page, total_pages)

    Returns:
        渲染出的有序图片路径列表 [slide_0001.png, slide_0002.png, ...]
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise ImportError("未安装 pypdfium2，请在虚拟环境中执行: pip install pypdfium2")

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    image_paths: list[Path] = []

    try:
        total_pages = len(pdf)
        logger.debug(f"开始渲染 PDF 页面 (共 {total_pages} 页, 缩放: {scale}x)...")

        for i, page in enumerate(pdf):
            page_num = i + 1
            image = page.render(scale=scale).to_pil()
            out_file = out_dir / f"slide_{page_num:04d}.png"
            image.save(out_file, format="PNG")
            image_paths.append(out_file)
            if progress_callback:
                progress_callback(page_num, total_pages)

        logger.debug(f"✅ PDF {total_pages} 页画面渲染完成")
        return image_paths
    finally:
        pdf.close()
