"""
PDF 转视频模块 (PDF to Video with Voice Narration)
"""
from tools.pdf2video.extractor import (
    clean_script_text,
    extract_script_from_pdf,
    load_script,
    save_script,
)
from tools.pdf2video.generator import generate_pdf_video
from tools.pdf2video.renderer import render_pdf_to_images

__all__ = [
    "clean_script_text",
    "extract_script_from_pdf",
    "save_script",
    "load_script",
    "render_pdf_to_images",
    "generate_pdf_video",
]
