"""
画面对比与分屏拼接模块
"""
from tools.compare.ffmpeg_compare import (
    create_video_comparison,
    extract_model_name_from_path,
)

__all__ = [
    "create_video_comparison",
    "extract_model_name_from_path",
]
