"""
x-tools 全局配置
"""
from pathlib import Path

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# 各功能输出子目录
OUTPUT_EXTRACT = OUTPUT_DIR / "extract"
OUTPUT_WATERMARK = OUTPUT_DIR / "watermark"
OUTPUT_UPSCALE = OUTPUT_DIR / "upscale"
OUTPUT_INTERPOLATION = OUTPUT_DIR / "interpolation"

# ============================================================
# 支持的视频格式
# ============================================================
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}

# ============================================================
# FFmpeg 配置
# ============================================================
FFMPEG_BIN = "ffmpeg"      # 如果不在 PATH 中，改为绝对路径
FFPROBE_BIN = "ffprobe"

# ============================================================
# 默认参数
# ============================================================

# 内容截取
EXTRACT_DEFAULT_FORMAT = "mp4"          # 默认输出格式
KEYFRAME_IMAGE_FORMAT = "jpg"           # 关键帧图片格式
KEYFRAME_IMAGE_QUALITY = 95             # 关键帧图片质量 (1-100)

# 去水印
WATERMARK_INPAINT_RADIUS = 5            # OpenCV inpaint 修复半径

# 高清重置
UPSCALE_FACTOR = 2                      # 默认放大倍数 (2x / 4x)

# 帧数补充
INTERPOLATION_TARGET_FPS = 60           # 目标帧率


def ensure_dirs():
    """确保所有必要目录存在"""
    for d in [INPUT_DIR, OUTPUT_EXTRACT, OUTPUT_WATERMARK,
              OUTPUT_UPSCALE, OUTPUT_INTERPOLATION]:
        d.mkdir(parents=True, exist_ok=True)
