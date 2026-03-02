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
OUTPUT_WATERMARK = OUTPUT_DIR / "watermark"
OUTPUT_ADD_WATERMARK = OUTPUT_DIR / "add_watermark"
OUTPUT_UPSCALE = OUTPUT_DIR / "upscale"
OUTPUT_INTERPOLATION = OUTPUT_DIR / "interpolation"
OUTPUT_CONVERT = OUTPUT_DIR / "convert"

# ============================================================
# 支持的视频格式
# ============================================================
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# ============================================================
# FFmpeg 配置
# ============================================================
FFMPEG_BIN = "ffmpeg"      # 如果不在 PATH 中，改为绝对路径
FFPROBE_BIN = "ffprobe"

# ============================================================
# 默认参数
# ============================================================

# 去水印
WATERMARK_INPAINT_RADIUS = 5            # OpenCV inpaint 修复半径

# 加水印
ADD_WATERMARK_FONT_SIZE = 50            # 默认字号
ADD_WATERMARK_OPACITY = 0.7             # 默认透明度 (0.0~1.0)
ADD_WATERMARK_COLOR = (255, 255, 255)   # 默认颜色 (白色, RGB)
ADD_WATERMARK_POSITION = "bottom-right" # 默认位置
ADD_WATERMARK_MARGIN = 20               # 距离边缘的像素间距
ADD_WATERMARK_LOGO_SCALE = 0.15         # Logo 水印占画面宽度比例
ADD_WATERMARK_TEXT = "XΞЯΛ.ΛI"          # 默认水印文字
ADD_WATERMARK_STROKE_WIDTH = 1          # 文字描边宽度 (模拟加粗, 0=不加粗)

# 高清重置
UPSCALE_FACTOR = 2                      # 默认放大倍数 (2x / 4x)

# 帧数补充
INTERPOLATION_TARGET_FPS = 60           # 目标帧率


def ensure_dirs():
    """确保所有必要目录存在"""
    for d in [INPUT_DIR, OUTPUT_WATERMARK,
              OUTPUT_ADD_WATERMARK, OUTPUT_UPSCALE, OUTPUT_INTERPOLATION,
              OUTPUT_CONVERT]:
        d.mkdir(parents=True, exist_ok=True)


# ============================================================
# 参数验证
# ============================================================
def clamp(value: float, min_val: float, max_val: float, name: str = "") -> float:
    """将值限制在合法范围内, 超出范围时打印警告"""
    if value < min_val or value > max_val:
        clamped = max(min_val, min(max_val, value))
        if name:
            import logging
            logging.getLogger("x-tools").warning(
                f"{name}={value} 超出范围 [{min_val}, {max_val}], 已修正为 {clamped}"
            )
        return clamped
    return value
