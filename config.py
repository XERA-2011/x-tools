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
OUTPUT_FILTER = OUTPUT_DIR / "filter"
OUTPUT_CROP = OUTPUT_DIR / "crop"
OUTPUT_CONCAT = OUTPUT_DIR / "concat"
OUTPUT_BGM = OUTPUT_DIR / "bgm"
OUTPUT_SUBTITLE = OUTPUT_DIR / "subtitle"

OUTPUT_MV = OUTPUT_DIR / "mv"
OUTPUT_COMPRESS = OUTPUT_DIR / "compress"
OUTPUT_SLIDESHOW = OUTPUT_DIR / "slideshow"
OUTPUT_PDF_SPLIT = OUTPUT_DIR / "pdf_split"

# 音乐目录
MUSIC_DIR = PROJECT_ROOT / "music"

# ============================================================
# 支持的视频格式
# ============================================================
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".aac", ".m4a", ".flac", ".ogg", ".wma"}

# ============================================================
# FFmpeg 配置
# ============================================================
import os
import shutil

# 尝试寻找更全功能的 ffmpeg-full (macOS Homebrew keg-only), 因为标准 ffmpeg(8.0+) 不包含 libass
_ffmpeg_full = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffmpeg")
_ffprobe_full = Path("/opt/homebrew/opt/ffmpeg-full/bin/ffprobe")

if _ffmpeg_full.exists():
    FFMPEG_BIN = str(_ffmpeg_full)
    FFPROBE_BIN = str(_ffprobe_full) if _ffprobe_full.exists() else "ffprobe"
    # 将 ffmpeg-full 的路径加入到环境变量 PATH 的最前面，
    # 以便第三方库（如 whisper 内部调用的 ffmpeg）能够找到并使用它。
    _full_bin_dir = str(_ffmpeg_full.parent)
    os.environ["PATH"] = f"{_full_bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
else:
    FFMPEG_BIN = shutil.which("ffmpeg") or "ffmpeg"
    FFPROBE_BIN = shutil.which("ffprobe") or "ffprobe"

# ============================================================
# 默认参数
# ============================================================

# 去水印
WATERMARK_INPAINT_RADIUS = 5            # OpenCV inpaint 修复半径
WATERMARK_BRAND_PRESETS = {
    # Veo 示例视频 (1080x1920) 右下角文字水印坐标
    # 坐标格式: (x1, y1, x2, y2), 参考分辨率用于自动缩放
    "veo": {
        "label": "Veo (右下角文字)",
        "ref_width": 1080,
        "ref_height": 1920,
        "regions": [(949, 1839, 1061, 1898)],
    },
}

# 加水印
ADD_WATERMARK_FONT_SIZE = 50            # 默认字号
ADD_WATERMARK_OPACITY = 0.7             # 默认透明度 (0.0~1.0)
ADD_WATERMARK_COLOR = (255, 255, 255)   # 默认颜色 (白色, RGB)
ADD_WATERMARK_POSITION = "bottom-right" # 默认位置
ADD_WATERMARK_MARGIN = 20               # 距离边缘的像素间距
ADD_WATERMARK_LOGO_SCALE = 0.15         # Logo 水印占画面宽度比例
ADD_WATERMARK_TEXT = "XΞЯΛ.ΛI"          # 默认水印文字
ADD_WATERMARK_STROKE_WIDTH = 0          # 文字描边宽度 (模拟加粗, 0=不加粗)

# 高清重置
UPSCALE_FACTOR = 2                      # 默认放大倍数 (2x / 4x)

# 帧数补充
INTERPOLATION_TARGET_FPS = 60           # 目标帧率

# ============================================================
# 自动质量检测 (QC) 默认阈值
# ============================================================
QC_MIN_WIDTH = 1280                     # 低于该宽度视为低清
QC_MIN_HEIGHT = 720                     # 低于该高度视为低清
QC_MIN_FPS = 23.0                       # 低于该帧率视为低帧率
QC_MIN_AUDIO_SAMPLE_RATE = 44100        # 低于该采样率视为低采样率

# 码率阈值 (bps), 按分辨率等级粗略判断
QC_MIN_BITRATE_720P = 800_000
QC_MIN_BITRATE_1080P = 2_500_000
QC_MIN_BITRATE_4K = 8_000_000

# 黑场 / 静音 / 冻结检测阈值 (秒)
QC_BLACK_SEGMENT_WARN_SEC = 2.0
QC_BLACK_TOTAL_WARN_SEC = 5.0
QC_SILENCE_SEGMENT_WARN_SEC = 3.0
QC_SILENCE_TOTAL_WARN_SEC = 10.0
QC_FREEZE_SEGMENT_WARN_SEC = 2.0
QC_FREEZE_TOTAL_WARN_SEC = 5.0


def ensure_dirs():
    """确保所有必要目录存在"""
    for d in [INPUT_DIR, OUTPUT_WATERMARK,
              OUTPUT_ADD_WATERMARK, OUTPUT_UPSCALE, OUTPUT_INTERPOLATION,
              OUTPUT_CONVERT, OUTPUT_FILTER, OUTPUT_CROP, OUTPUT_CONCAT,
              OUTPUT_BGM, OUTPUT_SUBTITLE, OUTPUT_MV, OUTPUT_COMPRESS,
              OUTPUT_SLIDESHOW, OUTPUT_PDF_SPLIT, MUSIC_DIR]:
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
