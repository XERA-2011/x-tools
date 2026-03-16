"""
Whisper 语音识别 → 字幕文件 (.srt)

功能:
  - 使用 OpenAI Whisper 模型识别视频中的语音
  - 自动生成 SRT 格式字幕文件
  - 支持中文/英文/日文等多语言
  - 可选模型大小 (tiny/base/small/medium/large)
"""
import subprocess
from pathlib import Path

from config import FFMPEG_BIN, OUTPUT_SUBTITLE
from tools.common import generate_output_name, logger

# Whisper 模型配置
WHISPER_MODELS = {
    "tiny": {"name": "⚡ tiny (最快, 精度低)", "size": "~39MB"},
    "base": {"name": "🔹 base (快速, 基本精度)", "size": "~74MB"},
    "small": {"name": "⭐ small (推荐, 精度与速度平衡)", "size": "~461MB"},
    "medium": {"name": "💎 medium (高精度, 较慢)", "size": "~1.5GB"},
    "large": {"name": "🏆 large (最高精度, 最慢)", "size": "~2.9GB"},
}


def _extract_audio(video_path: Path, audio_path: Path):
    """从视频中提取音频为 WAV (Whisper 需要)"""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vn",                      # 不要视频
        "-acodec", "pcm_s16le",     # WAV 格式
        "-ar", "16000",             # 16kHz (Whisper 要求)
        "-ac", "1",                 # 单声道
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"音频提取失败: {result.stderr[-300:]}")


def _format_timestamp(seconds: float) -> str:
    """将秒数转为 SRT 时间格式 HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _segments_to_srt(segments: list, language: str = "") -> str:
    """将 Whisper segments 转为 SRT 格式文本"""
    # 如果语言是中文，尝试使用 zhconv 转换为简体
    use_zhconv = False
    if language in ["zh", "chinese"]:
        try:
            import zhconv
            use_zhconv = True
        except ImportError:
            pass

    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_timestamp(seg["start"])
        end = _format_timestamp(seg["end"])
        text = seg["text"].strip()
        
        if use_zhconv and text:
            text = zhconv.convert(text, "zh-cn")
            
        if text:
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def transcribe_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    model_name: str = "small",
    language: str | None = None,
) -> dict:
    """
    使用 Whisper 识别视频语音并生成 SRT 字幕文件

    Args:
        video_path: 视频文件路径
        output_path: 字幕输出路径 (默认自动生成)
        model_name: Whisper 模型 (tiny/base/small/medium/large)
        language: 语言代码 (None=自动检测, "zh"=中文, "en"=英语, "ja"=日语)

    Returns:
        dict: {"output": str, "language": str, "segments": int}
    """
    try:
        import whisper
    except ImportError:
        raise ImportError("请先安装 whisper: pip install openai-whisper")

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 输出路径
    OUTPUT_SUBTITLE.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_name = generate_output_name(video_path.stem, ".srt", tag="sub")
        output_path = OUTPUT_SUBTITLE / output_name
    output_path = Path(output_path)

    # 1. 提取音频
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = Path(tmp.name)

    try:
        logger.info(f"提取音频: {video_path.name}")
        _extract_audio(video_path, audio_path)

        # 2. 加载 Whisper 模型
        logger.info(f"加载 Whisper 模型: {model_name} (首次会下载)")
        model = whisper.load_model(model_name)

        # 3. 识别
        logger.info("开始语音识别...")
        transcribe_opts = {"verbose": False}
        if language:
            transcribe_opts["language"] = language

        result = model.transcribe(str(audio_path), **transcribe_opts)

        # 4. 生成 SRT
        detected_lang = result.get("language", "unknown")
        segments = result.get("segments", [])
        srt_content = _segments_to_srt(segments, language=detected_lang)

        output_path.write_text(srt_content, encoding="utf-8")

        logger.info(
            f"✅ 字幕生成完成: {output_path.name} "
            f"({len(segments)} 条, 语言: {detected_lang})"
        )

        return {
            "output": str(output_path),
            "language": detected_lang,
            "segments": len(segments),
        }
    finally:
        # 清理临时音频
        if audio_path.exists():
            audio_path.unlink()
