"""
TTS 音频生成模块 (Text-to-Speech)

功能:
  - 将纯文本 (.txt) 文件批量转换为 MP3 音频
  - 支持多种英文/中文 AI 音色 (基于 edge-tts)
  - 适用于为 x-english 听力模块生成练习音频
"""
import asyncio
from pathlib import Path

from config import OUTPUT_TTS
from tools.common import generate_output_name, logger

# ============================================================
# 音色预设
# ============================================================
TTS_VOICES = {
    # 英文音色
    "jenny": {"name": "👩 Jenny (美式女声)", "voice": "en-US-JennyNeural", "lang": "en"},
    "guy": {"name": "👨 Guy (美式男声)", "voice": "en-US-GuyNeural", "lang": "en"},
    "sonia": {"name": "👩 Sonia (英式女声)", "voice": "en-GB-SoniaNeural", "lang": "en"},
    "ryan": {"name": "👨 Ryan (英式男声)", "voice": "en-GB-RyanNeural", "lang": "en"},
    # 中文音色
    "xiaoxiao": {"name": "👧 晓晓 (温柔女声)", "voice": "zh-CN-XiaoxiaoNeural", "lang": "zh"},
    "yunxi": {"name": "👦 云希 (活力男声)", "voice": "zh-CN-YunxiNeural", "lang": "zh"},
}


async def _generate_single(text: str, voice: str, output_path: Path):
    """使用 edge-tts 生成单条音频"""
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(output_path))


def generate_audio(
    text: str,
    voice_key: str = "jenny",
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    filename: str | None = None,
) -> dict:
    """
    将一段文本生成为 MP3 音频文件

    Args:
        text: 要朗读的文本内容
        voice_key: 音色 key (参见 TTS_VOICES)
        output_path: 指定输出文件完整路径 (优先级最高)
        output_dir: 输出目录 (默认 OUTPUT_TTS)
        filename: 输出文件名 (不含后缀, 与 output_dir 搭配使用)

    Returns:
        {"output": str, "size_mb": float}
    """
    voice_info = TTS_VOICES.get(voice_key, TTS_VOICES["jenny"])
    voice_id = voice_info["voice"]

    # 确定输出路径
    if output_path is None:
        out_dir = Path(output_dir) if output_dir else OUTPUT_TTS
        out_dir.mkdir(parents=True, exist_ok=True)
        if filename:
            out_file = out_dir / f"{filename}.mp3"
        else:
            out_file = out_dir / generate_output_name("tts", ".mp3")
    else:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

    # 生成音频
    asyncio.run(_generate_single(text, voice_id, out_file))

    if not out_file.is_file() or out_file.stat().st_size == 0:
        raise RuntimeError(f"TTS 生成失败: {out_file}")

    size_mb = out_file.stat().st_size / (1024 * 1024)
    logger.info(f"✅ 音频已生成: {out_file.name} ({size_mb:.2f} MB)")

    return {
        "output": str(out_file),
        "size_mb": round(size_mb, 2),
    }


def batch_generate(
    input_dir: str | Path | None = None,
    voice_key: str = "jenny",
    output_dir: str | Path | None = None,
    files: list[Path] | None = None,
) -> list[dict]:
    """
    批量将 .txt 文件转换为 MP3 音频

    Args:
        input_dir: 输入目录 (扫描其中的 .txt 文件)
        voice_key: 音色 key
        output_dir: 输出目录
        files: 已指定的文件列表 (优先使用)

    Returns:
        结果列表 [{"file": str, "status": str, "output": str, ...}, ...]
    """
    from config import INPUT_DIR

    # 扫描 .txt 文件
    if files is not None:
        txt_files = files
    else:
        scan_dir = Path(input_dir) if input_dir else INPUT_DIR
        txt_files = sorted(
            f for f in scan_dir.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        )

    if not txt_files:
        logger.warning("未找到 .txt 文件")
        return []

    out_dir = Path(output_dir) if output_dir else OUTPUT_TTS
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"扫描到 {len(txt_files)} 个文本文件, 音色: {voice_key}")

    results = []
    for txt_file in txt_files:
        txt_file = Path(txt_file)
        text = txt_file.read_text(encoding="utf-8").strip()
        if not text:
            logger.warning(f"跳过空文件: {txt_file.name}")
            results.append({"file": str(txt_file), "status": "skipped", "error": "文件为空"})
            continue

        try:
            result = generate_audio(
                text=text,
                voice_key=voice_key,
                output_dir=out_dir,
                filename=txt_file.stem,
            )
            results.append({"file": str(txt_file), "status": "success", **result})
        except Exception as e:
            logger.error(f"❌ 生成失败: {txt_file.name} — {e}")
            results.append({"file": str(txt_file), "status": "error", "error": str(e)})

    # 汇总
    success = sum(1 for r in results if r["status"] == "success")
    failed = len(results) - success
    logger.info(f"TTS 批量生成完成: ✅ {success} 成功, ❌ {failed} 失败/跳过")

    return results
