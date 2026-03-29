"""
多引擎翻译模块 (自动降级)

翻译优先级:
  1. Whisper task=translate (离线, 仅支持 X→英文)
  2. Google Translate      (在线, 国内可能需要梯子)
  3. MyMemory Translator   (在线, 国内可直连)
"""
from tools.common import logger

# deep-translator 语言代码映射
_LANG_MAP = {
    "zh": "zh-CN",
    "chinese": "zh-CN",
    "en": "en",
    "english": "en",
    "ja": "ja",
    "japanese": "ja",
}


def _normalize_lang(lang: str) -> str:
    """将 Whisper 语言代码映射为 deep-translator 使用的代码"""
    return _LANG_MAP.get(lang, lang)


def _try_google(texts: list[str], source: str, target: str, timeout: int = 5) -> list[str] | None:
    """尝试 Google Translate, 失败返回 None"""
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source=source, target=target)
        results = translator.translate_batch(texts)
        logger.info("✅ 翻译引擎: Google Translate")
        return results
    except Exception as e:
        logger.warning(f"Google Translate 不可用: {e}")
        return None


def _try_mymemory(texts: list[str], source: str, target: str) -> list[str] | None:
    """尝试 MyMemory Translator (国内可直连), 失败返回 None"""
    try:
        from deep_translator import MyMemoryTranslator
        translator = MyMemoryTranslator(source=source, target=target)
        # MyMemory 不支持 translate_batch, 逐条翻译
        results = []
        for text in texts:
            try:
                results.append(translator.translate(text))
            except Exception:
                results.append(text)  # 单条失败保留原文
        logger.info("✅ 翻译引擎: MyMemory (备用)")
        return results
    except Exception as e:
        logger.warning(f"MyMemory 翻译不可用: {e}")
        return None


def translate_texts(texts: list[str], source_lang: str, target_lang: str) -> list[str]:
    """
    翻译文本列表, 自动降级引擎

    Args:
        texts: 待翻译的文本列表
        source_lang: 源语言 (Whisper 语言代码, 如 "zh", "en")
        target_lang: 目标语言

    Returns:
        翻译后的文本列表 (与输入等长, 失败时返回原文)
    """
    if not texts:
        return []

    src = _normalize_lang(source_lang)
    tgt = _normalize_lang(target_lang)

    # 引擎 1: Google Translate
    result = _try_google(texts, src, tgt)
    if result:
        return result

    # 引擎 2: MyMemory (国内备用)
    result = _try_mymemory(texts, src, tgt)
    if result:
        return result

    # 全部失败, 返回原文
    logger.error("⚠️ 所有翻译引擎均不可用, 跳过翻译")
    return texts


def translate_segments(
    segments: list[dict],
    source_lang: str,
    target_lang: str,
) -> list[dict]:
    """
    批量翻译 Whisper segments

    Args:
        segments: Whisper 输出的 segments 列表,
                  每项含 {"start", "end", "text", ...}
        source_lang: 源语言
        target_lang: 目标语言

    Returns:
        翻译后的 segments 列表,
        每项新增 "translated_text" 字段
    """
    texts = [seg["text"].strip() for seg in segments]
    translated = translate_texts(texts, source_lang, target_lang)

    result = []
    for seg, trans in zip(segments, translated):
        new_seg = dict(seg)
        new_seg["translated_text"] = trans
        result.append(new_seg)

    return result
