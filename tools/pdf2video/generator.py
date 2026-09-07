"""
PDF 转视频核心合成引擎 (PDF to Video Generator)

功能:
  - 串联 PDF 正文提取、页面高清渲染、Edge-TTS 异步并发配音
  - 动态音画时长对齐 (每页时长严格由语音时长 + 缓冲留白控制)
  - 逐页微切片渲染 + FFmpeg 毫秒级 Concat 快速合并
  - 支持同步导出对齐的 .srt 字幕文件 (默认画面不硬烧录，避免遮挡)
  - 支持轻柔 BGM 混音与淡出
"""
import asyncio
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from config import FFMPEG_BIN, OUTPUT_PDF2VIDEO
from tools.common import console, generate_output_name, get_video_info, logger
from tools.pdf2video.extractor import extract_script_from_pdf
from tools.pdf2video.renderer import render_pdf_to_images
from tools.slideshow.generator import resize_and_pad
from tools.tts.tts_generate import TTS_VOICES


def _format_srt_time(seconds: float) -> str:
    """将秒数转为 SRT 时间戳 00:00:00,000"""
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    if millis >= 1000:
        secs += 1
        millis = 0
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{millis:03d}"


def _format_ass_time(sec: float) -> str:
    """将秒数转为 ASS 时间戳 0:00:00.00"""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _wrap_text_ass(text: str, max_chars_per_line: int = 36, max_lines: int = 2) -> str:
    """
    按中文标点与字符宽度智能均衡折行，避免孤立标点和排版割裂
    """
    single_line = " ".join(text.split())
    if not single_line:
        return ""

    def get_width(s: str) -> int:
        return sum(2 if ord(c) > 255 else 1 for c in s)

    total_w = get_width(single_line)
    max_w = max_chars_per_line * 2

    # 单行放得下直接返回单行
    if total_w <= max_w or max_lines <= 1:
        return single_line

    # 尝试在文本中间寻找最合适的标点断句点，使两行长度最为均衡
    puncts = "，。！？；、 "
    mid = len(single_line) // 2
    best_split = -1
    best_diff = float("inf")

    search_start = max(1, mid - 14)
    search_end = min(len(single_line) - 1, mid + 14)
    for idx in range(search_start, search_end):
        if single_line[idx] in puncts:
            w1 = get_width(single_line[: idx + 1])
            w2 = get_width(single_line[idx + 1 :])
            if w1 <= max_w and w2 <= max_w:
                diff = abs(w1 - w2)
                if diff < best_diff:
                    best_diff = diff
                    best_split = idx + 1

    if best_split > 0:
        line1 = single_line[:best_split].strip()
        line2 = single_line[best_split:].strip()
        return line1 + r"\N" + line2

    # 无标点时居中对称折行，避开行首标点
    avoid_start = "，。！？；、）》」』”’"
    split_idx = mid
    while split_idx < len(single_line) - 1 and single_line[split_idx] in avoid_start:
        split_idx += 1

    line1 = single_line[:split_idx].strip()
    line2 = single_line[split_idx:].strip()
    return line1 + r"\N" + line2


def _split_sentence_into_phrases(text: str, start: float, end: float) -> list[dict]:
    """
    按各类标点符号 (逗号、分号、句号、顿号等) 将长句拆分为节奏紧凑的小短句 (8~16 字)，
    并按字数权重精准分配时间区间，彻底消除长句导致底框横贯屏幕或字体缩太小的问题。
    """
    import re

    raw_tokens = re.split(r"([，。；、！？,;!?:：\s])", text)
    clauses: list[str] = []
    curr = ""
    for part in raw_tokens:
        if not part:
            continue
        if part in "，。；、！？,;!?:： \t\n":
            curr += part
            if len(curr) >= 6:
                clauses.append(curr.strip())
                curr = ""
        else:
            curr += part

    if curr:
        if clauses and len(curr) < 4:
            clauses[-1] += curr.strip()
        else:
            clauses.append(curr.strip())

    # 对没有标点但依然超过 20 字的超长短语进行二次语义/均分拆解
    final_clauses: list[str] = []
    for c in clauses:
        if len(c) > 20:
            mid = len(c) // 2
            final_clauses.append(c[:mid].strip())
            final_clauses.append(c[mid:].strip())
        elif c:
            final_clauses.append(c)

    if not final_clauses:
        return [{"start": start, "end": end, "text": text}]

    total_chars = sum(max(1, len(c)) for c in final_clauses)
    total_dur = max(0.2, end - start)
    result: list[dict] = []
    c_start = start

    for idx, c in enumerate(final_clauses):
        c_dur = total_dur * (len(c) / total_chars)
        c_end = c_start + c_dur if idx + 1 < len(final_clauses) else end
        s_s = round(c_start, 3)
        s_e = round(c_end, 3)
        if s_e <= s_s:
            s_e = s_s + 0.1
        result.append({
            "start": s_s,
            "end": s_e,
            "text": c,
        })
        c_start = c_end

    return result


def _build_rounded_box_ass_path(box_x0: int, box_y0: int, box_w: int, box_h: int, r: int) -> str:
    """生成 ASS 矢量绘图指令 (贝塞尔曲线精确拟合圆角矩形)"""
    k = int(r * 0.5522847498)
    return (
        f"m {box_x0 + r} {box_y0} "
        f"l {box_x0 + box_w - r} {box_y0} "
        f"b {box_x0 + box_w - r + k} {box_y0} {box_x0 + box_w} {box_y0 + r - k} {box_x0 + box_w} {box_y0 + r} "
        f"l {box_x0 + box_w} {box_y0 + box_h - r} "
        f"b {box_x0 + box_w} {box_y0 + box_h - r + k} {box_x0 + box_w - r + k} {box_y0 + box_h} {box_x0 + box_w - r} {box_y0 + box_h} "
        f"l {box_x0 + r} {box_y0 + box_h} "
        f"b {box_x0 + r - k} {box_y0 + box_h} {box_x0} {box_y0 + box_h - r + k} {box_x0} {box_y0 + box_h - r} "
        f"l {box_x0} {box_y0 + r} "
        f"b {box_x0} {box_y0 + r - k} {box_x0 + r - k} {box_y0} {box_x0 + r} {box_y0}"
    )


def build_single_slide_ass(
    text: str,
    duration: float,
    resolution: tuple[int, int],
    output_path: Path,
    sentences: list[dict] | None = None,
    subtitle_style: str = "white_box",
    subtitle_layout: str = "split_phrases",
) -> Path:
    """
    为单页生成底部精致小字幕 ASS 文件 (贴底显示，避免遮挡页面)

    参数:
        text: 兜底全文
        duration: 单页总时长
        resolution: (宽, 高) 分辨率
        output_path: 输出 .ass 文件路径
        sentences: 句级时间戳列表 [{"start": 0.0, "end": 2.5, "text": "..."}]
        subtitle_style: 字幕颜色样式:
          - "white_box": 白字半透明黑底 (高质感矢量贝塞尔圆角框)
          - "black_transparent": 黑字透明底 (带白色微描边轮廓)
        subtitle_layout: 长句排版方案:
          - "split_phrases": 方案 1 (标点短句拆分流转，小巧精炼，大字清晰，推荐)
          - "double_line": 方案 2 (智能双行折行卡片，大字清晰，紧凑贴合底框)
          - "single_line_scale" / "auto_scale" / "fixed_bar": 方案 3 (纯单行自适应字号，长句绝对不换行，动态等比微缩字号)
    """
    width, height = resolution
    # 保持饱满醒目的大字号 (1080p 下为 45px，清晰易读)
    font_size = max(36, int(height // 24))
    margin_v = max(18, int(height * 0.02))  # 紧贴底部，为上方图表留出充足的安全白边
    margin_h = max(24, int(width * 0.02))
    is_white_box = subtitle_style in ("white_box", "white_bg_box")

    format_line = (
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
    )

    if is_white_box:
        style_lines = [
            "Style: BgBox,Arial,10,&H00000000,&H00000000,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
            f"Style: Default,PingFang SC,{font_size},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,0,0,2,0,0,0,1",
        ]
    else:
        style_lines = [
            f"Style: Default,PingFang SC,{font_size},&H00000000,&H000000FF,&H00FFFFFF,"
            f"&H00000000,-1,0,0,0,100,100,0,0,1,1.2,0,2,{margin_h},{margin_h},{margin_v},1"
        ]

    # WrapStyle: 2 严格禁止 libass 自动折行
    ass_header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 2

[V4+ Styles]
{format_line}
{"\n".join(style_lines)}

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    available_width = width - (margin_h * 2)
    max_chars = max(20, int(available_width / (font_size * 1.05)))
    pad_x = max(26, int(font_size * 0.6))
    pad_y = max(14, int(font_size * 0.32))

    # 1. 规范化输入句子列表
    raw_sentences: list[dict] = []
    if sentences:
        for s in sentences:
            st = (s.get("text") or "").strip()
            if not st:
                continue
            raw_sentences.append({
                "start": max(0.0, float(s.get("start", 0.0))),
                "end": max(0.0, float(s.get("end", duration))),
                "text": st,
            })
    elif text and text.strip():
        raw_sentences.append({
            "start": 0.0,
            "end": duration,
            "text": text.strip(),
        })

    # 2. 方案 1 (split_phrases): 标点短句细化拆分
    processed_sentences: list[dict] = []
    if subtitle_layout == "split_phrases":
        for s in raw_sentences:
            phrases = _split_sentence_into_phrases(s["text"], s["start"], s["end"])
            processed_sentences.extend(phrases)
    else:
        processed_sentences = raw_sentences

    # 消除相邻片段微秒级时间重合
    for idx in range(len(processed_sentences)):
        s_cur = processed_sentences[idx]
        if idx + 1 < len(processed_sentences):
            next_s = processed_sentences[idx + 1]["start"]
            if s_cur["end"] >= next_s:
                s_cur["end"] = max(s_cur["start"] + 0.05, next_s)
            elif (next_s - s_cur["end"]) < 0.35:
                s_cur["end"] = next_s
        else:
            s_cur["end"] = min(duration, max(s_cur["end"], duration - 0.2))

        if s_cur["end"] <= s_cur["start"]:
            s_cur["end"] = s_cur["start"] + 0.1

    # 3. 按不同排版方案生成 ASS 事件 (确保所有方案均保持大字清晰饱满)
    event_lines = []

    for s in processed_sentences:
        s_start = s["start"]
        s_end = s["end"]
        s_text = s["text"]
        start_str = _format_ass_time(s_start)
        end_str = _format_ass_time(s_end)

        if subtitle_layout == "double_line":
            # 方案 2: 智能双行折行卡片 (始终保持 45px 大字，超长自动折 2 行)
            formatted = _wrap_text_ass(s_text, max_chars_per_line=18, max_lines=2)
            if r"\N" in formatted:
                lines = formatted.split(r"\N")
                char_w = max(
                    sum(font_size * 1.0 if ord(c) > 255 else font_size * 0.55 for c in line)
                    for line in lines
                )
                line_spacing = int(font_size * 0.18)
                box_h = int(font_size * 2 + line_spacing + pad_y * 2)
                box_w = int(min(available_width, char_w + pad_x * 2))
            else:
                clean_t = formatted.replace(r"\N", " ")
                char_w = sum(font_size * 1.0 if ord(c) > 255 else font_size * 0.55 for c in clean_t)
                box_w = int(min(available_width, char_w + pad_x * 2))
                box_h = int(font_size + pad_y * 2)

            r = int(min(18, box_h / 2, box_w / 2))
            box_x0 = int((width - box_w) / 2)
            box_y0 = int(height - box_h - margin_v)
            text_x = int(width / 2)
            text_y = int(box_y0 + box_h - pad_y)

            if is_white_box:
                box_path = _build_rounded_box_ass_path(box_x0, box_y0, box_w, box_h, r)
                event_lines.append(
                    f"Dialogue: 0,{start_str},{end_str},BgBox,,0,0,0,,{{\\an7\\pos(0,0)\\p1\\c&H000000&\\1a&H70&\\bord0\\shad0}}{box_path}{{\\p0}}"
                )
                event_lines.append(
                    f"Dialogue: 1,{start_str},{end_str},Default,,0,0,0,,{{\\an2\\pos({text_x},{text_y})}}{formatted}"
                )
            else:
                event_lines.append(
                    f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{{\\an2\\pos({text_x},{text_y})}}{formatted}"
                )

        elif subtitle_layout in ("single_line_scale", "auto_scale", "fixed_bar", "single_line"):
            # 方案 3: 纯单行自适应字号 (长句绝对不换行，根据安全宽度等比缩小字号)
            max_allowed_w = int(width * 0.85)
            clean_t = " ".join(s_text.split())
            raw_char_w = sum(font_size * 1.0 if ord(c) > 255 else font_size * 0.55 for c in clean_t)

            if raw_char_w + pad_x * 2 <= max_allowed_w:
                cur_fs = font_size
                box_w = int(raw_char_w + pad_x * 2)
                box_h = int(font_size + pad_y * 2)
                text_tag = ""
            else:
                cur_fs = max(16, int((max_allowed_w - pad_x * 2) / (raw_char_w / font_size)))
                cur_char_w = sum(cur_fs * 1.0 if ord(c) > 255 else cur_fs * 0.55 for c in clean_t)
                box_w = int(cur_char_w + pad_x * 2)
                box_h = int(cur_fs + pad_y * 2)
                text_tag = f"{{\\fs{cur_fs}}}"

            r = int(min(18, box_h / 2, box_w / 2))
            box_x0 = int((width - box_w) / 2)
            box_y0 = int(height - box_h - margin_v)
            text_x = int(width / 2)
            text_y = int(box_y0 + box_h - pad_y)

            if is_white_box:
                box_path = _build_rounded_box_ass_path(box_x0, box_y0, box_w, box_h, r)
                event_lines.append(
                    f"Dialogue: 0,{start_str},{end_str},BgBox,,0,0,0,,{{\\an7\\pos(0,0)\\p1\\c&H000000&\\1a&H70&\\bord0\\shad0}}{box_path}{{\\p0}}"
                )
                event_lines.append(
                    f"Dialogue: 1,{start_str},{end_str},Default,,0,0,0,,{{\\an2\\pos({text_x},{text_y})}}{text_tag}{clean_t}"
                )
            else:
                event_lines.append(
                    f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{{\\an2\\pos({text_x},{text_y})}}{text_tag}{clean_t}"
                )

        else:
            # 方案 1: split_phrases (标点短句拆分流转，小巧精炼，始终 45px 大字)
            clean_t = " ".join(s_text.split())
            char_w = sum(font_size * 1.0 if ord(c) > 255 else font_size * 0.55 for c in clean_t)
            box_w = int(min(available_width, char_w + pad_x * 2))
            box_h = int(font_size + pad_y * 2)
            r = int(min(18, box_h / 2, box_w / 2))
            box_x0 = int((width - box_w) / 2)
            box_y0 = int(height - box_h - margin_v)
            text_x = int(width / 2)
            text_y = int(box_y0 + box_h - pad_y)

            if is_white_box:
                box_path = _build_rounded_box_ass_path(box_x0, box_y0, box_w, box_h, r)
                event_lines.append(
                    f"Dialogue: 0,{start_str},{end_str},BgBox,,0,0,0,,{{\\an7\\pos(0,0)\\p1\\c&H000000&\\1a&H70&\\bord0\\shad0}}{box_path}{{\\p0}}"
                )
                event_lines.append(
                    f"Dialogue: 1,{start_str},{end_str},Default,,0,0,0,,{{\\an2\\pos({text_x},{text_y})}}{clean_t}"
                )
            else:
                event_lines.append(
                    f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{{\\an2\\pos({text_x},{text_y})}}{clean_t}"
                )

    ass_content = ass_header + "\n".join(event_lines) + "\n"
    output_path.write_text(ass_content, encoding="utf-8")
    return output_path


async def _generate_slide_tts(
    text: str,
    voice_id: str,
    output_path: Path,
    sem: asyncio.Semaphore | None = None,
    max_retries: int = 3,
    on_complete: Callable | None = None,
) -> list[dict]:
    """异步调用 edge-tts 生成单页音频并捕获句级时间戳 (SentenceBoundary)"""
    import edge_tts

    for attempt in range(max_retries):
        try:
            sentences: list[dict] = []
            if sem:
                async with sem:
                    communicate = edge_tts.Communicate(text, voice_id)
                    with open(output_path, "wb") as f:
                        async for chunk in communicate.stream():
                            if chunk["type"] == "audio":
                                f.write(chunk["data"])
                            elif chunk["type"] == "SentenceBoundary":
                                start = chunk["offset"] / 10_000_000
                                dur = chunk["duration"] / 10_000_000
                                s_text = (chunk.get("text") or "").strip()
                                if s_text:
                                    s_start = round(start, 3)
                                    s_end = round(start + dur, 3)
                                    if sentences and sentences[-1]["end"] > s_start:
                                        sentences[-1]["end"] = s_start
                                    sentences.append({
                                        "start": s_start,
                                        "end": s_end,
                                        "text": s_text,
                                    })
            else:
                communicate = edge_tts.Communicate(text, voice_id)
                with open(output_path, "wb") as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
                        elif chunk["type"] == "SentenceBoundary":
                            start = chunk["offset"] / 10_000_000
                            dur = chunk["duration"] / 10_000_000
                            s_text = (chunk.get("text") or "").strip()
                            if s_text:
                                s_start = round(start, 3)
                                s_end = round(start + dur, 3)
                                if sentences and sentences[-1]["end"] > s_start:
                                    sentences[-1]["end"] = s_start
                                sentences.append({
                                    "start": s_start,
                                    "end": s_end,
                                    "text": s_text,
                                })

            if on_complete:
                on_complete()
            return sentences
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"TTS 生成失败 (尝试 {max_retries} 次): {e}")
                raise e
            await asyncio.sleep(0.4 * (attempt + 1))
    return []


def generate_pdf_video(
    pdf_path: Path | str,
    voice_key: str | None = None,
    voice: str | None = None,
    padding_after: float | None = None,
    page_padding: float | None = None,
    burn_subtitles: bool = True,
    subtitle_style: str = "white_box",
    subtitle_layout: str = "split_phrases",
    resolution: tuple[int, int] | str = (1920, 1080),
    music_path: Path | str | None = None,
    bgm_path: Path | str | None = None,
    music_volume: float | None = None,
    bgm_volume: float | None = None,
    output_path: Path | str | None = None,
    custom_script: list[dict] | None = None,
    show_progress: bool = True,
) -> dict:
    """
    将 PDF 转换为每一页均有人声朗读的高清视频

    Args:
        pdf_path: PDF 文件路径
        voice_key: 配音音色 key (见 TTS_VOICES) 或完整 voice 名称
        voice: voice_key 别名
        padding_after: 朗读完毕后留在当前页的静音留白秒数 (默认 0.8s)
        page_padding: padding_after 别名
        burn_subtitles: 是否将朗读字幕烧录在视频画面上 (默认 True)
        subtitle_style: 字幕样式 ("white_box" 白字半透明黑底圆角框 或 "black_transparent" 黑字透明底)
        subtitle_layout: 长句排版方案 ("split_phrases" 标点短句拆分流转 / "double_line" 智能双行 / "single_line_scale" 纯单行不换行)
        resolution: 最终视频分辨率 (宽, 高) 或 "1080p" / "2k" / "720p", 默认 (1920, 1080)
        music_path: 背景音乐文件路径
        bgm_path: music_path 别名
        music_volume: 背景音乐音量 (0.0~1.0, 默认 0.15)
        bgm_volume: music_volume 别名
        output_path: 指定输出路径 (若无则自动在 output/pdf2video/ 下生成)
        custom_script: 外置预加载的文案列表
        show_progress: 是否显示终端丰富进度条 (默认 True)

    Returns:
        {
            "output": 视频文件路径,
            "srt": 字幕文件路径,
            "duration": 总时长秒数,
            "slides_count": 总页数,
            "size_mb": 视频大小 MB
        }
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_path}")

    # 0. 参数归一化
    v_param = voice or voice_key or "xiaoxiao"
    if v_param in TTS_VOICES:
        voice_id = TTS_VOICES[v_param]["voice"]
        voice_label = TTS_VOICES[v_param].get("name", voice_id)
    elif any(v.get("voice") == v_param for v in TTS_VOICES.values()):
        voice_id = v_param
        voice_label = v_param
    elif v_param.startswith("zh-") or v_param.startswith("en-"):
        voice_id = v_param
        voice_label = v_param
    else:
        voice_id = "zh-CN-YunxiNeural"
        voice_label = "云希 (男声)"

    padding_val = page_padding if page_padding is not None else (padding_after if padding_after is not None else 0.8)
    padding_after = padding_val

    chosen_music_path = bgm_path or music_path
    chosen_music_volume = bgm_volume if bgm_volume is not None else (music_volume if music_volume is not None else 0.15)

    if isinstance(resolution, str):
        res_str = resolution.lower()
        if res_str in ("1080p", "1080"):
            resolution = (1920, 1080)
        elif res_str in ("2k", "1440p"):
            resolution = (2560, 1440)
        elif res_str in ("720p", "720"):
            resolution = (1280, 720)
        else:
            resolution = (1920, 1080)

    # 1. 准备输出路径
    OUTPUT_PDF2VIDEO.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        out_name = generate_output_name(f"pdf_{pdf_path.stem}", ".mp4")
        output_path = OUTPUT_PDF2VIDEO / out_name
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    srt_output_path = output_path.with_suffix(".srt")

    # 3. 创建临时工作空间
    temp_dir = Path(tempfile.mkdtemp(prefix="x-tools_pdf2video_"))
    slides_img_dir = temp_dir / "images"
    audio_dir = temp_dir / "audios"
    segments_dir = temp_dir / "segments"
    slides_img_dir.mkdir()
    audio_dir.mkdir()
    segments_dir.mkdir()

    try:
        # ============================================================
        # 步骤 A: 提取文案
        # ============================================================
        if custom_script:
            script = custom_script
            logger.info(f"使用用户自定义文案 (共 {len(script)} 页)")
        else:
            script = extract_script_from_pdf(pdf_path)

        # 预估总页数以初始化进度条
        total_est = len(script) if script else None
        if total_est is None:
            try:
                import pypdfium2 as pdfium
                _doc = pdfium.PdfDocument(str(pdf_path))
                total_est = len(_doc)
                _doc.close()
            except Exception:
                total_est = None

        progress_cm = (
            Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
            )
            if show_progress
            else nullcontext()
        )

        with progress_cm as progress:
            task_id = (
                progress.add_task("[cyan][1/4] 📄 渲染 PDF 页面[/cyan]", total=total_est)
                if progress
                else None
            )

            # ============================================================
            # 步骤 B: 渲染 PDF 高清画面
            # ============================================================
            def _on_page_rendered(curr, total):
                if progress and task_id is not None:
                    if total:
                        progress.update(task_id, total=total)
                    progress.advance(task_id)

            image_paths = render_pdf_to_images(
                pdf_path, slides_img_dir, scale=2.0, progress_callback=_on_page_rendered
            )
            total_slides = len(image_paths)
            if total_slides == 0:
                raise RuntimeError("未渲染出任何页面")

            if progress and task_id is not None:
                progress.update(task_id, completed=total_slides, total=total_slides)

            # 对齐文案与页面数
            while len(script) < total_slides:
                script.append({"page": len(script) + 1, "text": "", "source": "empty"})

            # ============================================================
            # 步骤 C: Edge-TTS 并发配音生成
            # ============================================================
            if progress and task_id is not None:
                progress.reset(
                    task_id,
                    total=total_slides,
                    description=f"[yellow][2/4] 🎙️ 生成语音配音 ({voice_label})[/yellow]",
                )
            logger.debug(f"正在生成 AI 语音配音 (音色: {voice_label})...")
            slide_audios: list[dict] = []

            for i in range(total_slides):
                page_num = i + 1
                text = (script[i].get("text") or "").strip()
                audio_file = audio_dir / f"slide_{page_num:04d}.mp3"

                slide_audios.append({
                    "page": page_num,
                    "text": text,
                    "audio_path": audio_file,
                    "duration": 0.0,
                    "sentences": [],
                })

            items_with_text = [item for item in slide_audios if item["text"]]
            empty_count = total_slides - len(items_with_text)
            if empty_count > 0 and progress and task_id is not None:
                progress.advance(task_id, advance=empty_count)

            async def _worker(item, sem):
                sents = await _generate_slide_tts(
                    item["text"],
                    voice_id,
                    item["audio_path"],
                    sem=sem,
                    on_complete=lambda: progress.advance(task_id) if progress and task_id is not None else None,
                )
                item["sentences"] = sents

            async def _run_all(items):
                sem = asyncio.Semaphore(4)
                coros = [_worker(item, sem) for item in items]
                await asyncio.gather(*coros)

            if items_with_text:
                asyncio.run(_run_all(items_with_text))

            if progress and task_id is not None:
                progress.update(task_id, completed=total_slides)

            # 测量每个音频文件的时长
            for item in slide_audios:
                p = item["audio_path"]
                if p.is_file() and p.stat().st_size > 0:
                    info = get_video_info(p)
                    dur = float(info.get("duration", 0.0))
                    item["duration"] = dur
                else:
                    item["duration"] = 0.0

            # ============================================================
            # 步骤 D: 计算每页时长 & 生成同步 SRT 字幕文件
            # ============================================================
            default_silent_dur = 3.0
            slide_durations: list[float] = []
            srt_lines: list[str] = []
            current_time = 0.0

            for i, item in enumerate(slide_audios):
                audio_dur = item["duration"]
                text = item["text"]
                sentences = item.get("sentences") or []

                if audio_dur > 0:
                    slide_dur = audio_dur + padding_after
                    text_end_time = current_time + audio_dur
                else:
                    slide_dur = default_silent_dur
                    text_end_time = current_time + slide_dur

                slide_durations.append(slide_dur)

                # 生成 SRT 条目 (支持句级时间戳高精对齐，严格防止时间戳重合)
                if sentences:
                    valid_s = []
                    for s in sentences:
                        st = (s.get("text") or "").strip()
                        if st:
                            valid_s.append({
                                "start": max(0.0, float(s.get("start", 0.0))),
                                "end": max(0.0, float(s.get("end", audio_dur))),
                                "text": st,
                            })

                    if subtitle_layout == "split_phrases":
                        expanded_s = []
                        for s in valid_s:
                            phrases = _split_sentence_into_phrases(s["text"], s["start"], s["end"])
                            expanded_s.extend(phrases)
                        valid_s = expanded_s

                    for idx, s in enumerate(valid_s):
                        s_start = s["start"]
                        s_end = s["end"]
                        if idx + 1 < len(valid_s):
                            next_start = valid_s[idx + 1]["start"]
                            if s_end > next_start:
                                s_end = next_start
                        if s_end <= s_start:
                            s_end = s_start + 0.1
                        srt_start = _format_srt_time(current_time + s_start)
                        srt_end = _format_srt_time(current_time + min(slide_dur, s_end))
                        clean_text = " ".join(s["text"].split())
                        srt_idx = len(srt_lines) + 1
                        srt_lines.append(f"{srt_idx}\n{srt_start} --> {srt_end}\n{clean_text}\n")
                elif text:
                    if subtitle_layout == "split_phrases":
                        phrases = _split_sentence_into_phrases(text, 0.0, audio_dur if audio_dur > 0 else slide_dur)
                        for s in phrases:
                            s_start = s["start"]
                            s_end = s["end"]
                            srt_start = _format_srt_time(current_time + s_start)
                            srt_end = _format_srt_time(current_time + min(slide_dur, s_end))
                            clean_text = " ".join(s["text"].split())
                            srt_idx = len(srt_lines) + 1
                            srt_lines.append(f"{srt_idx}\n{srt_start} --> {srt_end}\n{clean_text}\n")
                    else:
                        srt_start = _format_srt_time(current_time)
                        srt_end = _format_srt_time(text_end_time)
                        clean_text = " ".join(text.split())
                        srt_idx = len(srt_lines) + 1
                        srt_lines.append(f"{srt_idx}\n{srt_start} --> {srt_end}\n{clean_text}\n")

                current_time += slide_dur

            total_video_duration = current_time
            if srt_lines:
                srt_output_path.write_text("\n".join(srt_lines), encoding="utf-8")
                logger.debug(f"✅ 同步 SRT 字幕已输出: {srt_output_path.name}")

            # ============================================================
            # 步骤 E: 逐页渲染独立 MP4 切片
            # ============================================================
            if progress and task_id is not None:
                progress.reset(
                    task_id,
                    total=total_slides,
                    description="[green][3/4] 🎬 合成视频切片与字幕[/green]",
                )
            logger.debug(f"开始合成单页视频切片 (共 {total_slides} 页)...")
            target_w, target_h = resolution
            segment_files: list[Path] = []

            for i in range(total_slides):
                page_num = i + 1
                img_p = image_paths[i]
                slide_dur = slide_durations[i]
                audio_dur = slide_audios[i]["duration"]
                audio_file = slide_audios[i]["audio_path"]
                text = slide_audios[i]["text"]

                # 1. 预处理图片到统一分辨率 (保持宽高比并填充黑底)
                fitted_img_path = temp_dir / f"fit_{page_num:04d}.jpg"
                img = Image.open(img_p).convert("RGB")
                fitted_img, _, _, _, _ = resize_and_pad(img, target_w, target_h)
                fitted_img.save(fitted_img_path, "JPEG", quality=95)

                # 2. 单页 ASS 字幕 (如开启，支持句级动态平滑流转与自定义样式)
                ass_path = None
                if burn_subtitles and text:
                    ass_path = temp_dir / f"sub_{page_num:04d}.ass"
                    build_single_slide_ass(
                        text,
                        slide_dur,
                        resolution,
                        ass_path,
                        sentences=slide_audios[i].get("sentences"),
                        subtitle_style=subtitle_style,
                        subtitle_layout=subtitle_layout,
                    )

                # 3. 构造单页 FFmpeg 命令
                seg_out = segments_dir / f"seg_{page_num:04d}.mp4"
                cmd = [FFMPEG_BIN, "-y"]

                # 输入 0: 静态图片循环
                cmd.extend(["-loop", "1", "-t", f"{slide_dur:.3f}", "-i", str(fitted_img_path)])

                # 输入 1: 音频输入
                if audio_dur > 0 and audio_file.is_file():
                    cmd.extend(["-i", str(audio_file)])
                    filter_complex_parts = []
                    if ass_path:
                        escaped_ass = ass_path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
                        filter_complex_parts.append(f"[0:v]subtitles='{escaped_ass}'[v_out]")
                    else:
                        filter_complex_parts.append("[0:v]null[v_out]")

                    filter_complex_parts.append(f"[1:a]apad=pad_dur={padding_after:.3f}[a_out]")

                    cmd.extend([
                        "-filter_complex", ";".join(filter_complex_parts),
                        "-map", "[v_out]",
                        "-map", "[a_out]",
                    ])
                else:
                    # 纯静音占位音频
                    cmd.extend(["-f", "lavfi", "-t", f"{slide_dur:.3f}", "-i", "anullsrc=r=44100:cl=stereo"])
                    if ass_path:
                        escaped_ass = ass_path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
                        cmd.extend([
                            "-filter_complex", f"[0:v]subtitles='{escaped_ass}'[v_out]",
                            "-map", "[v_out]",
                            "-map", "1:a",
                        ])
                    else:
                        cmd.extend(["-map", "0:v", "-map", "1:a"])

                cmd.extend([
                    "-t", f"{slide_dur:.3f}",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "19",
                    "-pix_fmt", "yuv420p",
                    "-r", "30",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-ar", "44100",
                    "-ac", "2",
                    str(seg_out),
                ])

                res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if res.returncode != 0:
                    raise RuntimeError(f"渲染第 {page_num} 页失败: {res.stderr}")

                segment_files.append(seg_out)
                if progress and task_id is not None:
                    progress.advance(task_id)

            # ============================================================
            # 步骤 F: Concat 快速拼合所有切片
            # ============================================================
            if progress and task_id is not None:
                progress.reset(
                    task_id,
                    total=1,
                    description="[magenta][4/4] ⚡ 拼合并混入背景音乐[/magenta]",
                )
            concat_list_file = temp_dir / "concat_list.txt"
            concat_content = "".join(f"file '{p.resolve()}'\n" for p in segment_files)
            concat_list_file.write_text(concat_content, encoding="utf-8")

            merged_video = temp_dir / "merged_no_bgm.mp4"
            cmd_concat = [
                FFMPEG_BIN, "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_list_file),
                "-c", "copy",
                "-movflags", "+faststart",
                str(merged_video),
            ]
            logger.debug("正在执行多页切片快速拼接 (Concat)...")
            res_concat = subprocess.run(cmd_concat, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if res_concat.returncode != 0:
                raise RuntimeError(f"Concat 拼接失败: {res_concat.stderr}")

            # ============================================================
            # 步骤 G: BGM 背景音乐铺底 (可选)
            # ============================================================
            if chosen_music_path and Path(chosen_music_path).is_file():
                logger.debug(f"正在混入背景音乐: {Path(chosen_music_path).name} (音量: {chosen_music_volume})...")
                afade_start = max(0.0, total_video_duration - 2.0)
                cmd_bgm = [
                    FFMPEG_BIN, "-y",
                    "-i", str(merged_video),
                    "-stream_loop", "-1",
                    "-i", str(chosen_music_path),
                    "-filter_complex",
                    f"[1:a]volume={chosen_music_volume:.2f},afade=t=out:st={afade_start:.2f}:d=2[bgm];"
                    f"[0:a][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]",
                    "-map", "0:v",
                    "-map", "[aout]",
                    "-c:v", "copy",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-t", f"{total_video_duration:.3f}",
                    "-movflags", "+faststart",
                    str(output_path),
                ]
                res_bgm = subprocess.run(cmd_bgm, capture_output=True, text=True, encoding="utf-8", errors="replace")
                if res_bgm.returncode != 0:
                    logger.warning(f"BGM 混音失败，回退为无 BGM 视频: {res_bgm.stderr}")
                    shutil.move(str(merged_video), str(output_path))
            else:
                shutil.move(str(merged_video), str(output_path))

            if progress and task_id is not None:
                progress.advance(task_id)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(
            f"🎉 PDF 视频生成成功!\n"
            f"   - 视频文件: {output_path}\n"
            f"   - 总时长: {total_video_duration:.1f} 秒 ({total_slides} 页)\n"
            f"   - 大小: {size_mb:.2f} MB"
        )

        return {
            "output": str(output_path),
            "srt": str(srt_output_path) if srt_output_path.is_file() else None,
            "duration": round(total_video_duration, 2),
            "slides_count": total_slides,
            "size_mb": round(size_mb, 2),
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
