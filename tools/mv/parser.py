"""
歌词解析器

支持格式:
  - 纯文本 (按行分割, 均匀分配时间)
  - LRC 格式 ([00:01.50]歌词内容)
  - SRT 格式 (标准字幕)
  - Whisper word-level segments (直接转换)
"""
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LyricWord:
    """单个词的时间信息"""
    text: str
    start_time: float  # 秒
    end_time: float    # 秒


@dataclass
class LyricLine:
    """一行歌词"""
    start_time: float     # 秒
    end_time: float       # 秒
    text: str             # 完整歌词文本
    words: list[LyricWord] = field(default_factory=list)  # 逐词时间

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def parse_lrc(content: str) -> list[LyricLine]:
    """解析 LRC 格式歌词"""
    pattern = re.compile(r"\[(\d{2}):(\d{2})\.(\d{2,3})\](.*)")
    lines = []

    for line in content.strip().split("\n"):
        m = pattern.match(line.strip())
        if not m:
            continue
        minutes, seconds, centis, text = m.groups()
        # 处理 2 位和 3 位毫秒
        if len(centis) == 2:
            ms = int(centis) * 10
        else:
            ms = int(centis)
        start_time = int(minutes) * 60 + int(seconds) + ms / 1000
        text = text.strip()
        if text:
            lines.append(LyricLine(start_time=start_time, end_time=0.0, text=text))

    # 根据下一行的开始时间计算每行的结束时间
    for i in range(len(lines) - 1):
        lines[i].end_time = lines[i + 1].start_time
    if lines:
        # 最后一行默认持续 4 秒
        lines[-1].end_time = lines[-1].start_time + 4.0

    # 自动拆分词
    for line in lines:
        _split_words(line)

    return lines


def parse_srt(content: str) -> list[LyricLine]:
    """解析 SRT 格式字幕"""
    time_pattern = re.compile(
        r"(\d{2}):(\d{2}):(\d{2}),(\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    )
    blocks = re.split(r"\n\n+", content.strip())
    lines = []

    for block in blocks:
        block_lines = block.strip().split("\n")
        if len(block_lines) < 3:
            continue

        m = time_pattern.search(block_lines[1])
        if not m:
            continue

        start = (
            int(m.group(1)) * 3600
            + int(m.group(2)) * 60
            + int(m.group(3))
            + int(m.group(4)) / 1000
        )
        end = (
            int(m.group(5)) * 3600
            + int(m.group(6)) * 60
            + int(m.group(7))
            + int(m.group(8)) / 1000
        )
        text = " ".join(block_lines[2:]).strip()
        if text:
            line = LyricLine(start_time=start, end_time=end, text=text)
            _split_words(line)
            lines.append(line)

    return lines


def parse_plain_text(content: str, total_duration: float) -> list[LyricLine]:
    """
    解析纯文本歌词, 按行均匀分配时间

    Args:
        content: 歌词文本 (每行一句)
        total_duration: 音乐总时长 (秒)
    """
    raw_lines = [line.strip() for line in content.strip().split("\n") if line.strip()]
    if not raw_lines:
        return []

    duration_per_line = total_duration / len(raw_lines)
    lines = []
    for i, text in enumerate(raw_lines):
        start = i * duration_per_line
        end = (i + 1) * duration_per_line
        line = LyricLine(start_time=start, end_time=end, text=text)
        _split_words(line)
        lines.append(line)

    return lines


def from_whisper_segments(segments: list[dict]) -> list[LyricLine]:
    """
    从 Whisper word-level segments 转换

    Whisper 的 segments 格式:
    [{"start": 0.0, "end": 2.0, "text": "hello world", "words": [
        {"word": "hello", "start": 0.0, "end": 1.0},
        {"word": "world", "start": 1.0, "end": 2.0}
    ]}, ...]
    """
    lines = []

    for seg in segments:
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)

        words_list = seg.get("words", [])
        
        # 如果 Whisper 没有 word-level (旧版本或未启用 word_timestamps)
        if not words_list:
            text = seg.get("text", "").strip()
            if not text:
                continue
            line = LyricLine(start_time=start, end_time=end, text=text, words=[])
            _split_words(line)
            lines.append(line)
            continue
            
        # 否则通过截断标点和最大单词数，将长段落切分为极具视觉冲击力的短句
        chunk_words = []
        for w in words_list:
            word_text = w.get("word", "").strip()
            
            # 过滤不需要的间奏转录词 ("Music", "♪" 等)
            lw = word_text.lower()
            if not word_text or lw == 'music' or '♪' in lw or lw.startswith('[') or lw.endswith(']'):
                continue
                
            chunk_words.append(LyricWord(
                text=word_text,
                start_time=w.get("start", start),
                end_time=w.get("end", end),
            ))
            
            # 使用标点符号或词数进行换句断点设计
            # 如果到达 5 个词，或者遇到句号/逗号，执行屏幕刷新
            if word_text[-1] in ".?!,;:，。？！、\n" or len(chunk_words) >= 5:
                # 抛弃最后的空格/标点（如果希望更干脆的字体效果，可以在渲染层丢弃，这里保留原词以防意外）
                text_chunk = " ".join(ww.text for ww in chunk_words)
                lines.append(LyricLine(
                    start_time=chunk_words[0].start_time,
                    end_time=chunk_words[-1].end_time,
                    text=text_chunk,
                    words=chunk_words
                ))
                chunk_words = []
                
        if chunk_words:
            text_chunk = " ".join(ww.text for ww in chunk_words)
            lines.append(LyricLine(
                start_time=chunk_words[0].start_time,
                end_time=chunk_words[-1].end_time,
                text=text_chunk,
                words=chunk_words
            ))

    return lines


def _split_words(line: LyricLine):
    """
    将一行歌词拆分为词，均匀分配时间

    中文按字拆分, 英文按空格拆分
    """
    text = line.text.strip()
    if not text:
        return

    # 智能拆分: 中文按字符, 英文按空格
    tokens = _tokenize(text)
    if not tokens:
        return

    duration = line.end_time - line.start_time
    if duration <= 0:
        duration = len(tokens) * 0.3  # 默认每词 0.3 秒
        line.end_time = line.start_time + duration

    dur_per_word = duration / len(tokens)
    words = []
    for i, token in enumerate(tokens):
        words.append(LyricWord(
            text=token,
            start_time=line.start_time + i * dur_per_word,
            end_time=line.start_time + (i + 1) * dur_per_word,
        ))
    line.words = words


def _tokenize(text: str) -> list[str]:
    """
    智能分词:
    - 英文/数字: 按空格分词
    - 中文/日文: 按字符分词
    - 标点符号附着到前一个词
    """
    tokens = []
    # 先按空格分割, 然后对中文/日文进一步按字符分
    parts = text.split()
    for part in parts:
        # 检测是否含有 CJK 字符
        has_cjk = any(_is_cjk(c) for c in part)
        if has_cjk:
            # 按字符分, 但标点附着到前一个
            for char in part:
                if _is_cjk(char):
                    tokens.append(char)
                elif tokens:
                    # 标点附着到前一个词
                    tokens[-1] += char
                else:
                    tokens.append(char)
        else:
            tokens.append(part)
    return tokens


def _is_cjk(char: str) -> bool:
    """检测字符是否为 CJK 字符"""
    cp = ord(char)
    return (
        (0x4E00 <= cp <= 0x9FFF)      # CJK Unified Ideographs
        or (0x3400 <= cp <= 0x4DBF)   # CJK Unified Ideographs Extension A
        or (0x3040 <= cp <= 0x30FF)   # Hiragana + Katakana
        or (0xAC00 <= cp <= 0xD7AF)   # Hangul
    )


def load_lyrics(path: str | Path, total_duration: float = 0.0) -> list[LyricLine]:
    """
    根据文件扩展名自动选择解析器

    Args:
        path: 歌词文件路径
        total_duration: 音乐总时长 (纯文本模式需要)
    """
    path = Path(path)
    content = path.read_text(encoding="utf-8")
    ext = path.suffix.lower()

    if ext == ".lrc":
        return parse_lrc(content)
    elif ext == ".srt":
        return parse_srt(content)
    else:
        # 尝试检测格式
        if re.search(r"\[\d{2}:\d{2}\.\d{2,3}\]", content):
            return parse_lrc(content)
        elif re.search(r"\d{2}:\d{2}:\d{2},\d{3}\s*-->", content):
            return parse_srt(content)
        else:
            return parse_plain_text(content, total_duration)
