"""
tools/subtitle/whisper_transcribe.py 单元测试

只测纯函数 (不需要 Whisper 模型)
"""


class TestFormatTimestamp:
    """_format_timestamp() SRT 时间戳格式化"""

    def test_zero(self):
        from tools.subtitle.whisper_transcribe import _format_timestamp
        assert _format_timestamp(0.0) == "00:00:00,000"

    def test_seconds(self):
        from tools.subtitle.whisper_transcribe import _format_timestamp
        assert _format_timestamp(5.5) == "00:00:05,500"

    def test_minutes(self):
        from tools.subtitle.whisper_transcribe import _format_timestamp
        assert _format_timestamp(90.0) == "00:01:30,000"

    def test_hours(self):
        from tools.subtitle.whisper_transcribe import _format_timestamp
        assert _format_timestamp(3661.123) == "01:01:01,123"

    def test_milliseconds_precision(self):
        from tools.subtitle.whisper_transcribe import _format_timestamp
        assert _format_timestamp(0.001) == "00:00:00,001"
        assert _format_timestamp(0.999) == "00:00:00,999"

    def test_large_value(self):
        """超过 24 小时也应正确"""
        from tools.subtitle.whisper_transcribe import _format_timestamp
        result = _format_timestamp(86400.0)  # 24 hours
        assert result == "24:00:00,000"


class TestSegmentsToSrt:
    """_segments_to_srt() Whisper segments → SRT 文本"""

    def test_single_segment(self):
        from tools.subtitle.whisper_transcribe import _segments_to_srt
        segments = [{"start": 0.0, "end": 5.0, "text": "Hello"}]
        result = _segments_to_srt(segments)
        assert "1\n00:00:00,000 --> 00:00:05,000\nHello" in result

    def test_multiple_segments(self):
        from tools.subtitle.whisper_transcribe import _segments_to_srt
        segments = [
            {"start": 0.0, "end": 3.0, "text": "First"},
            {"start": 3.0, "end": 6.0, "text": "Second"},
        ]
        result = _segments_to_srt(segments)
        lines = result.strip().split("\n")
        # 第一条: 序号 1
        assert lines[0] == "1"
        # 应有两条字幕
        assert "2\n" in result

    def test_empty_text_skipped(self):
        """空白文本的 segment 应被跳过"""
        from tools.subtitle.whisper_transcribe import _segments_to_srt
        segments = [
            {"start": 0.0, "end": 1.0, "text": "  "},
            {"start": 1.0, "end": 2.0, "text": "Valid"},
        ]
        result = _segments_to_srt(segments)
        assert "Valid" in result
        assert result.strip().startswith("1")

    def test_bilingual_segments(self):
        """含 translated_text 应输出双语"""
        from tools.subtitle.whisper_transcribe import _segments_to_srt
        segments = [{
            "start": 0.0, "end": 5.0,
            "text": "你好世界",
            "translated_text": "Hello World",
        }]
        result = _segments_to_srt(segments)
        assert "Hello World" in result
        assert "你好世界" in result

    def test_empty_segments(self):
        from tools.subtitle.whisper_transcribe import _segments_to_srt
        assert _segments_to_srt([]) == ""
