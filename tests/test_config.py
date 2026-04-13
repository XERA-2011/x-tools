"""
config.py 单元测试
"""
from config import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, clamp


class TestClamp:
    """clamp() 参数限制函数"""

    def test_within_range(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_below_minimum(self):
        assert clamp(-1.0, 0.0, 1.0) == 0.0

    def test_above_maximum(self):
        assert clamp(2.0, 0.0, 1.0) == 1.0

    def test_at_boundaries(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0
        assert clamp(1.0, 0.0, 1.0) == 1.0

    def test_integer_values(self):
        assert clamp(5, 1, 10) == 5
        assert clamp(0, 1, 10) == 1
        assert clamp(15, 1, 10) == 10

    def test_warning_logged_with_name(self, caplog):
        """超出范围且传入 name 时应输出警告"""
        import logging
        with caplog.at_level(logging.WARNING, logger="x-tools"):
            result = clamp(999, 0.0, 1.0, name="opacity")
        assert result == 1.0
        assert "opacity" in caplog.text

    def test_no_warning_within_range(self, caplog):
        """值在范围内时不应输出警告"""
        import logging
        with caplog.at_level(logging.WARNING, logger="x-tools"):
            clamp(0.5, 0.0, 1.0, name="opacity")
        assert caplog.text == ""


class TestExtensionConstants:
    """扩展名常量集合完整性"""

    def test_video_extensions_contain_common_formats(self):
        for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
            assert ext in VIDEO_EXTENSIONS, f"{ext} should be in VIDEO_EXTENSIONS"

    def test_image_extensions_contain_common_formats(self):
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
            assert ext in IMAGE_EXTENSIONS, f"{ext} should be in IMAGE_EXTENSIONS"

    def test_audio_extensions_contain_common_formats(self):
        for ext in [".mp3", ".wav", ".aac", ".flac", ".m4a"]:
            assert ext in AUDIO_EXTENSIONS, f"{ext} should be in AUDIO_EXTENSIONS"

    def test_no_overlap_between_sets(self):
        """视频、图片、音频扩展名不应有交集"""
        assert VIDEO_EXTENSIONS.isdisjoint(IMAGE_EXTENSIONS)
        assert VIDEO_EXTENSIONS.isdisjoint(AUDIO_EXTENSIONS)
        assert IMAGE_EXTENSIONS.isdisjoint(AUDIO_EXTENSIONS)

    def test_all_lowercase(self):
        """所有扩展名应为小写并以 . 开头"""
        for ext_set in [VIDEO_EXTENSIONS, IMAGE_EXTENSIONS, AUDIO_EXTENSIONS]:
            for ext in ext_set:
                assert ext.startswith("."), f"{ext} should start with '.'"
                assert ext == ext.lower(), f"{ext} should be lowercase"
