"""
tools/slideshow/generator.py 单元测试
"""
from pathlib import Path


class TestBuildTextsFromFilenames:
    """build_texts_from_filenames() 文件名 → 文案"""

    def test_basic(self):
        from tools.slideshow.generator import build_texts_from_filenames
        paths = [Path("hello_world.jpg"), Path("foo-bar.png")]
        result = build_texts_from_filenames(paths)
        assert result == ["hello world", "foo bar"]

    def test_no_extension(self):
        """应去掉扩展名"""
        from tools.slideshow.generator import build_texts_from_filenames
        result = build_texts_from_filenames([Path("test.jpg")])
        assert result == ["test"]

    def test_chinese_filename(self):
        from tools.slideshow.generator import build_texts_from_filenames
        result = build_texts_from_filenames([Path("新闻标题.png")])
        assert result == ["新闻标题"]

    def test_empty_list(self):
        from tools.slideshow.generator import build_texts_from_filenames
        assert build_texts_from_filenames([]) == []

    def test_mixed_separators(self):
        from tools.slideshow.generator import build_texts_from_filenames
        result = build_texts_from_filenames([Path("hello_world-2024.jpg")])
        assert result == ["hello world 2024"]


class TestLoadTextsFromCaptionFile:
    """load_texts_from_caption_file() 文案文件读取"""

    def test_basic(self, tmp_path):
        from tools.slideshow.generator import load_texts_from_caption_file
        f = tmp_path / "captions.txt"
        f.write_text("Line 1\nLine 2\nLine 3\n", encoding="utf-8")
        result = load_texts_from_caption_file(f)
        assert result == ["Line 1", "Line 2", "Line 3"]

    def test_skip_blank_lines(self, tmp_path):
        from tools.slideshow.generator import load_texts_from_caption_file
        f = tmp_path / "captions.txt"
        f.write_text("Line 1\n\n  \nLine 2\n", encoding="utf-8")
        result = load_texts_from_caption_file(f)
        assert result == ["Line 1", "Line 2"]

    def test_file_not_found(self, tmp_path):
        from tools.slideshow.generator import load_texts_from_caption_file
        result = load_texts_from_caption_file(tmp_path / "no_such_file.txt")
        assert result == []

    def test_chinese_content(self, tmp_path):
        from tools.slideshow.generator import load_texts_from_caption_file
        f = tmp_path / "captions.txt"
        f.write_text("第一条标题\n第二条标题\n", encoding="utf-8")
        result = load_texts_from_caption_file(f)
        assert result == ["第一条标题", "第二条标题"]

    def test_strips_whitespace(self, tmp_path):
        from tools.slideshow.generator import load_texts_from_caption_file
        f = tmp_path / "captions.txt"
        f.write_text("  spaced  \n  tabs\t\n", encoding="utf-8")
        result = load_texts_from_caption_file(f)
        assert result == ["spaced", "tabs"]
