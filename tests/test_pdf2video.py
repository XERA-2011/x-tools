"""
PDF 转视频模块单元测试
"""
from pathlib import Path

import pytest
from PIL import Image

from tools.pdf2video.extractor import (
    clean_script_text,
    extract_script_from_pdf,
    find_companion_script,
    load_script,
    save_script,
)
from tools.pdf2video.generator import (
    _format_ass_time,
    _format_srt_time,
    _wrap_text_ass,
    build_single_slide_ass,
)
from tools.pdf2video.renderer import render_pdf_to_images


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """生成一个包含 2 页图片的测试 PDF 文件"""
    pdf_path = tmp_path / "test_doc.pdf"
    img1 = Image.new("RGB", (640, 360), color="blue")
    img2 = Image.new("RGB", (640, 360), color="green")

    img1.save(str(pdf_path), "PDF", save_all=True, append_images=[img2])
    return pdf_path


class TestExtractor:
    """文案提取与清洗测试"""

    def test_clean_script_text(self):
        assert clean_script_text("") == ""
        # 移除末尾页码
        raw = "入库仓单详情\n核对数量\n14 / 62"
        assert clean_script_text(raw) == "入库仓单详情\n核对数量"

        raw_tight = "步骤说明 02/62"
        assert clean_script_text(raw_tight) == "步骤说明"

    def test_extract_pdf_pages(self, sample_pdf: Path):
        script = extract_script_from_pdf(sample_pdf)
        assert len(script) == 2
        assert script[0]["page"] == 1
        assert script[1]["page"] == 2

    def test_save_and_load_script(self, tmp_path: Path):
        data = [
            {"page": 1, "text": "第一页文案", "source": "pdf_text"},
            {"page": 2, "text": "第二页文案", "source": "pdf_text"},
        ]
        out_json = tmp_path / "script.json"
        save_script(data, out_json)

        assert out_json.is_file()
        loaded = load_script(out_json)
        assert loaded == data

    def test_find_companion_script(self, tmp_path: Path):
        pdf = tmp_path / "demo.pdf"
        pdf.touch()
        txt = tmp_path / "demo.txt"
        txt.touch()

        found = find_companion_script(pdf)
        assert found == txt


class TestFormatting:
    """时间戳与字幕格式化测试"""

    def test_format_srt_time(self):
        assert _format_srt_time(0.0) == "00:00:00,000"
        assert _format_srt_time(65.5) == "00:01:05,500"
        assert _format_srt_time(3661.123) == "01:01:01,123"

    def test_format_ass_time(self):
        assert _format_ass_time(0.0) == "0:00:00.00"
        assert _format_ass_time(65.5) == "0:01:05.50"
        assert _format_ass_time(3661.12) == "1:01:01.12"

    def test_wrap_text_ass_limits_lines(self):
        short_txt = "短文本"
        wrapped = _wrap_text_ass(short_txt, max_chars_per_line=10, max_lines=2)
        assert wrapped == "短文本"

        # 超过 2 行时被安全截断，防止遮挡画面
        long_txt = "第一行内容超长测试\n第二行内容超长测试\n第三行应当被截断以避免遮挡\n第四行"
        wrapped = _wrap_text_ass(long_txt, max_chars_per_line=8, max_lines=2)
        assert wrapped.count(r"\N") <= 1

    def test_build_single_slide_ass(self, tmp_path: Path):
        ass_path = tmp_path / "slide_test.ass"
        build_single_slide_ass(
            text="单页精致字幕测试",
            duration=5.0,
            resolution=(1920, 1080),
            output_path=ass_path,
        )
        assert ass_path.is_file()
        content = ass_path.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "Dialogue: 0," in content
        assert "单页精致字幕测试" in content


class TestRenderer:
    """页面渲染测试"""

    def test_render_pdf_to_images(self, sample_pdf: Path, tmp_path: Path):
        out_dir = tmp_path / "rendered_imgs"
        images = render_pdf_to_images(sample_pdf, out_dir, scale=1.0)

        assert len(images) == 2
        for img_path in images:
            assert img_path.is_file()
            with Image.open(img_path) as im:
                assert im.width > 0
                assert im.height > 0


class TestGeneratePdfVideo:
    """PDF 视频生成流水线及单进度条逻辑测试"""

    def test_generate_pdf_video_pipeline(self, sample_pdf: Path, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        from tools.pdf2video.generator import generate_pdf_video

        out_mp4 = tmp_path / "output.mp4"

        with (
            patch("tools.pdf2video.generator._generate_slide_tts") as mock_tts,
            patch("tools.pdf2video.generator.get_video_info") as mock_info,
            patch("tools.pdf2video.generator.subprocess.run") as mock_subproc,
        ):
            async def _fake_tts(text, voice_id, path, sem=None, on_complete=None):
                Path(path).write_bytes(b"fake_mp3_data")
                if on_complete:
                    on_complete()

            mock_tts.side_effect = _fake_tts
            mock_info.return_value = {"duration": 2.5}

            def _fake_subproc(cmd, **kwargs):
                out = cmd[-1]
                Path(out).write_bytes(b"dummy_video")
                return MagicMock(returncode=0)

            mock_subproc.side_effect = _fake_subproc

            res = generate_pdf_video(
                pdf_path=sample_pdf,
                output_path=out_mp4,
                custom_script=[
                    {"page": 1, "text": "第一页介绍"},
                    {"page": 2, "text": "第二页介绍"},
                ],
                show_progress=True,
            )

            assert out_mp4.is_file()
            assert res["slides_count"] == 2
            assert res["output"] == str(out_mp4)
            assert res["srt"] is not None
            assert Path(res["srt"]).is_file()

