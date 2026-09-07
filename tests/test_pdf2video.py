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
    _split_sentence_into_phrases,
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
            subtitle_style="black_transparent",
        )
        assert ass_path.is_file()
        content = ass_path.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "Dialogue: 0," in content
        assert "单页精致字幕测试" in content
        # 黑字透明底
        assert "&H00000000" in content
        assert "&H00FFFFFF" in content

    def test_build_single_slide_ass_white_box(self, tmp_path: Path):
        ass_path = tmp_path / "slide_white_box.ass"
        build_single_slide_ass(
            text="白字半透明底框测试",
            duration=5.0,
            resolution=(1920, 1080),
            output_path=ass_path,
            subtitle_style="white_box",
        )
        assert ass_path.is_file()
        content = ass_path.read_text(encoding="utf-8")
        assert "[Script Info]" in content
        assert "Style: BgBox" in content
        assert "Dialogue: 0," in content
        assert "Dialogue: 1," in content
        assert "白字半透明底框测试" in content
        # 白字半透明黑底矢量圆角框 (Layer 0 画布 \p1 包含贝塞尔曲线 b，Layer 1 白字 &H00FFFFFF)
        assert r"\p1" in content
        assert r"\1a&H70&" in content
        assert "&H00FFFFFF" in content

    def test_build_single_slide_ass_with_sentences(self, tmp_path: Path):
        ass_path = tmp_path / "slide_multi.ass"
        sentences = [
            {"start": 0.0, "end": 2.5, "text": "第一句话讲解"},
            {"start": 2.5, "end": 6.0, "text": "第二句话内容更加详细"},
        ]
        build_single_slide_ass(
            text="整段文本回退",
            duration=6.5,
            resolution=(1920, 1080),
            output_path=ass_path,
            sentences=sentences,
            subtitle_style="black_transparent",
        )
        assert ass_path.is_file()
        content = ass_path.read_text(encoding="utf-8")
        assert "第一句话讲解" in content
        assert "第二句话内容更加详细" in content
        dialogues = [line for line in content.splitlines() if line.startswith("Dialogue:")]
        assert len(dialogues) == 2
        assert "0:00:00.00" in dialogues[0]
        assert "0:00:02.50" in dialogues[1]

    def test_build_single_slide_ass_overlap_prevention(self, tmp_path: Path):
        ass_path = tmp_path / "overlap_test.ass"
        # Simulate Edge-TTS slight timestamp overlap (e.g. 0.05s)
        sentences = [
            {"start": 0.1, "end": 5.05, "text": "第一句结束稍晚"},
            {"start": 5.0, "end": 10.0, "text": "第二句提前开始"},
        ]
        build_single_slide_ass(
            text="回退文本",
            duration=10.0,
            resolution=(1920, 1080),
            output_path=ass_path,
            sentences=sentences,
            subtitle_style="black_transparent",
        )
        content = ass_path.read_text(encoding="utf-8")
        dialogues = [line for line in content.splitlines() if line.startswith("Dialogue:")]
        assert len(dialogues) == 2
        assert "0:00:00.10,0:00:05.00" in dialogues[0]
        assert "0:00:05.00,0:00:10.00" in dialogues[1]
    def test_split_sentence_into_phrases(self):
        text = "第三步，批量导入商品明细，核对货物数量与金额无误后点击确认，单据状态变为待提交，生成电子仓单"
        phrases = _split_sentence_into_phrases(text, 0.0, 10.0)
        assert len(phrases) >= 3
        assert phrases[0]["start"] == 0.0
        assert phrases[-1]["end"] == 10.0
        for p in phrases:
            assert p["end"] > p["start"]
            assert len(p["text"]) > 0

    def test_build_single_slide_ass_double_line(self, tmp_path: Path):
        ass_path = tmp_path / "double_line.ass"
        text = "这是一句超长的文案测试，包含足够多的汉字用于触发双行折行排版测试效果，确保在卡片内居中"
        build_single_slide_ass(
            text=text,
            duration=6.0,
            resolution=(1920, 1080),
            output_path=ass_path,
            subtitle_style="white_box",
            subtitle_layout="double_line",
        )
        content = ass_path.read_text(encoding="utf-8")
        assert "\\N" in content
        assert "Style: BgBox" in content

    def test_build_single_slide_ass_single_line_scale(self, tmp_path: Path):
        ass_path = tmp_path / "single_line_scale.ass"
        # 极长句子（40+字）：坚决不折行（无 \\N），通过 \\fs 等比微缩字号
        text_long = "超长单行句子测试用于检测纯单行自适应字号排版是否绝对不换行并动态等比微缩字号"
        build_single_slide_ass(
            text=text_long,
            duration=6.0,
            resolution=(1920, 1080),
            output_path=ass_path,
            subtitle_style="white_box",
            subtitle_layout="single_line_scale",
        )
        content_long = ass_path.read_text(encoding="utf-8")
        assert "\\N" not in content_long
        assert "\\fs" in content_long
        assert "Style: BgBox" in content_long

        # 兼容旧参数名 auto_scale 和 fixed_bar
        ass_path_alias = tmp_path / "alias.ass"
        build_single_slide_ass(
            text=text_long,
            duration=6.0,
            resolution=(1920, 1080),
            output_path=ass_path_alias,
            subtitle_style="white_box",
            subtitle_layout="auto_scale",
        )
        content_alias = ass_path_alias.read_text(encoding="utf-8")
        assert "\\N" not in content_alias
        assert "\\fs" in content_alias




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

    def test_generate_pdf_video_pipeline_white_box(self, sample_pdf: Path, tmp_path: Path):
        from unittest.mock import MagicMock, patch

        from tools.pdf2video.generator import generate_pdf_video

        out_mp4 = tmp_path / "output_white_box.mp4"

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
                burn_subtitles=True,
                subtitle_style="white_box",
                show_progress=False,
            )

            assert out_mp4.is_file()
            assert res["slides_count"] == 2

