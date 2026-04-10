"""
tools/common.py 单元测试

覆盖: generate_output_name, parse_region, calc_overlay_position,
       orient_resolution, scan_videos, scan_images
"""
import re
from pathlib import Path

import pytest


class TestGenerateOutputName:
    """generate_output_name() 文件名生成"""

    def test_basic_format(self):
        from tools.common import generate_output_name
        name = generate_output_name("video", ".mp4")
        assert name.startswith("video_")
        assert name.endswith(".mp4")

    def test_with_tag(self):
        from tools.common import generate_output_name
        name = generate_output_name("clip", ".mp4", tag="compressed")
        assert "compressed" in name
        assert name.endswith(".mp4")

    def test_unique_names(self):
        """连续生成的名称应不重复"""
        from tools.common import generate_output_name
        names = {generate_output_name("test", ".mp4") for _ in range(20)}
        assert len(names) == 20

    def test_timestamp_pattern(self):
        """名称应包含 MMDD_HHMMSS 时间戳格式"""
        from tools.common import generate_output_name
        name = generate_output_name("x", ".mp4")
        # 匹配 MMDD_HHMMSS_mmm_xxxx 模式
        assert re.search(r"\d{4}_\d{6}_\d{3}_[0-9a-f]{4}", name)


class TestParseRegion:
    """parse_region() 区域字符串解析"""

    def test_valid_region(self):
        from tools.common import parse_region
        result = parse_region("10,20,100,200")
        assert result == (10, 20, 100, 200)

    def test_with_spaces(self):
        from tools.common import parse_region
        result = parse_region("10, 20, 100, 200")
        assert result == (10, 20, 100, 200)

    def test_invalid_format_too_few(self):
        from tools.common import parse_region
        with pytest.raises(ValueError):
            parse_region("10,20,100")

    def test_invalid_format_too_many(self):
        from tools.common import parse_region
        with pytest.raises(ValueError):
            parse_region("10,20,100,200,300")

    def test_non_numeric(self):
        from tools.common import parse_region
        with pytest.raises(ValueError):
            parse_region("a,b,c,d")

    def test_negative_values(self):
        """负值虽然无意义, 但解析应不报错"""
        from tools.common import parse_region
        result = parse_region("-10,-20,100,200")
        assert result == (-10, -20, 100, 200)


class TestCalcOverlayPosition:
    """calc_overlay_position() 叠加层坐标计算"""

    def test_center(self):
        from tools.common import calc_overlay_position
        x, y = calc_overlay_position(1920, 1080, 100, 50, "center", margin=0)
        assert x == (1920 - 100) // 2
        assert y == (1080 - 50) // 2

    def test_top_left(self):
        from tools.common import calc_overlay_position
        x, y = calc_overlay_position(1920, 1080, 100, 50, "top-left", margin=10)
        assert x == 10
        assert y == 10

    def test_bottom_right(self):
        from tools.common import calc_overlay_position
        x, y = calc_overlay_position(1920, 1080, 100, 50, "bottom-right", margin=10)
        assert x == 1920 - 100 - 10
        assert y == 1080 - 50 - 10

    def test_tuple_passthrough(self):
        """传入坐标元组应直接返回"""
        from tools.common import calc_overlay_position
        x, y = calc_overlay_position(1920, 1080, 100, 50, (42, 84), margin=0)
        assert x == 42
        assert y == 84

    def test_unknown_position_falls_back(self):
        """未知位置名称应回退到 bottom-right"""
        from tools.common import calc_overlay_position
        x, y = calc_overlay_position(1920, 1080, 100, 50, "nowhere", margin=10)
        assert x == 1920 - 100 - 10
        assert y == 1080 - 50 - 10


class TestOrientResolution:
    """orient_resolution() 竖屏自动翻转"""

    def test_portrait_video_horizontal_target(self):
        """竖屏视频 + 横屏目标 → 翻转"""
        from tools.common import orient_resolution
        w, h = orient_resolution(1080, 1920, 1920, 1080)
        assert w == 1080
        assert h == 1920

    def test_landscape_video_horizontal_target(self):
        """横屏视频 + 横屏目标 → 不变"""
        from tools.common import orient_resolution
        w, h = orient_resolution(1920, 1080, 1920, 1080)
        assert w == 1920
        assert h == 1080

    def test_portrait_video_portrait_target(self):
        """竖屏视频 + 竖屏目标 → 不变"""
        from tools.common import orient_resolution
        w, h = orient_resolution(1080, 1920, 1080, 1920)
        assert w == 1080
        assert h == 1920

    def test_square_video(self):
        """正方形视频不应翻转"""
        from tools.common import orient_resolution
        w, h = orient_resolution(1080, 1080, 1920, 1080)
        assert w == 1920
        assert h == 1080


class TestScanFiles:
    """scan_videos / scan_images 文件扫描"""

    def test_scan_videos_finds_mp4(self, tmp_path):
        from tools.common import scan_videos
        (tmp_path / "test.mp4").touch()
        (tmp_path / "test.txt").touch()
        (tmp_path / "photo.jpg").touch()
        result = scan_videos(tmp_path)
        assert len(result) == 1
        assert result[0].name == "test.mp4"

    def test_scan_videos_empty_dir(self, tmp_path):
        from tools.common import scan_videos
        result = scan_videos(tmp_path)
        assert result == []

    def test_scan_videos_nonexistent_dir(self):
        from tools.common import scan_videos
        result = scan_videos(Path("/nonexistent/dir"))
        assert result == []

    def test_scan_images_finds_jpg_png(self, tmp_path):
        from tools.common import scan_images
        (tmp_path / "photo.jpg").touch()
        (tmp_path / "logo.png").touch()
        (tmp_path / "video.mp4").touch()
        result = scan_images(tmp_path)
        assert len(result) == 2

    def test_scan_videos_recursive(self, tmp_path):
        from tools.common import scan_videos
        sub = tmp_path / "subdir"
        sub.mkdir()
        (sub / "clip.mp4").touch()
        (tmp_path / "main.mp4").touch()

        # 非递归: 只找到根目录
        flat = scan_videos(tmp_path, recursive=False)
        assert len(flat) == 1

        # 递归: 找到根 + 子目录
        deep = scan_videos(tmp_path, recursive=True)
        assert len(deep) == 2

    def test_scan_videos_sorted(self, tmp_path):
        """结果应按文件名排序"""
        from tools.common import scan_videos
        for name in ["c.mp4", "a.mp4", "b.mp4"]:
            (tmp_path / name).touch()
        result = scan_videos(tmp_path)
        names = [r.name for r in result]
        assert names == ["a.mp4", "b.mp4", "c.mp4"]

    def test_scan_videos_case_insensitive_extension(self, tmp_path):
        """大小写混合的扩展名也应被识别"""
        from tools.common import scan_videos
        (tmp_path / "upper.MP4").touch()
        (tmp_path / "mixed.Mp4").touch()
        result = scan_videos(tmp_path)
        assert len(result) == 2
