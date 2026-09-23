"""
视频对比与分屏拼接单元测试
"""
from pathlib import Path

import pytest
from PIL import Image

from tools.compare.ffmpeg_compare import (
    create_video_comparison,
    determine_layout_specs,
    extract_model_name_from_path,
    render_label_badge,
)


def test_extract_model_name_known_models():
    """测试知名 AI 视频模型名称解析"""
    cases = [
        ("鹈鹕骑单车Seedance2.0.mp4", "Seedance 2.0"),
        ("鹈鹕骑单车Veo3.1 Quality.mp4", "Veo 3.1 Quality"),
        ("鹈鹕骑单车Omni1.1 Flash.mp4", "Omni 1.1 Flash"),
        ("demo_sora_v2.mp4", "Sora_v2"),
        ("kling1.5_test.mp4", "Kling 1.5_test"),
        ("my_hailuo_01.mp4", "Hailuo_01"),
    ]
    for filename, expected in cases:
        result = extract_model_name_from_path(filename)
        assert expected.lower() in result.lower(), f"Failed on {filename}: got {result}"


def test_extract_model_name_fallback():
    """测试未知模型名称回退到文件名"""
    assert extract_model_name_from_path("custom_benchmark_run.mp4") == "custom_benchmark_run"


def test_determine_layout_specs():
    """测试布局规格计算"""
    # 3个横屏视频自动匹配 9:16 竖屏三等分
    w, h, slots = determine_layout_specs(3, "auto", {"width": 1920, "height": 1080})
    assert w == 1080
    assert h == 1920
    assert len(slots) == 3
    for slot_w, slot_h, x, y in slots:
        assert slot_w == 1080
        assert slot_h == 608
        assert x == 0

    # 4个视频网格
    w, h, slots = determine_layout_specs(4, "grid_2x2", {"width": 1920, "height": 1080})
    assert w == 1920
    assert h == 1080
    assert len(slots) == 4
    assert slots[0] == (960, 540, 0, 0)
    assert slots[1] == (960, 540, 960, 0)
    assert slots[2] == (960, 540, 0, 540)
    assert slots[3] == (960, 540, 960, 540)

    # 2个横屏视频横排
    w, h, slots = determine_layout_specs(2, "horizontal_stack", {"width": 1920, "height": 1080})
    assert w == 1920
    assert h == 1080
    assert len(slots) == 2

    # 无效布局报错
    with pytest.raises(ValueError):
        determine_layout_specs(2, "invalid_layout", {"width": 1920, "height": 1080})


def test_render_label_badge(tmp_path: Path):
    """测试 Badge 渲染功能"""
    badge_file = tmp_path / "test_badge.png"
    out_path = render_label_badge("Seedance 2.0", badge_file, font_size=32)

    assert out_path.exists()
    img = Image.open(out_path)
    assert img.mode == "RGBA"
    # 宽度和高度应为偶数
    assert img.width % 2 == 0
    assert img.height % 2 == 0
    assert img.width > 50
    assert img.height > 20


def test_create_video_comparison_validation():
    """测试输入参数校验"""
    # 视频数量不足 2 个
    with pytest.raises(ValueError, match="至少需要提供 2 个视频文件"):
        create_video_comparison(["video1.mp4"])

    # 视频数量超过 4 个
    with pytest.raises(ValueError, match="最多支持 4 个视频"):
        create_video_comparison(["v1.mp4", "v2.mp4", "v3.mp4", "v4.mp4", "v5.mp4"])

    # 文件不存在
    with pytest.raises(FileNotFoundError):
        create_video_comparison(["non_existent_1.mp4", "non_existent_2.mp4"])
