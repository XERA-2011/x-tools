"""
画面对比与分屏拼接核心引擎

功能:
  - 将多个视频 (2~4 个) 同屏并联播放 (对比展示、模型横评 PK)
  - 自动检测并统一以最短视频时长截断对齐
  - 自动或手动为每个视频右上角打上模型名称 / 算法名称胶囊角标
  - 支持多种排版布局:
      - 竖屏 9:16 (1080x1920) 叠排 (适合抖音/TikTok/Shorts 移动端模型对比)
      - 宽屏 16:9 (1920x1080) 横排
      - 网格 2x2 (四宫格)
  - 多种音频策略 (静音 / 保留第一路 / 多路混音 / 自选 BGM)
"""
from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont

from config import FFMPEG_BIN, OUTPUT_COMPARE
from tools.add_watermark.text_watermark import _find_cjk_font
from tools.common import (
    ProcessResult,
    generate_output_name,
    get_video_info,
    logger,
    run_ffmpeg_with_progress,
)

# 支持的布局类型
LayoutType = Literal["vertical_stack", "horizontal_stack", "grid_2x2", "auto"]
AudioMode = Literal["first", "mute", "mix", "bgm"]


def extract_model_name_from_path(video_path: str | Path) -> str:
    """
    从视频文件路径/名称中智能提取模型或算法名称

    例如:
        - "鹈鹕骑单车Seedance2.0.mp4" -> "Seedance 2.0"
        - "鹈鹕骑单车Veo3.1 Quality.mp4" -> "Veo 3.1 Quality"
        - "鹈鹕骑单车Omni1.1 Flash.mp4" -> "Omni 1.1 Flash"
        - "kling_v1.5_pro.mp4" -> "Kling V1.5 Pro"
    """
    stem = Path(video_path).stem

    # 1. 尝试匹配常见知名模型前缀及其后续版本号
    known_models = [
        "seedance", "veo", "omni", "sora", "kling", "可灵", "hailuo", "海螺",
        "runway", "gen-3", "gen3", "pika", "luma", "flux", "midjourney",
        "sd", "stable diffusion", "wan", "vidu", "minimax", "hunyuan", "cogvideo"
    ]
    for model in known_models:
        pattern = rf"(?i)({re.escape(model)}[\s\-_.]*[\w\s\-_.]*)"
        match = re.search(pattern, stem)
        if match:
            extracted = match.group(1).strip()
            # 格式化: 如果字母和数字紧挨着 (例如 Seedance2.0 -> Seedance 2.0)
            extracted = re.sub(r"([a-zA-Z]{3,})(\d)", r"\1 \2", extracted)
            return extracted

    # 2. 尝试提取末尾的英文+数字串 (中文前缀后跟随的英文模型名)
    m = re.search(r"([A-Za-z0-9][A-Za-z0-9\.\-_ ]*)$", stem)
    if m:
        candidate = m.group(1).strip()
        candidate = re.sub(r"([a-zA-Z]{3,})(\d)", r"\1 \2", candidate)
        if len(candidate) >= 2:
            return candidate

    # 3. 兜底直接返回 stem
    return stem


def render_label_badge(
    text: str,
    output_png: str | Path,
    font_size: int = 32,
    bg_color: tuple[int, int, int, int] = (20, 20, 25, 210),
    text_color: tuple[int, int, int, int] = (255, 255, 255, 245),
    border_color: tuple[int, int, int, int] = (255, 255, 255, 60),
    radius: int = 14,
) -> Path:
    """
    使用 Pillow 渲染高质量圆角胶囊半透明 Badge 标签图像

    Args:
        text: 标签文字
        output_png: 输出 PNG 路径
        font_size: 字体大小
        bg_color: 背景 RGBA 颜色
        text_color: 文字 RGBA 颜色
        border_color: 边框 RGBA 颜色
        radius: 圆角半径

    Returns:
        Path: 输出的 PNG 图像路径
    """
    font_path = _find_cjk_font()
    try:
        if font_path and Path(font_path).exists():
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    # 计算文字边界
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    pad_x = int(font_size * 0.7)
    pad_y = int(font_size * 0.35)

    badge_w = text_w + pad_x * 2
    badge_h = text_h + pad_y * 2

    # 偶数宽度和高度利于编码
    badge_w = (badge_w + 1) // 2 * 2
    badge_h = (badge_h + 1) // 2 * 2

    img = Image.new("RGBA", (badge_w, badge_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 绘制半透明圆角矩形胶囊底衬
    draw.rounded_rectangle(
        [(0, 0), (badge_w - 1, badge_h - 1)],
        radius=radius,
        fill=bg_color,
        outline=border_color,
        width=1,
    )

    # 绘制轻微阴影以增强可读性
    text_x = pad_x - bbox[0]
    text_y = pad_y - bbox[1]
    draw.text((text_x + 1, text_y + 1), text, font=font, fill=(0, 0, 0, 110))
    draw.text((text_x, text_y), text, font=font, fill=text_color)

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_png), "PNG")
    return output_png


def determine_layout_specs(
    num_videos: int,
    layout: LayoutType,
    first_video_info: dict,
) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    """
    计算输出画布尺寸及每个子视频的槽位尺寸与放置坐标 (slot_w, slot_h, x, y)

    Returns:
        (canvas_w, canvas_h, slots)
        slots 为 [(slot_w, slot_h, x, y), ...]
    """
    aspect = first_video_info.get("width", 16) / max(first_video_info.get("height", 9), 1)

    # 自动推断最佳布局
    if layout == "auto":
        if num_videos == 3 and aspect > 1.3:
            # 3个横屏视频 -> 强烈推荐 9:16 竖屏三等分叠排
            layout = "vertical_stack"
        elif num_videos == 4:
            layout = "grid_2x2"
        elif num_videos == 2 and aspect > 1.3:
            layout = "vertical_stack"
        else:
            layout = "horizontal_stack"

    if layout == "vertical_stack":
        # 目标竖屏画布: 1080 x 1920 (9:16)
        canvas_w = 1080
        canvas_h = 1920

        if num_videos == 3:
            slot_w = 1080
            slot_h = 608  # 接近 16:9 (1080 * 9 / 16 = 607.5)
            # 24px 间隙: 24 + 608 + 24 + 608 + 24 + 608 + 24 = 1920
            slots = [
                (slot_w, slot_h, 0, 24),
                (slot_w, slot_h, 0, 656),
                (slot_w, slot_h, 0, 1288),
            ]
        elif num_videos == 2:
            slot_w = 1080
            slot_h = 608
            # 居中对称摆放: 上下各留 240px，中间间距 224px
            slots = [
                (slot_w, slot_h, 0, 240),
                (slot_w, slot_h, 0, 1072),
            ]
        else:
            # 通用 N 等分
            slot_w = 1080
            slot_h = int(1920 / num_videos) // 2 * 2
            slots = [(slot_w, slot_h, 0, i * slot_h) for i in range(num_videos)]

    elif layout == "horizontal_stack":
        # 目标横屏画布: 1920 x 1080 (16:9)
        canvas_w = 1920
        canvas_h = 1080

        if num_videos == 2:
            slot_w = 940
            slot_h = 528
            y = (1080 - slot_h) // 2
            slots = [
                (slot_w, slot_h, 13, y),
                (slot_w, slot_h, 967, y),
            ]
        elif num_videos == 3:
            slot_w = 620
            slot_h = 348
            y = (1080 - slot_h) // 2
            slots = [
                (slot_w, slot_h, 15, y),
                (slot_w, slot_h, 650, y),
                (slot_w, slot_h, 1285, y),
            ]
        else:
            slot_w = int(1920 / num_videos) // 2 * 2
            slot_h = int(slot_w * 9 / 16) // 2 * 2
            y = (1080 - slot_h) // 2
            slots = [(slot_w, slot_h, i * slot_w, y) for i in range(num_videos)]

    elif layout == "grid_2x2":
        canvas_w = 1920
        canvas_h = 1080
        slot_w = 960
        slot_h = 540
        slots = [
            (slot_w, slot_h, 0, 0),
            (slot_w, slot_h, 960, 0),
            (slot_w, slot_h, 0, 540),
            (slot_w, slot_h, 960, 540),
        ][:num_videos]
    else:
        raise ValueError(f"不支持的布局类型: {layout}")

    return canvas_w, canvas_h, slots


def create_video_comparison(
    video_paths: list[str | Path],
    labels: list[str] | None = None,
    layout: LayoutType = "auto",
    duration_mode: Literal["shortest", "longest"] = "shortest",
    audio_mode: AudioMode = "first",
    music_path: str | Path | None = None,
    music_volume: float = 0.3,
    output_path: str | Path | None = None,
    crf: int = 19,
) -> ProcessResult:
    """
    合成多视频对比同屏分屏视频

    Args:
        video_paths: 输入视频路径列表 (2~4 个)
        labels: 各视频右上角标注文本 (为空则自动从文件名提取)
        layout: 布局模式 ("vertical_stack", "horizontal_stack", "grid_2x2", "auto")
        duration_mode: 时长对齐策略 ("shortest" 以最短为准截断, "longest" 冻结末帧)
        audio_mode: 音频处理模式 ("first" 保留第一路, "mute" 静音, "mix" 混音, "bgm" 背景音乐)
        music_path: 背景音乐文件路径 (当 audio_mode="bgm" 时使用)
        music_volume: 背景音乐音量 (0.0~1.0)
        output_path: 输出文件路径 (可选)
        crf: H.264 编码质量 (越低画质越好, 默认 19)

    Returns:
        ProcessResult: 包含 output, duration, size_mb 等
    """
    if not video_paths or len(video_paths) < 2:
        raise ValueError("视频对比功能至少需要提供 2 个视频文件")
    if len(video_paths) > 4:
        raise ValueError("视频对比功能目前最多支持 4 个视频同时并联对比")

    video_paths = [Path(p).resolve() for p in video_paths]
    for p in video_paths:
        if not p.exists():
            raise FileNotFoundError(f"输入视频不存在: {p}")

    # 读取所有视频元数据
    infos = [get_video_info(p) for p in video_paths]
    durations = [info.get("duration", 0) for info in infos]
    target_fps = round(max((info.get("fps", 24.0) for info in infos), default=24.0), 2)

    # 确定整体视频时长
    if duration_mode == "shortest":
        final_duration = min(d for d in durations if d > 0)
    else:
        final_duration = max(durations)

    logger.info(
        f"开始处理 {len(video_paths)} 个视频对比合成, 对齐时长={final_duration:.2f}s, 目标帧率={target_fps}fps"
    )

    # 自动解析标签
    if labels is None:
        labels = [extract_model_name_from_path(p) for p in video_paths]
    elif len(labels) < len(video_paths):
        # 补全缺失的标签
        for i in range(len(labels), len(video_paths)):
            labels.append(extract_model_name_from_path(video_paths[i]))

    # 计算布局尺寸与槽位
    canvas_w, canvas_h, slots = determine_layout_specs(len(video_paths), layout, infos[0])

    # 准备临时目录存放 Badge 图片
    temp_dir = Path(tempfile.mkdtemp(prefix="xtools_compare_"))
    badge_paths: list[Path] = []

    try:
        # 生成每个视频对应的右上角胶囊角标
        for i, text in enumerate(labels):
            badge_file = temp_dir / f"badge_{i}.png"
            render_label_badge(text, badge_file, font_size=32)
            badge_paths.append(badge_file)

        # 构建 FFmpeg 命令
        cmd = [FFMPEG_BIN, "-y"]

        # 输入各个视频
        for vp in video_paths:
            cmd += ["-i", str(vp)]

        # 输入各个 Badge 图片
        badge_start_idx = len(video_paths)
        for bp in badge_paths:
            cmd += ["-i", str(bp)]

        # 如果有 BGM 音频输入
        bgm_input_idx = None
        if audio_mode == "bgm" and music_path:
            bgm_input_idx = badge_start_idx + len(badge_paths)
            cmd += ["-i", str(music_path)]

        # 构建 Filtergraph
        filter_parts: list[str] = []

        # 1. 黑色背景底板
        filter_parts.append(
            f"color=c=black:s={canvas_w}x{canvas_h}:d={final_duration}:r={target_fps}[base]"
        )

        # 2. 对各个输入视频进行裁切、缩放，并把角标贴在各自画面右上角
        subvideo_nodes: list[str] = []
        for i in range(len(video_paths)):
            slot_w, slot_h, _, _ = slots[i]
            badge_idx = badge_start_idx + i

            # 先按最短时长裁剪并统一缩放到 slot 尺寸 (保持宽高比填充并裁剪居中或 force_original_aspect_ratio)
            v_prep = (
                f"[{i}:v]trim=0:{final_duration},setpts=PTS-STARTPTS,"
                f"scale={slot_w}:{slot_h}:force_original_aspect_ratio=decrease,"
                f"pad={slot_w}:{slot_h}:(ow-iw)/2:(oh-ih)/2:black[v{i}_scaled]"
            )
            filter_parts.append(v_prep)

            # 将右上角 Badge 贴附到该子视频右上角 (留白边距 24px, 顶部边距 16px)
            v_badge = (
                f"[v{i}_scaled][{badge_idx}:v]overlay=W-w-24:16[v{i}_labeled]"
            )
            filter_parts.append(v_badge)
            subvideo_nodes.append(f"[v{i}_labeled]")

        # 3. 将各个子画面依次贴到底板 [base] 上
        current_canvas = "[base]"
        for i in range(len(video_paths)):
            _, _, pos_x, pos_y = slots[i]
            next_canvas = f"[stage_{i}]" if i < len(video_paths) - 1 else "[outv]"
            filter_parts.append(
                f"{current_canvas}{subvideo_nodes[i]}overlay={pos_x}:{pos_y}{next_canvas}"
            )
            current_canvas = next_canvas

        # 4. 音频处理
        has_audio_out = False
        if audio_mode == "mute":
            # 无音频输出
            pass
        elif audio_mode == "bgm" and bgm_input_idx is not None:
            filter_parts.append(
                f"[{bgm_input_idx}:a]atrim=0:{final_duration},asetpts=PTS-STARTPTS,"
                f"volume={music_volume}[outa]"
            )
            has_audio_out = True
        elif audio_mode == "mix":
            # 混合多路音频
            audio_inputs: list[str] = []
            for i in range(len(video_paths)):
                filter_parts.append(
                    f"[{i}:a]atrim=0:{final_duration},asetpts=PTS-STARTPTS[a{i}]"
                )
                audio_inputs.append(f"[a{i}]")
            filter_parts.append(
                f"{''.join(audio_inputs)}amix=inputs={len(audio_inputs)}:duration=first:dropout_transition=2[outa]"
            )
            has_audio_out = True
        else:
            # 默认 "first": 只保留第一路视频的原声
            filter_parts.append(
                f"[0:a]atrim=0:{final_duration},asetpts=PTS-STARTPTS[outa]"
            )
            has_audio_out = True

        filter_complex_str = ";".join(filter_parts)

        # 确定输出路径
        OUTPUT_COMPARE.mkdir(parents=True, exist_ok=True)
        if output_path is None:
            output_name = generate_output_name(
                "compare", ".mp4", tag=f"{len(video_paths)}in1"
            )
            output_path = OUTPUT_COMPARE / output_name
        output_path = Path(output_path)

        cmd += ["-filter_complex", filter_complex_str]
        cmd += ["-map", "[outv]"]
        if has_audio_out:
            cmd += ["-map", "[outa]"]
            cmd += ["-c:a", "aac", "-b:a", "192k"]
        else:
            cmd += ["-an"]

        cmd += [
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", "medium",
            "-pix_fmt", "yuv420p",
            "-t", str(final_duration),
            str(output_path),
        ]

        logger.info(f"正在渲染对比视频 → {output_path.name}")
        run_ffmpeg_with_progress(
            cmd,
            duration=final_duration,
            desc="🔲 画面对比视频渲染中",
        )

        size_mb = round(output_path.stat().st_size / (1024 * 1024), 2)
        logger.info(f"对比视频渲染完成: {output_path} ({size_mb} MB)")

        return ProcessResult(
            output=str(output_path),
            duration=round(final_duration, 2),
            size_mb=size_mb,
            frames_processed=int(final_duration * target_fps),
            skipped=False,
        )

    finally:
        # 清理临时角标图片目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
