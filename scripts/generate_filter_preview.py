#!/usr/bin/env python3
"""
生成滤镜效果对比网格图

用法:
  python scripts/generate_filter_preview.py
  python scripts/generate_filter_preview.py --sample docs/assets/my_photo.jpg

功能:
  对样图分别应用 3 种滤镜预设, 拼成 2×2 网格对比图,
  输出到 docs/assets/filter_preview.jpg
"""
import subprocess
import sys
import tempfile
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import FFMPEG_BIN
from tools.filter.ffmpeg_filter import FILTER_PRESETS

# ============================================================
# 配置
# ============================================================
DEFAULT_SAMPLE = PROJECT_ROOT / "docs" / "assets" / "filter_sample.png"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "assets" / "filter_preview.jpg"
COLS = 2
THUMB_WIDTH = 480           # 每个缩略图宽度 (px)
LABEL_HEIGHT = 36           # 标签区域高度
FONT_SIZE = 20
BG_COLOR = (30, 30, 30)     # 深灰背景
LABEL_BG = (24, 24, 24)     # 标签背景
TEXT_COLOR = (240, 240, 240) # 标签文字色
PADDING = 6                 # 格子间距


def apply_filter_to_image(input_path: Path, output_path: Path, vf: str) -> bool:
    """用 FFmpeg 对图片应用滤镜"""
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(input_path),
        "-vf", vf,
        "-q:v", "2",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return result.returncode == 0


def generate_preview(sample_path: Path | None = None):
    from PIL import Image, ImageDraw, ImageFont

    sample = sample_path or DEFAULT_SAMPLE
    if not sample.is_file():
        print(f"❌ 样图不存在: {sample}")
        sys.exit(1)

    # 准备所有项: 原图 + 3 滤镜
    items: list[tuple[str, str, Path | None]] = [
        ("原图", "Original", None),
    ]
    for key, info in FILTER_PRESETS.items():
        items.append((info["name"], key, None))

    # 应用滤镜, 生成临时文件
    with tempfile.TemporaryDirectory(prefix="xtools_preview_") as tmpdir:
        tmpdir = Path(tmpdir)
        processed: list[tuple[str, Path]] = []

        for label, key, _ in items:
            if key == "Original":
                processed.append((label, sample))
                continue

            out = tmpdir / f"{key}.jpg"
            vf = FILTER_PRESETS[key]["vf"]
            print(f"  🎨 {label} ({key})...", end=" ", flush=True)
            if apply_filter_to_image(sample, out, vf):
                processed.append((label, out))
                print("✅")
            else:
                print("❌ 跳过")

        # 拼网格
        rows_count = -(-len(processed) // COLS)  # ceil division

        # 加载缩略图并计算统一尺寸
        thumbs: list[tuple[str, Image.Image]] = []
        for label, path in processed:
            img = Image.open(path).convert("RGB")
            ratio = THUMB_WIDTH / img.width
            thumb_h = int(img.height * ratio)
            img = img.resize((THUMB_WIDTH, thumb_h), Image.LANCZOS)
            thumbs.append((label, img))

        thumb_h = thumbs[0][1].height  # 所有缩略图高度一致 (原图决定)
        cell_w = THUMB_WIDTH + PADDING
        cell_h = thumb_h + LABEL_HEIGHT + PADDING

        canvas_w = COLS * cell_w + PADDING
        canvas_h = rows_count * cell_h + PADDING

        canvas = Image.new("RGB", (canvas_w, canvas_h), BG_COLOR)
        draw = ImageDraw.Draw(canvas)

        # 尝试加载字体
        try:
            font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", FONT_SIZE)
        except OSError:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc", FONT_SIZE)
            except OSError:
                font = ImageFont.load_default()

        for idx, (label, thumb) in enumerate(thumbs):
            row, col = divmod(idx, COLS)
            x = PADDING + col * cell_w
            y = PADDING + row * cell_h

            # 粘贴缩略图
            canvas.paste(thumb, (x, y))

            # 绘制标签背景
            label_y = y + thumb_h
            draw.rectangle(
                [x, label_y, x + THUMB_WIDTH, label_y + LABEL_HEIGHT],
                fill=LABEL_BG,
            )

            # 绘制标签文字 (居中)
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            tx = x + (THUMB_WIDTH - tw) // 2
            ty = label_y + (LABEL_HEIGHT - th) // 2
            draw.text((tx, ty), label, fill=TEXT_COLOR, font=font)

        # 保存
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(OUTPUT_PATH), "JPEG", quality=92)
        print(f"\n✅ 对比图已生成: {OUTPUT_PATH}")
        print(f"   尺寸: {canvas_w}x{canvas_h}, 大小: {OUTPUT_PATH.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="生成滤镜效果对比网格图")
    parser.add_argument("--sample", type=Path, default=None, help="样图路径 (默认使用 docs/assets/filter_sample.png)")
    args = parser.parse_args()

    print("🖼️  生成滤镜效果对比图...\n")
    generate_preview(args.sample)
