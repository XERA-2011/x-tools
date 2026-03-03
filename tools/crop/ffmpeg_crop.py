"""
视频/图片裁切模块

功能:
  - 按比例居中裁切 (1:1, 3:4, 4:3, 9:16, 16:9)
  - 自动检测源尺寸, 计算最大裁切区域
  - 支持视频和图片
  - 基于 FFmpeg crop 滤镜

使用方式:
  python tools/crop/ffmpeg_crop.py video.mp4 -r 3:4
  python tools/crop/ffmpeg_crop.py image.jpg -r 1:1
"""
import subprocess
from pathlib import Path

from config import FFMPEG_BIN, OUTPUT_CROP, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from tools.common import get_video_info, generate_output_name, logger


# ============================================================
# 预设比例
# ============================================================
ASPECT_RATIOS = {
    "1:1":   (1, 1),
    "3:4":   (3, 4),
    "4:3":   (4, 3),
    "9:16":  (9, 16),
    "16:9":  (16, 9),
}


def calc_crop(src_w: int, src_h: int, ratio: tuple[int, int]) -> tuple[int, int, int, int]:
    """
    计算居中裁切参数

    Args:
        src_w, src_h: 源尺寸
        ratio: 目标宽高比 (w_ratio, h_ratio)

    Returns:
        (crop_w, crop_h, x_offset, y_offset) — 均为偶数 (FFmpeg 要求)
    """
    w_ratio, h_ratio = ratio

    # 计算最大裁切区域
    if src_w / src_h > w_ratio / h_ratio:
        # 源更宽 → 以高度为基准, 裁两侧
        crop_h = src_h
        crop_w = int(src_h * w_ratio / h_ratio)
    else:
        # 源更高 → 以宽度为基准, 裁上下
        crop_w = src_w
        crop_h = int(src_w * h_ratio / w_ratio)

    # FFmpeg 要求偶数
    crop_w = crop_w // 2 * 2
    crop_h = crop_h // 2 * 2

    # 居中偏移
    x_offset = (src_w - crop_w) // 2
    y_offset = (src_h - crop_h) // 2

    return crop_w, crop_h, x_offset, y_offset


def crop_media(
    input_path: str | Path,
    output_path: str | Path | None = None,
    ratio: str = "3:4",
    crf: int = 18,
) -> dict:
    """
    裁切视频或图片到指定比例

    Args:
        input_path: 输入文件路径
        output_path: 输出路径 (默认自动生成)
        ratio: 目标比例 (如 "3:4", "1:1")
        crf: 视频质量 CRF

    Returns:
        dict: {"output": str, ...}
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if ratio not in ASPECT_RATIOS:
        available = ", ".join(ASPECT_RATIOS.keys())
        raise ValueError(f"不支持的比例: {ratio} (可选: {available})")

    suffix = input_path.suffix.lower()
    is_image = suffix in IMAGE_EXTENSIONS
    is_video = suffix in VIDEO_EXTENSIONS

    if not is_image and not is_video:
        raise ValueError(f"不支持的文件格式: {suffix}")

    # 获取源尺寸
    info = get_video_info(str(input_path))
    src_w = info.get("width", 0)
    src_h = info.get("height", 0)
    if src_w == 0 or src_h == 0:
        raise RuntimeError(f"无法读取源尺寸: {input_path}")

    # 计算裁切参数
    ratio_tuple = ASPECT_RATIOS[ratio]
    crop_w, crop_h, x_off, y_off = calc_crop(src_w, src_h, ratio_tuple)

    # 检查是否需要裁切
    if crop_w == src_w and crop_h == src_h:
        logger.info(f"⏭️  {input_path.name} 已是 {ratio} 比例, 跳过")
        return {"output": str(input_path), "skipped": True}

    vf = f"crop={crop_w}:{crop_h}:{x_off}:{y_off}"

    # 输出路径
    OUTPUT_CROP.mkdir(parents=True, exist_ok=True)
    ratio_tag = ratio.replace(":", "x")
    if output_path is None:
        out_ext = suffix if is_image else ".mp4"
        output_name = generate_output_name(input_path.stem, out_ext, tag=ratio_tag)
        output_path = OUTPUT_CROP / output_name
    output_path = Path(output_path)

    # 构建 FFmpeg 命令
    cmd = [FFMPEG_BIN, "-y", "-i", str(input_path)]

    if is_image:
        cmd += ["-vf", vf, "-q:v", "2", str(output_path)]
    else:
        cmd += [
            "-vf", vf,
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]

    logger.info(f"裁切 [{ratio}]: {input_path.name} ({src_w}x{src_h} → {crop_w}x{crop_h})")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    if not output_path.is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ 裁切完成: {output_path.name} ({crop_w}x{crop_h}, {output_size_mb:.1f} MB)")

    return {
        "output": str(output_path),
        "ratio": ratio,
        "src_size": f"{src_w}x{src_h}",
        "crop_size": f"{crop_w}x{crop_h}",
        "size_mb": round(output_size_mb, 2),
    }


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="视频/图片裁切")
    parser.add_argument("input", help="输入文件路径")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument(
        "-r", "--ratio", default="3:4",
        help=f"目标比例 (可选: {', '.join(ASPECT_RATIOS.keys())})",
    )
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF (默认: 18)")

    args = parser.parse_args()

    result = crop_media(
        input_path=args.input,
        output_path=args.output,
        ratio=args.ratio,
        crf=args.crf,
    )
    if result.get("skipped"):
        print(f"\n⏭️  已是目标比例, 无需裁切")
    else:
        print(f"\n✅ 裁切完成: {result['src_size']} → {result['crop_size']}")
        print(f"   输出: {result['output']} ({result['size_mb']} MB)")
