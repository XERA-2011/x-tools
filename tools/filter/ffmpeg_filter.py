"""
FFmpeg 滤镜效果模块

功能:
  - 8 种热门预设滤镜 (电影感/复古/赛博朋克/日系/黑白/暖色/冷色/高对比)
  - 支持视频和图片
  - 基于 FFmpeg 视频滤镜, 处理速度快

使用方式:
  python tools/filter/ffmpeg_filter.py video.mp4 -f cinematic
  python tools/filter/ffmpeg_filter.py image.jpg -f vintage
"""
import subprocess
from pathlib import Path

from config import FFMPEG_BIN, OUTPUT_FILTER, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from tools.common import generate_output_name, logger


# ============================================================
# 预设滤镜定义
# ============================================================
FILTER_PRESETS = {
    "cinematic": {
        "name": "🎬 电影感",
        "desc": "温暖色调, 略增对比, 暗角效果",
        "vf": "eq=contrast=1.2:brightness=-0.03:saturation=1.1,"
              "colorbalance=rs=0.06:gs=-0.02:bs=-0.08:rh=0.03:gh=-0.01:bh=-0.05,"
              "vignette=PI/4",
    },
    "vintage": {
        "name": "📷 复古胶片",
        "desc": "褪色暖调, 降低饱和度, 柔和对比",
        "vf": "eq=contrast=0.9:brightness=0.05:saturation=0.65,"
              "colorbalance=rs=0.12:gs=0.05:bs=-0.08:rm=0.08:gm=0.03:bm=-0.05,"
              "curves=m='0/0.05 0.5/0.48 1/0.92'",
    },
    "cyberpunk": {
        "name": "🌆 赛博朋克",
        "desc": "高对比, 青橙色调, 霓虹感",
        "vf": "eq=contrast=1.4:saturation=1.3:brightness=-0.02,"
              "colorbalance=rs=-0.08:gs=0.08:bs=0.2:rh=0.12:gh=-0.05:bh=0.15",
    },
    "japanese": {
        "name": "🌸 日系清新",
        "desc": "明亮通透, 低饱和, 柔和淡雅",
        "vf": "eq=brightness=0.08:contrast=0.95:saturation=0.75,"
              "colorbalance=rs=-0.03:gs=0.02:bs=0.08:rm=-0.02:gm=0.02:bm=0.06,"
              "curves=m='0/0.03 0.5/0.55 1/1'",
    },
    "bw": {
        "name": "⬛ 黑白经典",
        "desc": "去色, 增强对比和细节",
        "vf": "hue=s=0,eq=contrast=1.25:brightness=0.02",
    },
    "warm": {
        "name": "🔥 暖色调",
        "desc": "温暖金色氛围, 适合风景和人像",
        "vf": "colorbalance=rs=0.12:gs=0.05:bs=-0.1:rm=0.1:gm=0.03:bm=-0.08:rh=0.05:gh=0.02:bh=-0.05,"
              "eq=saturation=1.1:brightness=0.02",
    },
    "cool": {
        "name": "❄️ 冷色调",
        "desc": "冷蓝氛围, 适合都市和科技感",
        "vf": "colorbalance=rs=-0.08:gs=0.02:bs=0.15:rm=-0.05:gm=0.02:bm=0.1:rh=-0.03:gh=0.01:bh=0.08,"
              "eq=saturation=1.05:brightness=-0.02",
    },
    "high_contrast": {
        "name": "💎 高对比",
        "desc": "强烈对比, 色彩鲜明, 视觉冲击",
        "vf": "eq=contrast=1.5:saturation=1.15:brightness=-0.03,"
              "curves=m='0/0 0.25/0.15 0.75/0.85 1/1'",
    },
    "leica_m3": {
        "name": "📸 徕卡 M3",
        "desc": "经典胶片质感, 温暖中间调, 柔和褪色, 浓郁暗部",
        "vf": "eq=contrast=1.1:brightness=0.02:saturation=0.8,"
              "colorbalance=rs=0.1:gs=0.04:bs=-0.06:rm=0.06:gm=0.03:bm=-0.03:rh=0.03:gh=0.01:bh=-0.02,"
              "curves=m='0/0.04 0.15/0.12 0.5/0.5 0.85/0.83 1/0.93',"
              "vignette=PI/3.5",
    },
    "leica_m9": {
        "name": "📷 徕卡 M9",
        "desc": "CCD 传感器色彩, 鲜艳红绿, 通透细腻, 微暖色温",
        "vf": "eq=contrast=1.15:brightness=-0.01:saturation=1.05,"
              "colorbalance=rs=0.08:gs=0.02:bs=-0.04:rm=0.04:gm=0.03:bm=-0.02,"
              "curves=r='0/0 0.5/0.53 1/1':g='0/0 0.5/0.52 1/1':b='0/0 0.5/0.46 1/0.95',"
              "vignette=PI/5",
    },
}


def apply_filter(
    input_path: str | Path,
    output_path: str | Path | None = None,
    preset: str = "cinematic",
    intensity: float = 1.0,
    crf: int = 18,
) -> dict:
    """
    使用 FFmpeg 应用滤镜效果

    Args:
        input_path: 输入文件路径 (视频或图片)
        output_path: 输出路径 (默认自动生成)
        preset: 滤镜预设名称
        intensity: 滤镜强度 (0.0~1.0, 暂保留, 当前全强度)
        crf: 视频质量 CRF

    Returns:
        dict: {"output": str, "preset": str, ...}
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if preset not in FILTER_PRESETS:
        available = ", ".join(FILTER_PRESETS.keys())
        raise ValueError(f"未知滤镜: {preset} (可选: {available})")

    preset_info = FILTER_PRESETS[preset]
    vf = preset_info["vf"]

    suffix = input_path.suffix.lower()
    is_image = suffix in IMAGE_EXTENSIONS
    is_video = suffix in VIDEO_EXTENSIONS

    if not is_image and not is_video:
        raise ValueError(f"不支持的文件格式: {suffix}")

    # 输出路径
    OUTPUT_FILTER.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        out_ext = suffix if is_image else ".mp4"
        output_name = generate_output_name(input_path.stem, out_ext, tag=preset)
        output_path = OUTPUT_FILTER / output_name
    output_path = Path(output_path)

    # 构建 FFmpeg 命令
    cmd = [FFMPEG_BIN, "-y", "-i", str(input_path)]

    if is_image:
        # 图片: 单帧处理
        cmd += ["-vf", vf, "-q:v", "2", str(output_path)]
    else:
        # 视频: 保留音频
        cmd += [
            "-vf", vf,
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]

    logger.info(f"滤镜 [{preset_info['name']}]: {input_path.name}")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    if not output_path.is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ 滤镜完成: {output_path.name} ({output_size_mb:.1f} MB)")

    return {
        "output": str(output_path),
        "preset": preset,
        "preset_name": preset_info["name"],
        "size_mb": round(output_size_mb, 2),
    }


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FFmpeg 滤镜效果")
    parser.add_argument("input", help="输入文件路径 (视频或图片)")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument(
        "-f", "--filter", dest="preset", default="cinematic",
        help=f"滤镜预设 (可选: {', '.join(FILTER_PRESETS.keys())})",
    )
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF (默认: 18)")

    args = parser.parse_args()

    # 列出可用预设
    if args.preset == "list":
        print("\n可用滤镜预设:")
        for key, info in FILTER_PRESETS.items():
            print(f"  {info['name']:12s}  {key:16s}  {info['desc']}")
        print()
    else:
        result = apply_filter(
            input_path=args.input,
            output_path=args.output,
            preset=args.preset,
            crf=args.crf,
        )
        print(f"\n✅ {result['preset_name']} 完成")
        print(f"   输出: {result['output']} ({result['size_mb']} MB)")
