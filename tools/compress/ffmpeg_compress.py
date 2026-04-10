"""
FFmpeg 视频无损/视觉无损压缩模块

支持平台：微信视频号、抖音等主流自媒体平台
核心参数：主要使用 H.264 (兼容最好) 或 H.265 (体积最小)，搭配合适的 CRF 和 preset。
"""
import subprocess
from pathlib import Path

from config import FFMPEG_BIN, OUTPUT_COMPRESS
from tools.common import generate_output_name, logger


def compress_video(
    input_video: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    codec: str = "libx264",
    crf: int = 24,
    preset: str = "slow",
    audio_bitrate: str = "128k",
) -> dict:
    """
    使用 FFmpeg 压缩视频

    Args:
        input_video: 输入视频路径
        output_path: 输出路径 (默认自动生成)
        codec: 视频编码器 (libx264 或 libx265)
        crf: 反映画质程度的常数(CRF), 控制视频质量 (数值越小画质越好, 体积越大)
        preset: x264/x265 的压缩效率预设 (slow = 体积更小, 但压缩时间更长)
        audio_bitrate: 音频比特率

    Returns:
        dict: 输出文件的信息字典
    """
    input_video = Path(input_video)
    if not input_video.is_file():
        raise FileNotFoundError(f"文件不存在: {input_video}")

    OUTPUT_COMPRESS.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        target_dir = Path(output_dir) if output_dir else OUTPUT_COMPRESS
        target_dir.mkdir(parents=True, exist_ok=True)
        crf_str = f"crf{crf}"
        output_name = generate_output_name(input_video.stem, ".mp4", tag=f"compressed_{codec}_{crf_str}")
        output_path = target_dir / output_name
    output_path = Path(output_path)

    # 对于各大自媒体平台，标准兼容设定:
    # 视频: yuv420p 色彩空间确保平台上传不会变色
    # 音频: AAC (高兼容性)
    # 选项: +faststart 优化平台网络流媒体缓冲
    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(input_video),
        "-c:v", codec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", audio_bitrate,
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(f"正在压缩: {input_video.name} | codec={codec}, crf={crf}, preset={preset}")

    # 获取原文件大小
    input_size_mb = input_video.stat().st_size / (1024 * 1024)

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 压缩失败:\n{result.stderr[-500:]}")

    if not output_path.is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    ratio = output_size_mb / input_size_mb if input_size_mb > 0 else 0

    logger.info(f"✅ 压缩完成: {output_path.name}")
    logger.info(f"   📊 体积变化: {input_size_mb:.2f} MB -> {output_size_mb:.2f} MB (压缩至原体积的: {ratio:.1%})")

    return {
        "output": str(output_path),
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "ratio": round(ratio, 4),
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="视频无损压缩工具")
    parser.add_argument("input", help="输入视频文件")
    parser.add_argument("-o", "--output", help="输出视频路径")
    parser.add_argument("--codec", default="libx264", choices=["libx264", "libx265"], help="视频编码器")
    parser.add_argument("--crf", type=int, default=24, help="压缩率 CRF (越低画质越好, 默认24)")
    args = parser.parse_args()

    compress_video(args.input, args.output, codec=args.codec, crf=args.crf)
