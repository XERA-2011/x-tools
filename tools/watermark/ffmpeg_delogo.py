"""
FFmpeg delogo 快速去水印模块

方法说明:
  - 使用 FFmpeg 内置的 delogo 滤镜
  - 利用水印边缘周围的色彩/纹理梯度来重建被覆盖区域
  - 速度极快 (纯 C 流水线, 比 OpenCV 逐帧快 10~50 倍)
  - 效果优于 OpenCV inpaint, 特别适合角落固定位置的纯色 Logo

使用方式:
  python tools/watermark/ffmpeg_delogo.py video.mp4 --region 10,10,200,60
"""
from pathlib import Path

from config import FFMPEG_BIN, OUTPUT_WATERMARK
from tools.common import generate_output_name, get_video_info, logger, run_ffmpeg_with_progress


def _scale_region(
    x1: int, y1: int, x2: int, y2: int,
    ref_width: int, ref_height: int,
    target_width: int, target_height: int,
) -> tuple[int, int, int, int]:
    """按比例缩放水印区域坐标"""
    if ref_width <= 0 or ref_height <= 0:
        return x1, y1, x2, y2
    sx = target_width / ref_width
    sy = target_height / ref_height
    if abs(sx - 1.0) < 0.01 and abs(sy - 1.0) < 0.01:
        return x1, y1, x2, y2
    return int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)


def remove_watermark_delogo(
    video_path: str | Path,
    regions: list[tuple[int, int, int, int]] | None = None,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    ref_width: int = 0,
    ref_height: int = 0,
    show_mode: int = 0,
    crf: int = 18,
) -> dict:
    """
    使用 FFmpeg delogo 滤镜去除视频水印

    Args:
        video_path: 输入视频路径
        regions: 水印矩形区域列表 [(x1,y1,x2,y2), ...]
        output_path: 输出路径, 为 None 时自动生成
        output_dir: 批量输出目录
        ref_width: ROI 坐标的参考分辨率宽度 (0=不缩放)
        ref_height: ROI 坐标的参考分辨率高度 (0=不缩放)
        show_mode: 0=正常输出, 1=显示边界 (调试用)
        crf: 输出视频质量

    Returns:
        dict: {"output": str, ...}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if not regions:
        raise ValueError("必须指定至少一个水印区域 (regions)")

    # 获取视频信息
    info = get_video_info(video_path)
    vid_w = info.get("width", 0)
    vid_h = info.get("height", 0)
    duration = info.get("duration", 0)

    if vid_w == 0 or vid_h == 0:
        raise RuntimeError(f"无法读取视频尺寸: {video_path}")

    # 构建 delogo 滤镜链 (支持多区域, 用逗号分隔多个 delogo)
    delogo_filters = []
    for region in regions:
        x1, y1, x2, y2 = region
        # 按参考分辨率缩放坐标
        x1, y1, x2, y2 = _scale_region(
            x1, y1, x2, y2, ref_width, ref_height, vid_w, vid_h
        )
        # delogo 参数: x, y, w, h
        w = x2 - x1
        h = y2 - y1
        # 确保坐标在视频范围内
        x1 = max(0, min(x1, vid_w - 1))
        y1 = max(0, min(y1, vid_h - 1))
        w = min(w, vid_w - x1)
        h = min(h, vid_h - y1)

        delogo_filters.append(f"delogo=x={x1}:y={y1}:w={w}:h={h}:show={show_mode}")

    vf = ",".join(delogo_filters)

    # 输出路径
    OUTPUT_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        target_dir = Path(output_dir) if output_dir else OUTPUT_WATERMARK
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / generate_output_name(video_path.stem, ".mp4", "_no_wm_delogo")
    output_path = Path(output_path)

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]

    logger.info(f"FFmpeg delogo 去水印: {video_path.name} ({len(regions)} 个区域)")

    run_ffmpeg_with_progress(
        cmd, duration=duration,
        desc=f"delogo 去水印 {video_path.name}",
    )

    if not output_path.is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ delogo 去水印完成: {output_path.name} ({output_size_mb:.1f} MB)")

    return {
        "output": str(output_path),
        "size_mb": round(output_size_mb, 2),
    }


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    from tools.common import parse_region

    parser = argparse.ArgumentParser(description="FFmpeg delogo 快速去水印")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument(
        "-r", "--region",
        type=parse_region,
        action="append",
        required=True,
        help="水印区域 (x1,y1,x2,y2), 可多次指定",
    )
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument("--crf", type=int, default=18, help="输出质量 CRF (默认: 18)")

    args = parser.parse_args()

    result = remove_watermark_delogo(
        video_path=args.input,
        regions=args.region,
        output_path=args.output,
        crf=args.crf,
    )
    print(f"\n✅ 去水印完成: {result['output']}")
