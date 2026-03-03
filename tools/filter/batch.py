"""
批量滤镜效果入口
"""
from pathlib import Path

from config import INPUT_DIR, ensure_dirs
from tools.common import scan_media, batch_process, print_summary
from tools.filter.ffmpeg_filter import apply_filter, FILTER_PRESETS


def batch_filter(
    input_dir: str | Path | None = None,
    files: list[Path] | None = None,
    preset: str = "cinematic",
    crf: int = 18,
    **kwargs,
) -> list[dict]:
    """
    批量应用滤镜

    Args:
        input_dir: 输入目录
        files: 文件列表 (优先于 input_dir)
        preset: 滤镜预设名称
        crf: 视频质量
    """
    if files is None:
        videos, images = scan_media(input_dir or INPUT_DIR)
        files = videos + images

    preset_info = FILTER_PRESETS.get(preset, {})
    name = preset_info.get("name", preset)
    desc = f"批量滤镜 ({name})"

    results = batch_process(
        files,
        apply_filter,
        desc=desc,
        preset=preset,
        crf=crf,
        **kwargs,
    )
    print_summary(results)
    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量滤镜效果")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入目录")
    parser.add_argument(
        "-f", "--filter", dest="preset", default="cinematic",
        help=f"滤镜预设 (可选: {', '.join(FILTER_PRESETS.keys())})",
    )
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF")

    args = parser.parse_args()
    ensure_dirs()

    batch_filter(
        input_dir=args.input_dir,
        preset=args.preset,
        crf=args.crf,
    )
