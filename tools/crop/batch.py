"""
批量裁切入口
"""
from pathlib import Path

from config import INPUT_DIR, ensure_dirs
from tools.common import resolve_media_files, run_batch
from tools.crop.ffmpeg_crop import ASPECT_RATIOS, crop_media


def batch_crop(
    input_dir: str | Path | None = None,
    files: list[Path] | None = None,
    ratio: str = "3:4",
    crf: int = 18,
    **kwargs,
) -> list[dict]:
    """
    批量裁切

    Args:
        input_dir: 输入目录
        files: 文件列表
        ratio: 目标比例
        crf: 视频质量
    """
    files = resolve_media_files(input_dir, files, kind="media", media_order="videos_first")

    desc = f"批量裁切 ({ratio})"

    return run_batch(
        files,
        crop_media,
        desc=desc,
        ratio=ratio,
        crf=crf,
        **kwargs,
    )


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量裁切")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入目录")
    parser.add_argument(
        "-r", "--ratio", default="3:4",
        help=f"目标比例 (可选: {', '.join(ASPECT_RATIOS.keys())})",
    )
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF")

    args = parser.parse_args()
    ensure_dirs()

    batch_crop(
        input_dir=args.input_dir,
        ratio=args.ratio,
        crf=args.crf,
    )
