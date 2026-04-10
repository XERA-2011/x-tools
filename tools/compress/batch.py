"""
批量视频压缩入口
"""
from pathlib import Path

from config import INPUT_DIR, OUTPUT_COMPRESS, ensure_dirs
from tools.common import resolve_media_files, run_batch
from tools.compress.ffmpeg_compress import compress_video



def batch_compress(
    input_dir: str | Path | None = None,
    files: list[Path] | None = None,
    codec: str = "libx264",
    crf: int = 24,
    preset: str = "slow",
    audio_bitrate: str = "128k",
    **kwargs,
) -> list[dict]:
    """
    批量压缩视频
    """
    files = resolve_media_files(input_dir, files, kind="video", media_order="videos_first")

    desc = f"批量无损压缩 ({codec} CRF{crf})"

    return run_batch(
        files,
        compress_video,
        desc=desc,
        base_output_dir=OUTPUT_COMPRESS,
        codec=codec,
        crf=crf,
        preset=preset,
        audio_bitrate=audio_bitrate,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="批量视频压缩")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入目录")
    parser.add_argument("--codec", default="libx264", choices=["libx264", "libx265"], help="视频编码器")
    parser.add_argument("--crf", type=int, default=24, help="视频质量 CRF")
    parser.add_argument("--preset", default="slow", help="压缩预设")

    args = parser.parse_args()
    ensure_dirs()

    batch_compress(
        input_dir=args.input_dir,
        codec=args.codec,
        crf=args.crf,
        preset=args.preset,
    )
