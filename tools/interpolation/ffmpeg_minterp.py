"""
FFmpeg 运动补偿插帧模块

方法说明:
  - 使用 FFmpeg minterpolate 滤镜进行运动补偿插帧
  - 无需 GPU, 速度适中, 效果尚可
  - 适合: 快速插帧、不想安装 AI 依赖

使用方式:
  python tools/interpolation/ffmpeg_minterp.py video.mp4
  python tools/interpolation/ffmpeg_minterp.py video.mp4 --target-fps 60
  python tools/interpolation/ffmpeg_minterp.py video.mp4 --target-fps 48 --mode blend
"""
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FFMPEG_BIN, OUTPUT_INTERPOLATION, INTERPOLATION_TARGET_FPS
from tools.common import get_video_info, logger


def interpolate_video_ffmpeg(
    video_path: str | Path,
    output_path: str | Path | None = None,
    target_fps: float = INTERPOLATION_TARGET_FPS,
    mode: str = "mci",
    mc_mode: str = "aobmc",
    me_mode: str = "bidir",
    crf: int = 18,
) -> dict:
    """
    使用 FFmpeg minterpolate 进行运动补偿插帧

    Args:
        video_path: 输入视频路径
        output_path: 输出路径
        target_fps: 目标帧率
        mode: 插帧模式
            - "mci": 运动补偿插帧 (最佳效果, 默认)
            - "blend": 帧混合 (较快, 有残影)
            - "dup": 帧复制 (最快, 无平滑)
        mc_mode: 运动补偿模式 ("aobmc" 自适应 / "obmc" 重叠)
        me_mode: 运动估计模式 ("bidir" 双向 / "bilat" 双侧)
        crf: 输出质量

    Returns:
        dict: {"output": str, "original_fps": float, "new_fps": float}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    info = get_video_info(video_path)
    if not info:
        raise RuntimeError(f"无法读取视频信息: {video_path}")

    orig_fps = info["fps"]

    if target_fps <= orig_fps:
        logger.warning(
            f"目标帧率 ({target_fps}) ≤ 原始帧率 ({orig_fps}), "
            f"将使用原始帧率"
        )
        target_fps = orig_fps

    # 输出路径
    OUTPUT_INTERPOLATION.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_INTERPOLATION / f"{video_path.stem}_{target_fps:.0f}fps_ffmpeg.mp4"
    output_path = Path(output_path)

    # minterpolate 滤镜
    mi_filter = (
        f"minterpolate=fps={target_fps}"
        f":mi_mode={mode}"
        f":mc_mode={mc_mode}"
        f":me_mode={me_mode}"
        f":vsbmc=1"
    )

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(video_path),
        "-vf", mi_filter,
        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
        "-c:a", "copy",
        str(output_path),
    ]

    logger.info(
        f"FFmpeg 插帧: {video_path.name} ({orig_fps:.1f}fps → {target_fps:.0f}fps, "
        f"mode={mode})"
    )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    out_info = get_video_info(output_path)
    new_fps = out_info.get("fps", target_fps)

    logger.info(f"✅ 插帧完成: {output_path.name} ({orig_fps:.1f}fps → {new_fps}fps)")

    return {
        "output": str(output_path),
        "original_fps": round(orig_fps, 2),
        "new_fps": round(new_fps, 2),
    }


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FFmpeg 运动补偿插帧")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument(
        "-t", "--target-fps", type=float, default=INTERPOLATION_TARGET_FPS,
        help=f"目标帧率 (默认: {INTERPOLATION_TARGET_FPS})",
    )
    parser.add_argument(
        "--mode", choices=["mci", "blend", "dup"], default="mci",
        help="插帧模式 (默认: mci)",
    )
    parser.add_argument(
        "--mc-mode", choices=["aobmc", "obmc"], default="aobmc",
        help="运动补偿模式 (默认: aobmc)",
    )
    parser.add_argument(
        "--me-mode", choices=["bidir", "bilat"], default="bidir",
        help="运动估计模式 (默认: bidir)",
    )
    parser.add_argument("--crf", type=int, default=18, help="输出质量 (默认: 18)")

    args = parser.parse_args()

    result = interpolate_video_ffmpeg(
        video_path=args.input,
        output_path=args.output,
        target_fps=args.target_fps,
        mode=args.mode,
        mc_mode=args.mc_mode,
        me_mode=args.me_mode,
        crf=args.crf,
    )
    print(f"\n✅ 插帧完成: {result['original_fps']}fps → {result['new_fps']}fps")
    print(f"   输出: {result['output']}")
