"""
RIFE 视频插帧模块

方法说明:
  - 使用 RIFE (Real-Time Intermediate Flow Estimation) 进行 AI 插帧
  - 速度快, 效果好, 社区活跃
  - 支持 MPS 加速 (Mac M4), 可 2x/4x 倍帧率

依赖 (二选一):
  方案 A (推荐, 无需 CUDA): pip install rife-ncnn-vulkan-python
  方案 B (需 PyTorch):       git clone https://github.com/hzwer/Practical-RIFE

使用方式:
  python tools/interpolation/rife.py video.mp4
  python tools/interpolation/rife.py video.mp4 --target-fps 60
  python tools/interpolation/rife.py video.mp4 --multiplier 4
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FFMPEG_BIN, OUTPUT_INTERPOLATION, INTERPOLATION_TARGET_FPS
from tools.common import get_video_info, logger, merge_audio


def _check_rife_ncnn():
    """检查 rife-ncnn-vulkan-python 是否可用"""
    try:
        from rife_ncnn_vulkan import Rife  # noqa: F401
        return True
    except ImportError:
        return False


def _interpolate_frames_ncnn(frame1: np.ndarray, frame2: np.ndarray, rife) -> np.ndarray:
    """使用 rife-ncnn-vulkan 在两帧间插入一帧"""
    from PIL import Image

    # OpenCV BGR → PIL RGB
    img1 = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    img2 = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))

    # 插帧
    mid = rife.process(img1, img2)

    # PIL RGB → OpenCV BGR
    return cv2.cvtColor(np.array(mid), cv2.COLOR_RGB2BGR)


def interpolate_video_rife(
    video_path: str | Path,
    output_path: str | Path | None = None,
    multiplier: int | None = None,
    target_fps: float | None = None,
) -> dict:
    """
    使用 RIFE 对视频进行插帧

    Args:
        video_path: 输入视频路径
        output_path: 输出路径
        multiplier: 帧率倍数 (2=双倍帧率, 4=四倍), 与 target_fps 二选一
        target_fps: 目标帧率 (如 60), 与 multiplier 二选一

    Returns:
        dict: {"output": str, "frames_processed": int,
               "original_fps": float, "new_fps": float}
    """
    try:
        from rife_ncnn_vulkan import Rife
    except ImportError:
        raise RuntimeError(
            "❌ 未安装 rife-ncnn-vulkan-python，请运行:\n"
            "   pip install rife-ncnn-vulkan-python\n"
            "   或使用 FFmpeg 方案: python tools/interpolation/ffmpeg_minterp.py"
        )

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 计算倍数
    if target_fps and not multiplier:
        multiplier = max(2, round(target_fps / fps))
    elif not multiplier:
        multiplier = max(2, round(INTERPOLATION_TARGET_FPS / fps))

    # 只支持 2 的幂次 (RIFE 每次插入 1 帧 = 2x)
    rife_passes = 0
    m = multiplier
    while m > 1:
        m //= 2
        rife_passes += 1
    actual_multiplier = 2 ** rife_passes

    new_fps = fps * actual_multiplier

    # 初始化 RIFE
    rife = Rife(gpuid=0)

    # 输出路径
    OUTPUT_INTERPOLATION.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_INTERPOLATION / f"{video_path.stem}_{actual_multiplier}x_{new_fps:.0f}fps.mp4"
    output_path = Path(output_path)

    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video_path, fourcc, new_fps, (width, height))

    logger.info(
        f"RIFE 插帧: {video_path.name} ({fps:.1f}fps → {new_fps:.0f}fps, "
        f"{actual_multiplier}x, {rife_passes} passes, {total_frames} 帧)"
    )

    from tqdm import tqdm

    # 读取所有帧
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 2:
        raise ValueError("视频帧数不足 (至少需要 2 帧)")

    # 逐 pass 插帧
    for pass_idx in range(rife_passes):
        new_frames = []
        desc = f"RIFE Pass {pass_idx + 1}/{rife_passes}"
        for i in tqdm(range(len(frames) - 1), desc=desc, unit="帧"):
            new_frames.append(frames[i])
            mid = _interpolate_frames_ncnn(frames[i], frames[i + 1], rife)
            new_frames.append(mid)
        new_frames.append(frames[-1])  # 最后一帧
        frames = new_frames

    # 写入所有帧
    for frame in tqdm(frames, desc="写入视频", unit="帧"):
        writer.write(frame)

    writer.release()
    frames_out = len(frames)

    # 合并音频 (变速处理: 音频保持原始时长)
    merge_audio(video_path, temp_video_path, output_path)
    Path(temp_video_path).unlink(missing_ok=True)

    logger.info(
        f"✅ 插帧完成: {output_path.name} "
        f"({fps:.1f}fps → {new_fps:.0f}fps, {total_frames} → {frames_out} 帧)"
    )

    return {
        "output": str(output_path),
        "frames_processed": frames_out,
        "original_fps": round(fps, 2),
        "new_fps": round(new_fps, 2),
    }




# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RIFE 视频插帧")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-o", "--output", help="输出路径")

    fps_group = parser.add_mutually_exclusive_group()
    fps_group.add_argument(
        "-m", "--multiplier", type=int,
        help="帧率倍数 (2/4/8)",
    )
    fps_group.add_argument(
        "-t", "--target-fps", type=float,
        help=f"目标帧率 (默认: {INTERPOLATION_TARGET_FPS})",
    )

    args = parser.parse_args()

    if not _check_rife_ncnn():
        print("❌ 未安装 rife-ncnn-vulkan-python，请运行: pip install rife-ncnn-vulkan-python")
        sys.exit(1)

    result = interpolate_video_rife(
        video_path=args.input,
        output_path=args.output,
        multiplier=args.multiplier,
        target_fps=args.target_fps,
    )
    print(f"\n✅ 插帧完成: {result['original_fps']}fps → {result['new_fps']}fps")
    print(f"   输出: {result['output']}")
