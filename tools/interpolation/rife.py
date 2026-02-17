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
import tempfile
from pathlib import Path

import cv2
import numpy as np


from config import OUTPUT_INTERPOLATION, INTERPOLATION_TARGET_FPS
from tools.common import logger, VideoFrameProcessor, generate_output_name


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

    # 获取基本信息计算倍数
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

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

    OUTPUT_INTERPOLATION.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_INTERPOLATION / generate_output_name(
            video_path.stem, ".mp4", f"_{actual_multiplier}x_{new_fps:.0f}fps"
        )
    output_path = Path(output_path)

    logger.info(
        f"RIFE 插帧: {video_path.name} ({fps:.1f}fps → {new_fps:.0f}fps, "
        f"{actual_multiplier}x, {rife_passes} passes)"
    )

    # 多趟处理 (streaming processing to avoid OOM)
    current_input = video_path
    current_output = None
    intermediate_files = []

    for pass_idx in range(rife_passes):
        is_last_pass = (pass_idx == rife_passes - 1)
        
        if is_last_pass:
            current_output = output_path
        else:
            # 中间文件使用临时文件
            tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            current_output = Path(tf.name)
            tf.close()
            intermediate_files.append(current_output)

        desc = f"RIFE Pass {pass_idx + 1}/{rife_passes}"
        
        # 使用 VideoFrameProcessor 进行处理
        # 只有最后一趟需要合并音频 (直接从原始 video_path 合并? 不, VideoFrameProcessor 默认从 input_path 合并)
        # 这里的 input_path 在 pass 2 是中间文件 (无音频或有音频). 
        # VideoFrameProcessor(enable_merge_audio=True) 会尝试从 input_path 提取音频
        # 建议: 第一趟不合并音频 (因输出是temp), 最后一趟合并音频 (从原始 video_path).
        # 但 VideoFrameProcessor 只能从 *它的* input_path 合并.
        # 所以我们需要 ensure intermediate files have audio? or manually merge at the end.
        # VideoFrameProcessor merge logic is: `merge_audio(self.input_path, ...)`
        # RIFE 导致视频变慢(如果音频不处理)? 不, fps变了, 时长不变.
        # 所以音频可以直接 copy.
        # 方案: 
        # Pass 1 (Input=Source, Output=Temp1): Copy audio? Yes.
        # Pass 2 (Input=Temp1, Output=Output): Copy audio? Yes.
        # 这样每一趟都带音频.
        
        with VideoFrameProcessor(
            current_input, 
            current_output, 
            enable_merge_audio=True 
        ) as vp:
            # 必须重新计算 fps, 因为每一趟 fps 翻倍
            current_pass_fps = fps * (2 ** (pass_idx + 1))
            vp.init_writer(fps=current_pass_fps)
            
            frame_gen = vp.frames(desc=desc)
            
            try:
                frame1 = next(frame_gen)
            except StopIteration:
                break
                
            for frame2 in frame_gen:
                # 插帧
                mid = _interpolate_frames_ncnn(frame1, frame2, rife)
                vp.write(frame1)
                vp.write(mid)
                frame1 = frame2
            
            # 写入最后一帧
            vp.write(frame1)

        # 更新下一趟的输入
        current_input = current_output

    # 清理中间文件
    for f in intermediate_files:
        if f.exists():
            f.unlink()

    logger.info(
        f"✅ 插帧完成: {output_path.name} "
        f"({fps:.1f}fps → {new_fps:.0f}fps)"
    )

    return {
        "output": str(output_path),
        "frames_processed": total_frames * actual_multiplier, # 近似值
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
