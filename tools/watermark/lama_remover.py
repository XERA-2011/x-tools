"""
LaMA 深度学习去水印模块 (预留)

方法说明:
  - 使用 LaMA (Large Mask Inpainting) 模型进行深度学习修复
  - 效果远优于传统 OpenCV inpaint, 尤其对大面积/复杂水印
  - 需要安装额外依赖: pip install lama-cleaner 或 iopaint

使用方式:
  安装: pip install iopaint
  单文件: python tools/watermark/lama_remover.py video.mp4 --region 10,10,200,60

依赖:
  - iopaint (前身 lama-cleaner): pip install iopaint
  - torch: pip install torch torchvision
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FFMPEG_BIN, OUTPUT_WATERMARK
from tools.common import logger
from tools.watermark.opencv_inpaint import create_mask_from_regions, _merge_audio


def _check_iopaint():
    """检查 iopaint 是否可用"""
    try:
        from iopaint import api  # noqa: F401
        return True
    except ImportError:
        return False


def remove_watermark_lama(
    video_path: str | Path,
    regions: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model: str = "lama",
    device: str = "mps",
    feather: int = 5,
) -> dict:
    """
    使用 LaMA 模型去除视频水印

    Args:
        video_path: 输入视频路径
        regions: 水印矩形区域列表 [(x1,y1,x2,y2), ...]
        mask_path: mask 图片路径
        output_path: 输出路径
        model: 模型名称 ("lama" / "ldm" / "zits" / "mat" / "fcf" / "manga")
        device: 推理设备 ("mps" / "cpu" / "cuda")
        feather: 区域边缘羽化

    Returns:
        dict: {"output": str, "frames_processed": int}
    """
    # 延迟导入, 避免未安装时报错
    try:
        from iopaint.model_manager import ModelManager
        from iopaint.schema import InpaintRequest, HDStrategy
    except ImportError:
        raise RuntimeError(
            "❌ 未安装 iopaint, 请先运行:\n"
            "   pip install iopaint torch torchvision\n"
            "   或使用 OpenCV 方案: python tools/watermark/opencv_inpaint.py"
        )

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if regions is None and mask_path is None:
        raise ValueError("必须指定 regions 或 mask_path 其中之一")

    # 打开视频
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建或加载 mask
    if mask_path:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"无法读取 mask: {mask_path}")
        if mask.shape[:2] != (height, width):
            mask = cv2.resize(mask, (width, height))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = create_mask_from_regions(width, height, regions, feather)

    # 加载模型
    logger.info(f"加载 {model} 模型 (设备: {device})...")
    model_manager = ModelManager(name=model, device=device)

    # 输出路径
    OUTPUT_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_WATERMARK / f"{video_path.stem}_no_wm_lama.mp4"
    output_path = Path(output_path)

    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    logger.info(f"LaMA 去水印: {video_path.name} ({total_frames} 帧)")

    from tqdm import tqdm
    frames_done = 0

    # 构造 inpaint 请求
    config = InpaintRequest(
        hd_strategy=HDStrategy.ORIGINAL,
    )

    for _ in tqdm(range(total_frames), desc="LaMA 修复", unit="帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # LaMA 推理: RGB 输入
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        repaired_rgb = model_manager(frame_rgb, mask, config)
        repaired = cv2.cvtColor(repaired_rgb, cv2.COLOR_RGB2BGR)

        writer.write(repaired)
        frames_done += 1

    cap.release()
    writer.release()

    # 合并音频
    _merge_audio(video_path, temp_video_path, output_path)
    Path(temp_video_path).unlink(missing_ok=True)

    logger.info(f"✅ LaMA 去水印完成: {output_path.name} ({frames_done} 帧)")
    return {"output": str(output_path), "frames_processed": frames_done}


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    def parse_region(s: str) -> tuple[int, int, int, int]:
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError("区域格式: x1,y1,x2,y2")
        return tuple(parts)

    parser = argparse.ArgumentParser(description="LaMA 深度学习去水印")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument("-r", "--region", type=parse_region, action="append", help="水印区域 (x1,y1,x2,y2)")
    parser.add_argument("-m", "--mask", help="mask 图片路径")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument("--model", default="lama", help="模型 (默认: lama)")
    parser.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"], help="设备 (默认: mps)")
    parser.add_argument("--feather", type=int, default=5, help="边缘羽化 (默认: 5)")

    args = parser.parse_args()

    if not args.region and not args.mask:
        parser.error("必须指定 --region 或 --mask")

    if not _check_iopaint():
        print("❌ 未安装 iopaint, 请先运行: pip install iopaint torch torchvision")
        sys.exit(1)

    result = remove_watermark_lama(
        video_path=args.input,
        regions=args.region,
        mask_path=args.mask,
        output_path=args.output,
        model=args.model,
        device=args.device,
        feather=args.feather,
    )
    print(f"\n✅ 完成: {result['output']} ({result['frames_processed']} 帧)")
