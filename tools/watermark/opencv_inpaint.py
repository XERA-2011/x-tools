"""
OpenCV 传统修复去水印模块

方法说明:
  - 使用 OpenCV 的 inpaint 算法 (Telea / Navier-Stokes) 逐帧修复水印区域
  - 需要指定水印区域 (矩形坐标) 或提供 mask 图片
  - 适合: 固定位置的静态水印 (如 logo、台标)

使用方式:
  单文件: python tools/watermark/opencv_inpaint.py video.mp4 --region 10,10,200,60
  多区域: python tools/watermark/opencv_inpaint.py video.mp4 --region 10,10,200,60 --region 500,10,700,60
"""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config import FFMPEG_BIN, OUTPUT_WATERMARK, WATERMARK_INPAINT_RADIUS
from tools.common import get_video_info, logger, merge_audio


def create_mask_from_regions(
    width: int,
    height: int,
    regions: list[tuple[int, int, int, int]],
    feather: int = 3,
) -> np.ndarray:
    """
    根据矩形区域列表创建 mask

    Args:
        width: 视频宽度
        height: 视频高度
        regions: 水印区域列表, 每项为 (x1, y1, x2, y2)
        feather: 边缘羽化像素数 (使修复过渡更自然)

    Returns:
        np.ndarray: 二值 mask (白色=水印区域)
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for x1, y1, x2, y2 in regions:
        # 确保坐标在范围内
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # 对 mask 做轻微膨胀和模糊, 使边缘过渡更自然
    if feather > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather * 2 + 1, feather * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def remove_watermark_opencv(
    video_path: str | Path,
    regions: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    output_path: str | Path | None = None,
    method: str = "telea",
    inpaint_radius: int = WATERMARK_INPAINT_RADIUS,
    feather: int = 3,
) -> dict:
    """
    使用 OpenCV inpaint 去除视频水印

    Args:
        video_path: 输入视频路径
        regions: 水印矩形区域列表 [(x1,y1,x2,y2), ...], 与 mask_path 二选一
        mask_path: 已有 mask 图片路径 (白色=水印), 与 regions 二选一
        output_path: 输出路径, 为 None 时自动生成
        method: 修复算法 ("telea" 或 "ns")
        inpaint_radius: 修复半径 (越大修复范围越大, 但越模糊)
        feather: 区域边缘羽化像素

    Returns:
        dict: {"output": str, "frames_processed": int}
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if regions is None and mask_path is None:
        raise ValueError("必须指定 regions 或 mask_path 其中之一")

    # 确定修复算法
    inpaint_flags = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

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

    # 检查 mask 是否有白色区域
    if cv2.countNonZero(mask) == 0:
        logger.warning("mask 中没有水印区域, 跳过处理")
        cap.release()
        return {"output": str(video_path), "frames_processed": 0}

    # 输出路径: 先写无音频的临时文件, 最后混合原音频
    OUTPUT_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_WATERMARK / f"{video_path.stem}_no_wm.mp4"
    output_path = Path(output_path)

    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    # 写视频 (无音频)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    logger.info(f"去水印处理: {video_path.name} ({total_frames} 帧, {method})")

    from tqdm import tqdm
    frames_done = 0

    for _ in tqdm(range(total_frames), desc="逐帧修复", unit="帧"):
        ret, frame = cap.read()
        if not ret:
            break

        # inpaint 修复
        repaired = cv2.inpaint(frame, mask, inpaint_radius, inpaint_flags)
        writer.write(repaired)
        frames_done += 1

    cap.release()
    writer.release()

    # 混合原视频音频 + 修复后的视频画面
    merge_audio(video_path, temp_video_path, output_path)

    # 清理临时文件
    Path(temp_video_path).unlink(missing_ok=True)

    logger.info(f"✅ 去水印完成: {output_path.name} ({frames_done} 帧)")
    return {"output": str(output_path), "frames_processed": frames_done}




# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    def parse_region(s: str) -> tuple[int, int, int, int]:
        """解析区域字符串 'x1,y1,x2,y2'"""
        parts = [int(x.strip()) for x in s.split(",")]
        if len(parts) != 4:
            raise argparse.ArgumentTypeError("区域格式: x1,y1,x2,y2")
        return tuple(parts)

    parser = argparse.ArgumentParser(description="OpenCV 去水印")
    parser.add_argument("input", help="输入视频路径")
    parser.add_argument(
        "-r", "--region",
        type=parse_region,
        action="append",
        help="水印区域 (x1,y1,x2,y2), 可多次指定",
    )
    parser.add_argument("-m", "--mask", help="mask 图片路径 (白色=水印区域)")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument(
        "--method",
        choices=["telea", "ns"],
        default="telea",
        help="修复算法 (默认: telea)",
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=WATERMARK_INPAINT_RADIUS,
        help=f"修复半径 (默认: {WATERMARK_INPAINT_RADIUS})",
    )
    parser.add_argument(
        "--feather",
        type=int,
        default=3,
        help="边缘羽化像素 (默认: 3)",
    )

    args = parser.parse_args()

    if not args.region and not args.mask:
        parser.error("必须指定 --region 或 --mask")

    result = remove_watermark_opencv(
        video_path=args.input,
        regions=args.region,
        mask_path=args.mask,
        output_path=args.output,
        method=args.method,
        inpaint_radius=args.radius,
        feather=args.feather,
    )
    print(f"\n✅ 去水印完成: {result['output']} ({result['frames_processed']} 帧)")
