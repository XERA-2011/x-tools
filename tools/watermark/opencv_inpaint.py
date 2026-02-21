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
from pathlib import Path

import cv2
import numpy as np



from config import OUTPUT_WATERMARK, WATERMARK_INPAINT_RADIUS
from tools.common import logger, VideoFrameProcessor, generate_output_name, load_or_create_mask, parse_region


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


def _scale_regions(
    regions: list[tuple[int, int, int, int]],
    ref_width: int,
    ref_height: int,
    target_width: int,
    target_height: int,
) -> list[tuple[int, int, int, int]]:
    """
    按比例缩放水印区域坐标 (从参考分辨率映射到目标分辨率)

    仅当参考分辨率与目标分辨率差异 > 1% 时才缩放
    """
    sx = target_width / ref_width
    sy = target_height / ref_height
    if abs(sx - 1.0) < 0.01 and abs(sy - 1.0) < 0.01:
        return regions
    return [
        (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))
        for x1, y1, x2, y2 in regions
    ]


def remove_watermark_opencv(
    video_path: str | Path,
    regions: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    output_path: str | Path | None = None,
    method: str = "telea",
    inpaint_radius: int = WATERMARK_INPAINT_RADIUS,
    feather: int = 3,
    ref_width: int = 0,
    ref_height: int = 0,
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
        ref_width: ROI 坐标的参考分辨率宽度 (0=不缩放)
        ref_height: ROI 坐标的参考分辨率高度 (0=不缩放)

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

    if output_path is None:
        output_path = OUTPUT_WATERMARK / generate_output_name(video_path.stem, ".mp4", "_no_wm")
    
    # 获取视频信息以创建 mask
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # 创建或加载 mask (使用公共函数)
    mask = load_or_create_mask(
        width, height, regions, mask_path, feather, ref_width, ref_height,
    )

    # 检查 mask 是否有白色区域
    if cv2.countNonZero(mask) == 0:
        logger.warning("mask 中没有水印区域, 跳过处理")
        return {"output": str(video_path), "frames_processed": 0}

    logger.info(f"去水印处理: {video_path.name} ({method})")

    frames_processed = 0
    with VideoFrameProcessor(video_path, output_path) as vp:
        for frame in vp.frames(desc="逐帧修复"):
            # inpaint 修复
            repaired = cv2.inpaint(frame, mask, inpaint_radius, inpaint_flags)
            vp.write(repaired)
            frames_processed += 1

    logger.info(f"✅ 去水印完成: {output_path.name}")
    return {"output": str(output_path), "frames_processed": frames_processed}




# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

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
