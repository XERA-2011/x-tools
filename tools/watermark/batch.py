"""
批量去水印入口

支持:
  - 批量 OpenCV 去水印 (同一水印位置)
  - 批量 LaMA 去水印 (同一水印位置)
"""
from pathlib import Path


from config import INPUT_DIR, ensure_dirs
from tools.common import scan_videos, batch_process, print_summary, logger, parse_region
from tools.watermark.opencv_inpaint import remove_watermark_opencv


def batch_remove_watermark_opencv(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    regions: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    method: str = "telea",
    ref_width: int = 0,
    ref_height: int = 0,
    **kwargs,
) -> list[dict]:
    """
    批量去水印 (OpenCV) — 对 input 目录下所有视频应用相同水印区域

    Args:
        input_dir: 输入目录
        regions: 水印区域列表 [(x1,y1,x2,y2), ...]
        mask_path: mask 图片路径
        method: 修复算法
        ref_width: ROI 坐标的参考分辨率宽度 (0=不缩放)
        ref_height: ROI 坐标的参考分辨率高度 (0=不缩放)
        inpaint_radius: 修复半径
        feather: 边缘羽化
    """
    ensure_dirs()
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)
    results = batch_process(
        videos,
        remove_watermark_opencv,
        desc="批量去水印 (OpenCV)",
        regions=regions,
        mask_path=mask_path,
        method=method,
        ref_width=ref_width,
        ref_height=ref_height,
        **kwargs,
    )
    print_summary(results)
    return results


def batch_remove_watermark_lama(
    input_dir: str | Path | None = None,
    videos: list[Path] | None = None,
    regions: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    device: str | None = None,
    ref_width: int = 0,
    ref_height: int = 0,
    **kwargs,
) -> list[dict]:
    """
    批量去水印 (LaMA) — 对 input 目录下所有视频应用相同水印区域

    Args:
        input_dir: 输入目录
        regions: 水印区域列表
        mask_path: mask 图片路径
        model: 模型名称
        device: 推理设备
        ref_width: ROI 坐标的参考分辨率宽度 (0=不缩放)
        ref_height: ROI 坐标的参考分辨率高度 (0=不缩放)
        feather: 边缘羽化
    """
    ensure_dirs()
    if videos is None:
        videos = scan_videos(input_dir or INPUT_DIR)
    from tools.watermark.lama_remover import remove_watermark_lama
    results = batch_process(
        videos,
        remove_watermark_lama,
        desc="批量去水印 (LaMA)",
        regions=regions,
        mask_path=mask_path,
        device=device,
        ref_width=ref_width,
        ref_height=ref_height,
        **kwargs,
    )
    print_summary(results)
    return results


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse



    parser = argparse.ArgumentParser(description="批量去水印")
    parser.add_argument("-i", "--input-dir", default=str(INPUT_DIR), help="输入视频目录")
    parser.add_argument("-r", "--region", type=parse_region, action="append", help="水印区域 (x1,y1,x2,y2)")
    parser.add_argument("-m", "--mask", help="mask 图片路径")

    sub = parser.add_subparsers(dest="engine", required=True)

    # OpenCV
    opencv_p = sub.add_parser("opencv", help="使用 OpenCV inpaint")
    opencv_p.add_argument("--method", choices=["telea", "ns"], default="telea")
    opencv_p.add_argument("--radius", type=int, default=5)
    opencv_p.add_argument("--feather", type=int, default=3)

    # LaMA
    lama_p = sub.add_parser("lama", help="使用 LaMA 深度学习")
    lama_p.add_argument("--model", default="lama")
    lama_p.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    lama_p.add_argument("--feather", type=int, default=5)

    args = parser.parse_args()

    if not args.region and not args.mask:
        parser.error("必须指定 --region 或 --mask")

    if args.engine == "opencv":
        batch_remove_watermark_opencv(
            input_dir=args.input_dir, regions=args.region, mask_path=args.mask,
            method=args.method, inpaint_radius=args.radius, feather=args.feather,
        )
    elif args.engine == "lama":
        batch_remove_watermark_lama(
            input_dir=args.input_dir, regions=args.region, mask_path=args.mask,
            device=args.device, feather=args.feather,
        )
