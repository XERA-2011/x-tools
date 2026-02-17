"""
LaMA 深度学习去水印模块

方法说明:
  - 使用 LaMA (Large Mask Inpainting) 模型进行深度学习修复
  - 效果远优于传统 OpenCV inpaint, 尤其对大面积/复杂水印
  - 模型会自动下载到 ~/.cache/lama/big-lama.pt (~196MB)
  - 使用 PyTorch + MPS (Mac) 加速

使用方式:
  python tools/watermark/lama_remover.py video.mp4 --region 10,10,200,60

依赖:
  pip install torch torchvision
"""
import urllib.request
import os
import sys
from pathlib import Path

import cv2
import numpy as np


from config import OUTPUT_WATERMARK
from tools.common import logger, VideoFrameProcessor, generate_output_name
from tools.watermark.opencv_inpaint import create_mask_from_regions


def _check_torch():
    """检查 torch 是否可用"""
    try:
        import torch
        return True
    except ImportError:
        return False


def _download_model(model_path: Path):
    """下载 LaMA 模型"""
    url = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
    
    logger.info(f"下载 LaMA 模型 ({url})...")
    logger.info("如果下载慢，请设置代理 (export https_proxy=...)")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 删除可能损坏的旧文件
    if model_path.exists():
        model_path.unlink()

    def progress(count, block_size, total_size):
        pct = count * block_size * 100 / total_size
        mb = count * block_size / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        sys.stdout.write(f"\r  Downloading: {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
        sys.stdout.flush()

    try:
        # 检测代理环境变量
        proxies = {}
        if os.environ.get("https_proxy"):
            proxies["https"] = os.environ.get("https_proxy")
        if os.environ.get("http_proxy"):
            proxies["http"] = os.environ.get("http_proxy")
            
        if proxies:
            proxy_handler = urllib.request.ProxyHandler(proxies)
            opener = urllib.request.build_opener(proxy_handler)
            urllib.request.install_opener(opener)
            
        urllib.request.urlretrieve(url, str(model_path), reporthook=progress)
        print() # 换行
        logger.info(f"✅ 模型下载完成: {model_path}")
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"模型下载失败: {e}")


def remove_watermark_lama(
    video_path: str | Path,
    regions: list[tuple[int, int, int, int]] | None = None,
    mask_path: str | Path | None = None,
    output_path: str | Path | None = None,
    device: str | None = None,
    feather: int = 5,
) -> dict:
    """
    使用 LaMA 模型去除视频水印

    Args:
        video_path: 输入视频路径
        regions: 水印矩形区域列表 [(x1,y1,x2,y2), ...]
        mask_path: mask 图片路径
        output_path: 输出路径
        device: 推理设备 ("mps" / "cpu" / "cuda"), None 自动选择
        feather: 区域边缘羽化 (mask处理用)
    """
    if not _check_torch():
        raise RuntimeError("❌ 未安装 torch, 请运行: pip install torch torchvision")
        
    import torch
    
    # 自动选择设备
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if regions is None and mask_path is None:
        raise ValueError("必须指定 regions 或 mask_path 其中之一")

    # 1. 准备模型
    model_path = Path.home() / ".cache" / "lama" / "big-lama.pt"
    if not model_path.exists():
        _download_model(model_path)
        
    logger.info(f"加载 LaMA 模型 (设备: {device})...")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    # 3. 准备输出
    OUTPUT_WATERMARK.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = OUTPUT_WATERMARK / generate_output_name(video_path.stem, ".mp4", "_no_wm_lama")
    output_path = Path(output_path)

    frames_processed = 0

    # 使用 VideoFrameProcessor 简化流程
    with VideoFrameProcessor(video_path, output_path) as vp:
        # 获取尺寸用于 mask 创建/padding 计算
        width, height = vp.width, vp.height

        # 2. 准备 Mask
        if mask_path:
            mask_org = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_org is None:
                raise FileNotFoundError(f"无法读取 mask: {mask_path}")
            if mask_org.shape[:2] != (height, width):
                mask_org = cv2.resize(mask_org, (width, height))
        else:
            mask_org = create_mask_from_regions(width, height, regions, feather)
            
        # 确保 mask 是二值化的
        _, mask_thresh = cv2.threshold(mask_org, 127, 255, cv2.THRESH_BINARY)
        
        # 预处理 Mask 为 Tensor
        pad_h = (8 - height % 8) % 8
        pad_w = (8 - width % 8) % 8
        
        mask_float = (mask_thresh > 127).astype(np.float32)
        if pad_h or pad_w:
            mask_float = np.pad(mask_float, ((0, pad_h), (0, pad_w)), mode='reflect')
            
        mask_t = torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0).to(device)

        logger.info(f"LaMA 去水印: {video_path.name}")

        # 4. 推理循环
        for frame in vp.frames(desc="LaMA 修复"):
            # BGR -> RGB -> Tensor
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            
            if pad_h or pad_w:
                img_rgb = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
                
            # (H, W, 3) -> (1, 3, H, W)
            img_t = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)

            # Infer
            with torch.no_grad():
                output_t = model(img_t, mask_t)

            # Tensor -> numpy -> BGR
            output = output_t[0].cpu().numpy().transpose(1, 2, 0)
            output = np.clip(output * 255, 0, 255).astype(np.uint8)
            
            if pad_h or pad_w:
                output = output[:height, :width]
                
            output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
            vp.write(output_bgr)
            frames_processed += 1

    logger.info(f"✅ LaMA 去水印完成: {output_path.name}")
    return {"output": str(output_path), "frames_processed": frames_processed}


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
    parser.add_argument("--device", choices=["mps", "cpu", "cuda"], help="设备 (默认: 自动选择)")

    args = parser.parse_args()

    if not args.region and not args.mask:
        parser.error("必须指定 --region 或 --mask")

    if not _check_torch():
        print("❌ 未安装 torch, 请运行: pip install torch torchvision")
        sys.exit(1)

    remove_watermark_lama(
        video_path=args.input,
        regions=args.region,
        mask_path=args.mask,
        output_path=args.output,
        device=args.device,
    )
