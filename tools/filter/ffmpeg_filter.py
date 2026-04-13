"""
FFmpeg 滤镜效果模块

功能:
  - 预设滤镜 (高对比/徕卡M3/徕卡M9)
  - 支持视频和图片
  - 基于 FFmpeg 视频滤镜, 处理速度快

使用方式:
  python tools/filter/ffmpeg_filter.py video.mp4 -f high_contrast
  python tools/filter/ffmpeg_filter.py image.jpg -f leica_m9
"""
import subprocess
from pathlib import Path

from config import FFMPEG_BIN, IMAGE_EXTENSIONS, OUTPUT_FILTER, VIDEO_EXTENSIONS
from tools.common import generate_output_name, logger

# ============================================================
# 预设滤镜定义
# ============================================================
FILTER_PRESETS = {
    "saturate": {
        "name": "🎨 高饱和",
        "desc": "色彩浓郁鲜艳, 适度对比, 画面通透",
        "vf": "eq=saturation=1.1:contrast=1.0:brightness=0.01",
    },
    "high_contrast": {
        "name": "💎 高对比",
        "desc": "强烈对比, 色彩鲜明, 视觉冲击",
        "vf": "eq=contrast=1.05:saturation=1.0:brightness=-0.01",
    },
    "vivid": {
        "name": "🌈 高饱和高对比",
        "desc": "极致鲜艳, 强烈对比, 色彩浓郁冲击力",
        "vf": "eq=contrast=1.05:saturation=1.1:brightness=-0.01",
    },
    "leica_m3": {
        "name": "📸 徕卡 M3",
        "desc": "MONOPAN 50 黑白胶片, 细腻颗粒感, 丰富灰阶层次, 人文街拍",
        "vf": "hue=s=0,"
              "eq=contrast=1.2:brightness=0.01:gamma=1.05,"
              "curves=m='0/0.03 0.15/0.1 0.4/0.38 0.6/0.62 0.85/0.88 1/0.95',"
              "noise=alls=12:allf=t,"
              "vignette=PI/3.5",
    },
    "leica_m9": {
        "name": "📷 徕卡 M9",
        "desc": "CCD 德味色彩, 高饱和油画质感, 浓郁暖调, 经典数码徕卡",
        "vf": "eq=contrast=1.2:brightness=-0.01:saturation=1.25,"
              "colorbalance=rs=0.1:gs=0.04:bs=-0.06:rm=0.08:gm=0.05:bm=-0.04:rh=0.05:gh=0.02:bh=-0.03,"
              "curves=r='0/0 0.4/0.44 0.7/0.75 1/1':"
              "g='0/0 0.4/0.42 0.7/0.73 1/0.98':"
              "b='0/0.02 0.4/0.38 0.7/0.68 1/0.9',"
              "vignette=PI/5",
    },
}


def apply_filter(
    input_path: str | Path,
    output_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    preset: str = "high_contrast",
    intensity: float = 1.0,
    crf: int = 18,
) -> dict:
    """
    使用 FFmpeg 应用滤镜效果

    Args:
        input_path: 输入文件路径 (视频或图片)
        output_path: 输出路径 (默认自动生成)
        preset: 滤镜预设名称
        intensity: 滤镜强度 (0.0~1.0, 暂保留, 当前全强度)
        crf: 视频质量 CRF

    Returns:
        dict: {"output": str, "preset": str, ...}
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"文件不存在: {input_path}")

    if preset not in FILTER_PRESETS:
        available = ", ".join(FILTER_PRESETS.keys())
        raise ValueError(f"未知滤镜: {preset} (可选: {available})")

    preset_info = FILTER_PRESETS[preset]
    use_fc = "fc" in preset_info  # 是否使用 filter_complex
    vf = preset_info.get("vf", "")
    fc = preset_info.get("fc", "")

    suffix = input_path.suffix.lower()
    is_image = suffix in IMAGE_EXTENSIONS
    is_video = suffix in VIDEO_EXTENSIONS

    if not is_image and not is_video:
        raise ValueError(f"不支持的文件格式: {suffix}")

    # 输出路径
    OUTPUT_FILTER.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        target_dir = Path(output_dir) if output_dir else OUTPUT_FILTER
        target_dir.mkdir(parents=True, exist_ok=True)
        out_ext = suffix if is_image else ".mp4"
        output_name = generate_output_name(input_path.stem, out_ext, tag=preset)
        output_path = target_dir / output_name
    output_path = Path(output_path)

    # 构建 FFmpeg 命令
    cmd = [FFMPEG_BIN, "-y", "-i", str(input_path)]

    if use_fc:
        # 使用 filter_complex (复杂滤镜图, 如径向模糊)
        if is_image:
            cmd += ["-filter_complex", fc, "-map", "[out]", "-q:v", "2", str(output_path)]
        else:
            cmd += [
                "-filter_complex", fc, "-map", "[out]", "-map", "0:a?",
                "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                "-c:a", "copy",
                "-movflags", "+faststart",
                str(output_path),
            ]
    elif is_image:
        # 图片: 单帧处理
        cmd += ["-vf", vf, "-q:v", "2", str(output_path)]
    else:
        # 视频: 保留音频
        cmd += [
            "-vf", vf,
            "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(output_path),
        ]

    logger.info(f"滤镜 [{preset_info['name']}]: {input_path.name}")

    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg 错误:\n{result.stderr[-500:]}")

    if not output_path.is_file():
        raise RuntimeError(f"输出文件未生成: {output_path}")

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"✅ 滤镜完成: {output_path.name} ({output_size_mb:.1f} MB)")

    return {
        "output": str(output_path),
        "preset": preset,
        "preset_name": preset_info["name"],
        "size_mb": round(output_size_mb, 2),
    }


# ============================================================
# 交互式预览
# ============================================================
def preview_filter(input_path: str | Path):
    """
    预览所有滤镜效果: 截取一帧, 应用全部滤镜, 拼成对比图弹窗展示。

    Args:
        input_path: 输入文件路径 (视频或图片)
    
    Returns:
        Callable: 关闭预览窗口的函数，如果失败则返回 None
    """
    import shutil
    import tempfile

    input_path = Path(input_path)
    if not input_path.is_file():
        logger.warning(f"文件不存在: {input_path}")
        return None

    suffix = input_path.suffix.lower()
    is_image = suffix in IMAGE_EXTENSIONS
    is_video = suffix in VIDEO_EXTENSIONS

    if not is_image and not is_video:
        logger.warning(f"不支持的文件格式: {suffix}")
        return None

    # 手动管理临时目录, 在窗口关闭后清理
    tmpdir = Path(tempfile.mkdtemp(prefix="xtools_preview_"))

    # 1) 获取预览帧
    if is_video:
        frame_path = tmpdir / "frame.jpg"
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", str(input_path),
            "-ss", "1",
            "-frames:v", "1",
            "-q:v", "2",
            str(frame_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0 or not frame_path.is_file():
            logger.warning("无法从视频截取帧")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return None
        sample = frame_path
    else:
        sample = input_path

    # 2) 对帧应用所有滤镜
    filtered: list[tuple[str, Path]] = [("原图", sample)]
    for key, info in FILTER_PRESETS.items():
        out = tmpdir / f"{key}.jpg"
        if "fc" in info:
            cmd = [
                FFMPEG_BIN, "-y",
                "-i", str(sample),
                "-filter_complex", info["fc"],
                "-map", "[out]",
                "-q:v", "2",
                str(out),
            ]
        else:
            cmd = [
                FFMPEG_BIN, "-y",
                "-i", str(sample),
                "-vf", info["vf"],
                "-q:v", "2",
                str(out),
            ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode == 0 and out.is_file():
            filtered.append((info["name"], out))

    # 3) 用 Pillow 拼成网格
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("Pillow 未安装，无法生成预览网格")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return None

    cols = 3
    thumb_w = 400
    padding = 4
    bg_color = (30, 30, 30)
    text_color = (255, 255, 255)

    # 大号数字字体, 叠加到缩略图左上角
    font_size = 64
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    thumbs: list[Image.Image] = []
    for idx, (label, path) in enumerate(filtered):
        img = Image.open(path).convert("RGBA")
        ratio = thumb_w / img.width
        img = img.resize((thumb_w, int(img.height * ratio)), Image.LANCZOS)

        # 左上角叠加序号
        num = str(idx + 1)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        od = ImageDraw.Draw(overlay)
        bbox = od.textbbox((0, 0), num, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        badge_w, badge_h = tw + 24, th + 16
        od.rounded_rectangle([8, 8, 8 + badge_w, 8 + badge_h],
                             radius=10, fill=(0, 0, 0, 180))
        od.text((8 + 12, 8 + 8), num, fill=text_color, font=font)
        img = Image.alpha_composite(img, overlay).convert("RGB")

        thumbs.append(img)

    # 打印序号对照表到终端
    logger.info("滤镜序号对照:")
    for idx, (label, _) in enumerate(filtered):
        logger.info(f"  {idx + 1:2d} → {label}")

    thumb_h = thumbs[0].height
    rows_count = -(-len(thumbs) // cols)
    cell_w = thumb_w + padding
    cell_h = thumb_h + padding

    canvas = Image.new("RGB", (cols * cell_w + padding, rows_count * cell_h + padding), bg_color)

    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, cols)
        x = padding + col * cell_w
        y = padding + row * cell_h
        canvas.paste(thumb, (x, y))

    # 4) 外部进程弹窗展示 (跳过 macOS 主线程限制)
    try:
        import sys
        
        # 预先缩放图片并保存
        import cv2
        import numpy as np

        cv_img = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

        max_w, max_h = 1920, 1080
        h, w = cv_img.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        disp_w, disp_h = int(w * scale), int(h * scale)
        if scale < 1.0:
            cv_img = cv2.resize(cv_img, (disp_w, disp_h))

        preview_path = tmpdir / "preview.jpg"
        cv2.imwrite(str(preview_path), cv_img)
        
        # 写一个临时的脚本用于显示窗口
        script_path = tmpdir / "show.py"
        script_content = f"""
import cv2
import sys

img = cv2.imread(r"{preview_path}")
if img is None:
    sys.exit(1)

win = "Filter Preview - Please select in terminal"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.resizeWindow(win, {disp_w}, {disp_h})
cv2.imshow(win, img)
# 阻塞等待直到被主进程 kill 或用户手动关闭窗口
while True:
    cv2.waitKey(100)
    try:
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    except:
        break
"""
        script_path.write_text(script_content, encoding="utf-8")

        # 启动外部进程
        proc = subprocess.Popen([sys.executable, str(script_path)])
        logger.info("📸 预览窗口已打开 (将在终端选择完成后自动关闭)...")

        def close_preview():
            import shutil
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
            shutil.rmtree(tmpdir, ignore_errors=True)

        return close_preview

    except Exception as e:
        # 无 GUI 环境 fallback: 保存到临时文件并提示路径
        preview_path = tmpdir / "preview.jpg"
        canvas.save(str(preview_path), "JPEG", quality=90)
        logger.warning(f"无法打开图形预览 ({e}), 已保存到: {preview_path}")
        return None


# ============================================================
# CLI 入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FFmpeg 滤镜效果")
    parser.add_argument("input", help="输入文件路径 (视频或图片)")
    parser.add_argument("-o", "--output", help="输出路径")
    parser.add_argument(
        "-f", "--filter", dest="preset", default="high_contrast",
        help=f"滤镜预设 (可选: {', '.join(FILTER_PRESETS.keys())})",
    )
    parser.add_argument("--crf", type=int, default=18, help="视频质量 CRF (默认: 18)")

    args = parser.parse_args()

    # 列出可用预设
    if args.preset == "list":
        print("\n可用滤镜预设:")
        for key, info in FILTER_PRESETS.items():
            print(f"  {info['name']:12s}  {key:16s}  {info['desc']}")
        print()
    else:
        result = apply_filter(
            input_path=args.input,
            output_path=args.output,
            preset=args.preset,
            crf=args.crf,
        )
        print(f"\n✅ {result['preset_name']} 完成")
        print(f"   输出: {result['output']} ({result['size_mb']} MB)")
