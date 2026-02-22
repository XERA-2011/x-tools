"""
x-tools 交互式终端入口 (TUI)

功能:
  - 引导用户选择处理模块 (提取/去水印/超分/插帧)
  - 引导用户配置参数
  - 扫描 input/ 目录或选择单文件
  - 调用 Rich 显示进度
"""
import shutil
import sys
from pathlib import Path
from typing import Callable

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import INPUT_DIR, UPSCALE_FACTOR, INTERPOLATION_TARGET_FPS, ensure_dirs
from tools.common import scan_videos, scan_media, logger

# 引入各个批量处理函数
from tools.extract.batch import batch_extract_clips, batch_extract_keyframes, batch_extract_interval, batch_extract_scenes
from tools.watermark.batch import batch_remove_watermark_opencv, batch_remove_watermark_lama
from tools.upscale.batch import batch_upscale_ffmpeg, batch_upscale_realesrgan
from tools.interpolation.batch import batch_interpolate_ffmpeg, batch_interpolate_rife
from tools.add_watermark.batch import batch_add_text_watermark, batch_add_image_watermark


def get_input_videos() -> list[Path]:
    """获取待处理视频列表"""
    # 选项: 扫描 input/ 目录 或 输入路径
    mode = inquirer.select(
        message="选择输入源:",
        choices=[
            Choice("scan", f"📂 扫描 input/ 目录"),
            Choice("path", "📄 指定单个文件路径"),
            Choice("manual_dir", "📁 指定其他目录"),
        ],
    ).execute()

    if mode == "scan":
        recursive = inquirer.confirm(message="是否递归扫描子目录?", default=False).execute()
        videos = scan_videos(INPUT_DIR, recursive=recursive)
        print(f"扫描到 {len(videos)} 个视频")
        return videos
    
    elif mode == "path":
        path_str = inquirer.filepath(
            message="输入视频文件路径:",
            validate=lambda x: Path(x).is_file(),
        ).execute()
        return [Path(path_str)]
    
    elif mode == "manual_dir":
        path_str = inquirer.filepath(
            message="输入目录路径:",
            default=str(INPUT_DIR),
            validate=lambda x: Path(x).is_dir(),
            only_directories=True,
        ).execute()
        recursive = inquirer.confirm(message="是否递归扫描子目录?", default=False).execute()
        videos = scan_videos(path_str, recursive=recursive)
        print(f"扫描到 {len(videos)} 个视频")
        return videos
    
    return []


def get_input_media() -> list[Path]:
    """获取待处理媒体文件列表 (视频 + 图片)"""
    mode = inquirer.select(
        message="选择输入源:",
        choices=[
            Choice("scan", f"📂 扫描 input/ 目录"),
            Choice("path", "📄 指定单个文件路径"),
            Choice("manual_dir", "📁 指定其他目录"),
        ],
    ).execute()

    if mode == "scan":
        recursive = inquirer.confirm(message="是否递归扫描子目录?", default=False).execute()
        videos, images = scan_media(INPUT_DIR, recursive=recursive)
        files = images + videos
        print(f"扫描到 {len(images)} 个图片, {len(videos)} 个视频")
        return files
    
    elif mode == "path":
        path_str = inquirer.filepath(
            message="输入文件路径:",
            validate=lambda x: Path(x).is_file(),
        ).execute()
        return [Path(path_str)]
    
    elif mode == "manual_dir":
        path_str = inquirer.filepath(
            message="输入目录路径:",
            default=str(INPUT_DIR),
            validate=lambda x: Path(x).is_dir(),
            only_directories=True,
        ).execute()
        recursive = inquirer.confirm(message="是否递归扫描子目录?", default=False).execute()
        videos, images = scan_media(path_str, recursive=recursive)
        files = images + videos
        print(f"扫描到 {len(images)} 个图片, {len(videos)} 个视频")
        return files
    
    return []


def menu_extract(videos: list[Path]):
    """内容提取菜单"""
    action = inquirer.select(
        message="选择提取模式:",
        choices=[
            Choice("clip", "✂️  视频片段截取"),
            Choice("keyframe", "🖼️  关键帧提取"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if action == "back":
        return

    if action == "clip":
        start = inquirer.text(message="开始时间 (秒 or 00:00:00):", default="0").execute()
        duration = inquirer.text(message="持续时长 (秒):", default="10").execute()
        
        if inquirer.confirm(message=f"是否查看将要处理的 {len(videos)} 个文件列表?", default=False).execute():
            print("\n文件列表:")
            for v in videos:
                print(f"  - {v.name}")
            print()

        if inquirer.confirm(message=f"确认截取 {len(videos)} 个视频?", default=True).execute():
            batch_extract_clips(videos=videos, start=start, duration=duration)

    elif action == "keyframe":
        mode = inquirer.select(
            message="关键帧提取规则:",
            choices=[
                Choice("keyframes", "仅 I-帧 (关键帧)"),
                Choice("interval", "按时间间隔"),
                Choice("scene", "按场景变化"),
            ],
        ).execute()
        
        interval = 5
        if mode == "interval":
            interval = int(inquirer.number(message="间隔秒数:", default=5).execute())
            
        if inquirer.confirm(message=f"确认提取 {len(videos)} 个视频的关键帧?", default=True).execute():
            if mode == "keyframes":
                batch_extract_keyframes(videos=videos)
            elif mode == "interval":
                batch_extract_interval(videos=videos, interval=float(interval))
            elif mode == "scene":
                batch_extract_scenes(videos=videos)


def menu_watermark(videos: list[Path]):
    """去水印菜单"""
    engine = inquirer.select(
        message="选择去水印引擎:",
        choices=[
            Choice("opencv", "🔧 OpenCV (传统算法, 快速, 适合简单水印)"),
            Choice("lama", "🧠 LaMA (深度学习, 慢, 效果好, 适合复杂水印)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if engine == "back":
        return

    # 参考分辨率 (仅在鼠标框选时记录)
    ref_width = 0
    ref_height = 0

    # 使用循环代替递归, 避免栈溢出
    while True:
        print("请输入水印区域坐标: x1,y1,x2,y2")
        print("提示: 输入 's' 或 'select' 可开启鼠标框选 (需本地运行)")
        region_input = inquirer.text(message="区域坐标 (或 s):").execute()

        if region_input.lower() in ["s", "select"]:
            try:
                import cv2
                sample_video = videos[0]
                cap = cv2.VideoCapture(str(sample_video))
                ref_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ref_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * 0.2))
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    print("❌ 无法读取视频帧，请手动输入")
                    ref_width = ref_height = 0
                    continue
                    
                print("\n📸 请在弹出的窗口中框选水印区域，按 Enter 或 Space 确认...")
                x, y, w, h = cv2.selectROI("Select Watermark", frame, showCrosshair=True)
                cv2.destroyAllWindows()
                x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
                print(f"✅ 已选择: {x1},{y1},{x2},{y2}")
                
                if w == 0 or h == 0:
                    print("⚠️ 未选择区域")
                    ref_width = ref_height = 0
                    continue
                    
            except Exception as e:
                print(f"❌ 启动图形界面失败: {e}\n请尝试手动输入坐标。")
                ref_width = ref_height = 0
                continue
        else:
            try:
                x1, y1, x2, y2 = [int(p.strip()) for p in region_input.split(',')]
            except (ValueError, TypeError):
                print("❌ 格式错误，请使用 x1,y1,x2,y2")
                continue

        # 成功解析坐标, 跳出循环
        break
    
    if inquirer.confirm(message=f"确认处理 {len(videos)} 个视频?", default=True).execute():
        if engine == "opencv":
            batch_remove_watermark_opencv(
                videos=videos, regions=[(x1, y1, x2, y2)],
                ref_width=ref_width, ref_height=ref_height,
            )
        else:
            batch_remove_watermark_lama(
                videos=videos, regions=[(x1, y1, x2, y2)],
                ref_width=ref_width, ref_height=ref_height,
            )


def menu_upscale(videos: list[Path]):
    """超分菜单"""
    engine = inquirer.select(
        message="选择放大引擎:",
        choices=[
            Choice("ffmpeg", "⚙️  FFmpeg (传统插值, 无需GPU)"),
            Choice("realesrgan", "🚀 Real-ESRGAN (AI超分, 需GPU/MPS)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if engine == "back":
        return

    # 选择放大方式
    upscale_mode = inquirer.select(
        message="放大方式:",
        choices=[
            Choice("multiplier", "🔢 倍数放大 (2x / 4x)"),
            Choice("resolution", "🖥️  目标分辨率 (1080p / 2K / 4K)"),
        ],
    ).execute()

    scale = None
    target_width = None
    target_height = None

    if upscale_mode == "multiplier":
        scale = int(inquirer.select(
            message="放大倍数:",
            choices=[Choice(2, "2x"), Choice(4, "4x")],
            default=2
        ).execute())
    else:
        res = inquirer.select(
            message="目标分辨率:",
            choices=[
                Choice("1080p", "1080p (1920×1080)"),
                Choice("2k", "2K (2560×1440)"),
                Choice("4k", "4K (3840×2160)"),
            ],
        ).execute()
        resolutions = {"1080p": (1920, 1080), "2k": (2560, 1440), "4k": (3840, 2160)}
        target_width, target_height = resolutions[res]

        if engine == "realesrgan":
            print(f"💡 将使用 AI 超分 + 精确缩放 → {target_width}x{target_height}")

    if inquirer.confirm(message=f"是否查看将要处理的 {len(videos)} 个文件列表?", default=False).execute():
        print("\n文件列表:")
        for v in videos:
            print(f"  - {v.name}")
        print()

    if inquirer.confirm(message=f"确认放大 {len(videos)} 个视频?", default=True).execute():
        if engine == "ffmpeg":
            if target_width and target_height:
                batch_upscale_ffmpeg(videos=videos, scale=None, width=target_width, height=target_height)
            else:
                batch_upscale_ffmpeg(videos=videos, scale=scale)
        else:
            batch_upscale_realesrgan(
                videos=videos, scale=scale,
                target_width=target_width, target_height=target_height,
            )


def menu_interpolate(videos: list[Path]):
    """插帧菜单"""
    engine = inquirer.select(
        message="选择插帧引擎:",
        choices=[
            Choice("ffmpeg", "⚙️  FFmpeg (运动补偿, 无需GPU)"),
            Choice("rife", "🌊 RIFE (AI插帧, 需GPU/MPS)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()
    
    if engine == "back":
        return
    
    target_fps = 60
    multiplier = 2
    
    if engine == "ffmpeg":
        target_fps = float(inquirer.text(message="目标帧率 (FPS):", default="60").execute())
    else:
        multiplier = int(inquirer.select(
            message="倍数:",
            choices=[Choice(2, "2x"), Choice(4, "4x")],
            default=2
        ).execute())

    if inquirer.confirm(message=f"确认处理 {len(videos)} 个视频?", default=True).execute():
        if engine == "ffmpeg":
            batch_interpolate_ffmpeg(videos=videos, target_fps=target_fps)
        else:
            batch_interpolate_rife(videos=videos, multiplier=multiplier)


def menu_add_watermark(media: list[Path]):
    """加水印菜单"""
    wm_type = inquirer.select(
        message="选择水印类型:",
        choices=[
            Choice("text", "📝 文字水印 (支持中文)"),
            Choice("image", "🖼️  图片水印 (Logo)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if wm_type == "back":
        return

    if wm_type == "text":
        text = inquirer.text(message="水印文字:").execute()
        if not text.strip():
            print("❌ 水印文字不能为空")
            return

        position = inquirer.select(
            message="水印位置:",
            choices=[
                Choice("bottom-right", "↘️  右下角"),
                Choice("bottom-left", "↙️  左下角"),
                Choice("top-right", "↗️  右上角"),
                Choice("top-left", "↖️  左上角"),
                Choice("center", "⊕  居中"),
            ],
            default="bottom-right",
        ).execute()

        font_size = int(inquirer.number(message="字号:", default=36).execute())
        opacity = float(inquirer.text(message="透明度 (0.0~1.0):", default="0.7").execute())

        if inquirer.confirm(message=f"是否查看将要处理的 {len(media)} 个文件列表?", default=False).execute():
            print("\n文件列表:")
            for v in media:
                print(f"  - {v.name}")
            print()

        if inquirer.confirm(message=f"确认为 {len(media)} 个文件添加文字水印?", default=True).execute():
            batch_add_text_watermark(
                files=media, text=text,
                position=position, font_size=font_size, opacity=opacity,
            )

    elif wm_type == "image":
        logo_path = inquirer.filepath(
            message="Logo 图片路径 (推荐 PNG):",
            validate=lambda x: Path(x).is_file(),
        ).execute()

        position = inquirer.select(
            message="水印位置:",
            choices=[
                Choice("bottom-right", "↘️  右下角"),
                Choice("bottom-left", "↙️  左下角"),
                Choice("top-right", "↗️  右上角"),
                Choice("top-left", "↖️  左上角"),
                Choice("center", "⊕  居中"),
            ],
            default="bottom-right",
        ).execute()

        scale = float(inquirer.text(message="Logo 大小比例 (0.0~1.0):", default="0.15").execute())
        opacity = float(inquirer.text(message="透明度 (0.0~1.0):", default="0.7").execute())

        if inquirer.confirm(message=f"是否查看将要处理的 {len(media)} 个文件列表?", default=False).execute():
            print("\n文件列表:")
            for v in media:
                print(f"  - {v.name}")
            print()

        if inquirer.confirm(message=f"确认为 {len(media)} 个文件添加 Logo 水印?", default=True).execute():
            batch_add_image_watermark(
                files=media, watermark_path=logo_path,
                position=position, scale=scale, opacity=opacity,
            )


def _check_ffmpeg():
    """检测 FFmpeg 是否可用"""
    if not shutil.which("ffmpeg"):
        print("❌ 未检测到 FFmpeg, 请先安装:")
        print("   macOS:   brew install ffmpeg")
        print("   Ubuntu:  sudo apt install ffmpeg")
        print("   Windows: https://ffmpeg.org/download.html")
        sys.exit(1)


def main():
    _check_ffmpeg()
    ensure_dirs()

    print(r"""
 __   __        ______            _     
 \ \ / /       |  ____|          | |    
  \ V / ______ | |__   ___   ___ | |___ 
   > < |______||  __| / _ \ / _ \| / __|
  / . \        | |   | (_) | (_) | \__ \
 /_/ \_\       |_|    \___/ \___/|_|___/
    """)
    print("视频处理工具箱 v0.1\n")

    while True:
        module = inquirer.select(
            message="选择功能模块:",
            choices=[
                Choice("extract", "✂️  内容截取 (Extract)"),
                Choice("watermark", "💧 去水印 (Watermark)"),
                Choice("add_watermark", "🏷️  增加水印 (Add Watermark)"),
                Choice("upscale", "🆙 高清重置 (Upscale)"),
                Choice("interpolate", "⏯️  帧数补充 (Interpolate)"),
                Separator(),
                Choice("exit", "❌ 退出"),
            ],
            default="watermark",
        ).execute()

        if module == "exit":
            print("Bye!")
            sys.exit(0)

        # 获取输入
        if module == "add_watermark":
            media = get_input_media()
            if not media:
                print("❌ 未找到媒体文件")
                continue
            menu_add_watermark(media)
        else:
            videos = get_input_videos()
            if not videos:
                print("❌ 未找到视频文件")
                continue

            # 进入子菜单
            if module == "extract":
                menu_extract(videos)
            elif module == "watermark":
                menu_watermark(videos)
            elif module == "upscale":
                menu_upscale(videos)
            elif module == "interpolate":
                menu_interpolate(videos)

        print()
        if not inquirer.confirm(message="继续其他操作?", default=True).execute():
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消操作")
        sys.exit(0)
