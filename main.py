"""
x-tools 交互式终端入口 (TUI)

功能:
  - 引导用户配置参数
  - 扫描 input/ 目录或选择单文件
  - 调用 Rich 显示进度
"""
import shutil
import sys
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import INPUT_DIR, FFMPEG_BIN, ADD_WATERMARK_TEXT, ensure_dirs
from tools.common import scan_videos, scan_media

# 引入各个批量处理去水、超分等函数
from tools.watermark.batch import batch_remove_watermark_opencv, batch_remove_watermark_lama
from tools.upscale.batch import batch_upscale_ffmpeg, batch_upscale_realesrgan
from tools.interpolation.batch import batch_interpolate_ffmpeg, batch_interpolate_rife
from tools.add_watermark.batch import batch_add_text_watermark, batch_add_image_watermark
from tools.convert.batch import batch_convert
from tools.convert.ffmpeg_convert import VIDEO_FORMATS, AUDIO_FORMATS
from tools.mediainfo.probe import get_detailed_info, display_info, display_batch_summary
from tools.filter.batch import batch_filter
from tools.filter.ffmpeg_filter import FILTER_PRESETS
from tools.crop.batch import batch_crop
from tools.concat.ffmpeg_concat import concat_videos, get_available_music, TRANSITION_PRESETS
from tools.subtitle.whisper_transcribe import transcribe_video, WHISPER_MODELS
from tools.subtitle.ffmpeg_subtitle import burn_subtitles, SUBTITLE_STYLES


def _prompt_input_mode() -> str:
    """统一输入源选择菜单"""
    return inquirer.select(
        message="选择输入源:",
        choices=[
            Choice("scan", "📂 扫描 input/ 目录"),
            Choice("path", "📄 指定单个文件路径"),
            Choice("manual_dir", "📁 指定其他目录"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()


def _prompt_directory(recursive_default: bool = False) -> tuple[Path, bool]:
    """统一目录输入与递归选项"""
    path_str = inquirer.filepath(
        message="输入目录路径:",
        default=str(INPUT_DIR),
        validate=lambda x: Path(x).is_dir(),
        only_directories=True,
    ).execute()
    recursive = inquirer.confirm(message="是否递归扫描子目录?", default=recursive_default).execute()
    return Path(path_str), recursive


def get_input_videos() -> list[Path] | None:
    """获取待处理视频列表, 返回 None 表示用户选择返回"""
    mode = _prompt_input_mode()

    if mode == "back":
        return None

    if mode == "scan":
        recursive = inquirer.confirm(message="是否递归扫描子目录?", default=False).execute()
        videos = scan_videos(INPUT_DIR, recursive=recursive)
        print(f"扫描到 {len(videos)} 个视频")
        return videos

    if mode == "path":
        path_str = inquirer.filepath(
            message="输入视频文件路径:",
            validate=lambda x: Path(x).is_file(),
        ).execute()
        return [Path(path_str)]

    if mode == "manual_dir":
        directory, recursive = _prompt_directory()
        videos = scan_videos(directory, recursive=recursive)
        print(f"扫描到 {len(videos)} 个视频")
        return videos

    return []


def get_input_media() -> list[Path] | None:
    """获取待处理媒体文件列表 (视频 + 图片), 返回 None 表示用户选择返回"""
    mode = _prompt_input_mode()

    if mode == "back":
        return None

    if mode == "scan":
        recursive = inquirer.confirm(message="是否递归扫描子目录?", default=False).execute()
        videos, images = scan_media(INPUT_DIR, recursive=recursive)
        files = images + videos
        print(f"扫描到 {len(images)} 个图片, {len(videos)} 个视频")
        return files

    if mode == "path":
        path_str = inquirer.filepath(
            message="输入文件路径:",
            validate=lambda x: Path(x).is_file(),
        ).execute()
        return [Path(path_str)]

    if mode == "manual_dir":
        directory, recursive = _prompt_directory()
        videos, images = scan_media(directory, recursive=recursive)
        files = images + videos
        print(f"扫描到 {len(images)} 个图片, {len(videos)} 个视频")
        return files

    return []

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

                # 缩放帧以适配屏幕 (高分辨率/高DPI 屏幕)
                max_w, max_h = 1280, 720
                fh, fw = frame.shape[:2]
                scale_ratio = min(max_w / fw, max_h / fh, 1.0)
                if scale_ratio < 1.0:
                    display = cv2.resize(frame, (int(fw * scale_ratio), int(fh * scale_ratio)))
                else:
                    display = frame

                print("\n📸 请在弹出的窗口中框选水印区域，按 Enter 或 Space 确认...")
                cv2.namedWindow("Select Watermark", cv2.WINDOW_NORMAL)
                x, y, w, h = cv2.selectROI("Select Watermark", display, showCrosshair=True)
                cv2.destroyAllWindows()

                # 还原到原始分辨率坐标
                if scale_ratio < 1.0:
                    x, y, w, h = (
                        x / scale_ratio, y / scale_ratio,
                        w / scale_ratio, h / scale_ratio,
                    )
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
    
    # FFmpeg 高级设置默认值
    mode_val = "mci"
    mb_size_val = 16
    search_param_val = 32

    if engine == "ffmpeg":
        target_fps = float(inquirer.text(message="目标帧率 (FPS):", default="60").execute())
        
        advanced = inquirer.confirm(message="是否进行高级抗抖动设置?", default=False).execute()
        if advanced:
            mode_val = inquirer.select(
                message="插帧模式 (解决抖动/撕裂):",
                choices=[
                    Choice("mci", "🚀 运动补偿 (默认, 最平滑但可能抖动)"),
                    Choice("blend", "🐢 帧混合 (绝对稳定无抖动, 但有重影)"),
                ],
                default="mci",
            ).execute()
            
            if mode_val == "mci":
                mb_size_choice = inquirer.select(
                    message="运动搜索块大小 (影响细节还原):",
                    choices=[
                        Choice(16, "大块 16x16 (默认, 速度快)"),
                        Choice(8, "小块 8x8 (更精细, 可减轻小物体抖动, 极慢)"),
                    ],
                    default=16,
                ).execute()
                mb_size_val = mb_size_choice

                search_choice = inquirer.select(
                    message="运动搜索范围 (处理快速大范围运动):",
                    choices=[
                        Choice(32, "标准 32 (默认)"),
                        Choice(64, "扩大 64 (可改善大动作撕裂, 极慢)"),
                    ],
                    default=32,
                ).execute()
                search_param_val = search_choice

    else:
        multiplier = int(inquirer.select(
            message="倍数:",
            choices=[Choice(2, "2x"), Choice(4, "4x")],
            default=2
        ).execute())

    if inquirer.confirm(message=f"确认处理 {len(videos)} 个视频?", default=True).execute():
        if engine == "ffmpeg":
            batch_interpolate_ffmpeg(
                videos=videos, target_fps=target_fps,
                mode=mode_val, mb_size=mb_size_val, search_param=search_param_val
            )
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
        text = inquirer.text(message="水印文字:", default=ADD_WATERMARK_TEXT).execute()
        if not text.strip():
            print("❌ 水印文字不能为空")
            return

        position = inquirer.select(
            message="水印位置 (居中靠下):",
            choices=[
                Choice("bottom-center-5", "1/5 (距底较远)"),
                Choice("bottom-center-10", "1/10"),
                Choice("bottom-center-16", "1/16 (距底较近)"),
            ],
            default="bottom-center-16",
        ).execute()

        font_size = int(inquirer.number(message="字号:", default=50).execute())
        opacity = float(inquirer.text(message="透明度 (0.0~1.0):", default="0.6").execute())

        blend_mode = inquirer.select(
            message="混合模式:",
            choices=[
                Choice("overlay", "🔆 叠加"),
                Choice("normal", "📋 普通叠加"),
            ],
            default="overlay",
        ).execute()

        bold = inquirer.confirm(message="是否加粗?", default=True).execute()
        stroke_width = 1 if bold else 0

        wide_spacing = inquirer.confirm(message="是否拉开字间距?", default=True).execute()
        letter_spacing = font_size // 3 if wide_spacing else 0

        if inquirer.confirm(message=f"是否查看将要处理的 {len(media)} 个文件列表?", default=False).execute():
            print("\n文件列表:")
            for v in media:
                print(f"  - {v.name}")
            print()

        if inquirer.confirm(message=f"确认为 {len(media)} 个文件添加文字水印?", default=True).execute():
            batch_add_text_watermark(
                files=media, text=text,
                position=position, font_size=font_size, opacity=opacity,
                blend_mode=blend_mode, stroke_width=stroke_width,
                letter_spacing=letter_spacing,
            )

    elif wm_type == "image":
        logo_path = inquirer.filepath(
            message="Logo 图片路径 (推荐 PNG):",
            validate=lambda x: Path(x).is_file(),
        ).execute()

        position = inquirer.select(
            message="水印位置 (居中靠下):",
            choices=[
                Choice("bottom-center-5", "1/5 (距底较远)"),
                Choice("bottom-center-6", "1/6"),
                Choice("bottom-center-7", "1/7 (距底较近)"),
            ],
            default="bottom-center-7",
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


def menu_convert(media: list[Path]):
    """格式转换菜单"""
    mode = inquirer.select(
        message="选择转换模式:",
        choices=[
            Choice("transcode", "🎬 视频格式转换 (MKV/MOV/AVI ↔ MP4 等)"),
            Choice("audio", "🎵 提取音频 (视频 → MP3/AAC/WAV/FLAC)"),
            Choice("strip", "🔇 去除音频 (仅保留视频流)"),
            Choice("remux", "⚡ 快速封装 (无损换容器, 极快)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if mode == "back":
        return

    video_fmts = sorted(VIDEO_FORMATS.keys())
    audio_fmts = sorted(AUDIO_FORMATS.keys())

    if mode == "audio":
        target_format = inquirer.select(
            message="目标音频格式:",
            choices=[Choice(f, f".{f}") for f in audio_fmts],
            default="mp3",
        ).execute()
    elif mode in ("transcode", "strip", "remux"):
        target_format = inquirer.select(
            message="目标视频格式:",
            choices=[Choice(f, f".{f}") for f in video_fmts],
            default="mp4",
        ).execute()

    # 转码模式可选编码器
    video_codec = None
    if mode == "transcode":
        codec_choice = inquirer.select(
            message="视频编码器:",
            choices=[
                Choice(None, "🔧 默认 (根据格式自动选择)"),
                Choice("libx264", "H.264 (兼容性最好)"),
                Choice("libx265", "H.265/HEVC (更好压缩, 部分平台不支持)"),
            ],
            default=None,
        ).execute()
        video_codec = codec_choice

    if inquirer.confirm(message=f"是否查看将要处理的 {len(media)} 个文件列表?", default=False).execute():
        print("\n文件列表:")
        for f in media:
            print(f"  - {f.name}")
        print()

    if inquirer.confirm(message=f"确认转换 {len(media)} 个文件 → .{target_format}?", default=True).execute():
        batch_convert(
            files=media,
            target_format=target_format,
            video_codec=video_codec,
            copy_streams=(mode == "remux"),
            strip_audio=(mode == "strip"),
        )


def menu_mediainfo(media: list[Path]):
    """查看媒体信息菜单"""
    if len(media) == 1:
        # 单文件: 直接展示详细信息
        info = get_detailed_info(media[0])
        display_info(info)
    else:
        view_mode = inquirer.select(
            message=f"已选择 {len(media)} 个文件, 查看方式:",
            choices=[
                Choice("summary", "📊 汇总表格 (所有文件对比)"),
                Choice("detail", "📋 逐个查看 (每个文件详细信息)"),
                Separator(),
                Choice("back", "⬅️  返回上一级"),
            ],
        ).execute()

        if view_mode == "back":
            return

        if view_mode == "summary":
            display_batch_summary(media)
        else:
            for f in media:
                info = get_detailed_info(f)
                display_info(info)


def menu_crop(media: list[Path]):
    """裁切比例菜单"""
    ratio = inquirer.select(
        message="选择目标比例:",
        choices=[
            Choice("1:1", "⬜ 1:1 正方形"),
            Choice("3:4", "📱 3:4 竖屏经典"),
            Choice("4:3", "🖥️  4:3 横屏经典"),
            Choice("9:16", "📲 9:16 竖屏全面"),
            Choice("16:9", "🎬 16:9 横屏宽幅"),
        ],
        default="3:4",
    ).execute()

    if inquirer.confirm(message=f"确认将 {len(media)} 个文件裁切为 {ratio}?", default=True).execute():
        batch_crop(files=media, ratio=ratio)


def menu_filter(media: list[Path]):
    """滤镜效果菜单"""
    # 构建选项列表: 展示滤镜名称 + 描述
    filter_choices = [
        Choice(key, f"{info['name']}  {info['desc']}")
        for key, info in FILTER_PRESETS.items()
    ]
    preset = inquirer.select(
        message="选择滤镜:",
        choices=filter_choices,
        default="cinematic",
    ).execute()

    if inquirer.confirm(message=f"是否查看将要处理的 {len(media)} 个文件列表?", default=False).execute():
        print("\n文件列表:")
        for f in media:
            print(f"  - {f.name}")
        print()

    preset_name = FILTER_PRESETS[preset]["name"]
    if inquirer.confirm(message=f"确认为 {len(media)} 个文件应用 {preset_name} 滤镜?", default=True).execute():
        batch_filter(files=media, preset=preset)


def menu_concat(videos: list[Path]):
    """拼接视频菜单"""
    if len(videos) == 1:
        print(f"\n检测到 1 个视频，可以为其添加背景音乐:")
    else:
        print(f"\n将按以下顺序拼接 {len(videos)} 个视频:")
    
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v.name}")
    print()

    # 背景音乐 (单个视频时默认为 True)
    add_bgm = inquirer.confirm(
        message="是否添加背景音乐?", 
        default=True if len(videos) == 1 else False
    ).execute()

    music_path = None
    music_volume = 0.3
    keep_audio = False

    if add_bgm:
        available_music = get_available_music()

        music_choices = []
        if available_music:
            for m in available_music:
                music_choices.append(Choice(str(m), f"🎵 {m.stem}"))
        music_choices.append(Choice("custom", "📂 指定音乐文件路径"))

        music_choice = inquirer.select(
            message="选择背景音乐:",
            choices=music_choices,
        ).execute()

        if music_choice == "custom":
            music_path = inquirer.filepath(
                message="音乐文件路径:",
                validate=lambda x: Path(x).is_file(),
            ).execute()
        else:
            music_path = music_choice

        music_volume = float(inquirer.text(message="音乐音量 (0.0~1.0):", default="0.3").execute())
        keep_audio = inquirer.confirm(message="是否保留原视频声音 (混合)?", default=False).execute()

    # 过渡效果
    transition_choices = [
        Choice(key, preset["name"]) for key, preset in TRANSITION_PRESETS.items()
    ]
    transition = inquirer.select(
        message="过渡效果:",
        choices=transition_choices,
        default="none",
    ).execute()

    transition_duration = 1.0
    if transition != "none":
        transition_duration = float(inquirer.text(message="过渡时长 (秒):", default="1").execute())

    # 首尾裁剪 (去除 AI 生成视频的静止帧)
    trim_choice = inquirer.select(
        message="裁剪每个视频的首尾? (去除静止帧, 使拼接更流畅)",
        choices=[
            Choice("0", "⏩ 不裁剪"),
            Choice("0.2", "✂️  各裁剪 0.2 秒"),
            Choice("0.4", "✂️  各裁剪 0.4 秒"),
            Choice("1.0", "✂️  各裁剪 1.0 秒"),
            Choice("custom", "✏️  自定义"),
        ],
        default="0",
    ).execute()

    trim_start = 0.0
    trim_end = 0.0
    if trim_choice == "custom":
        trim_start = float(inquirer.text(message="裁剪开头 (秒):").execute())
        trim_end = float(inquirer.text(message="裁剪结尾 (秒):").execute())
    elif trim_choice != "0":
        trim_start = trim_end = float(trim_choice)

    # 音频过渡 (避免接缝爆音)
    audio_fade_choice = inquirer.select(
        message="是否进行音频平滑(淡入淡出), 防止拼接处突兀/破音?",
        choices=[
            Choice("0", "🔊 保持原声不变"),
            Choice("1.0", "🔉 平滑过渡 1 秒 (推荐)"),
            Choice("2.0", "🔉 平滑过渡 2 秒"),
            Choice("custom", "✏️  自定义"),
        ],
        default="1.0",
    ).execute()
    
    audio_fade_in = 0.0
    audio_fade_out = 0.0
    if audio_fade_choice == "custom":
        audio_fade_in = float(inquirer.text(message="音频首部淡入 (秒):").execute())
        audio_fade_out = float(inquirer.text(message="音频尾部淡出 (秒):").execute())
    else:
        audio_fade_in = audio_fade_out = float(audio_fade_choice)

    if inquirer.confirm(message=f"确认{'添加音乐' if len(videos) == 1 else f'拼接 {len(videos)} 个视频'}?", default=True).execute():
        concat_videos(
            video_paths=videos,
            music_path=music_path,
            music_volume=music_volume,
            keep_original_audio=keep_audio,
            transition=transition,
            transition_duration=transition_duration,
            trim_start=trim_start,
            trim_end=trim_end,
            audio_fade_in=audio_fade_in,
            audio_fade_out=audio_fade_out,
        )


def menu_subtitle(media: list[Path]):
    """字幕菜单"""
    from tools.subtitle.whisper_transcribe import WHISPER_MODELS, transcribe_video
    from tools.subtitle.ffmpeg_subtitle import SUBTITLE_STYLES, burn_subtitles
    from tools.subtitle.tts_dubbing import TTS_VOICES, dub_video_with_tts

    # 只处理视频
    videos = [f for f in media if f.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm", ".m4v"}]
    if not videos:
        print("❌ 未找到视频文件")
        return

    mode = inquirer.select(
        message="字幕功能:",
        choices=[
            Choice("auto", "🎙️ 自动生成字幕 (AI 语音识别)"),
            Choice("burn", "🔥 烧录字幕 (导入 .srt)"),
            Choice("oneclick", "⚡ 一键字幕 (识别 + 烧录)"),
            Choice("tts", "🎤 视频配音 (基于字幕生成语音)"),
        ],
        default="oneclick",
    ).execute()

    if mode in ("auto", "oneclick"):
        # 选择 Whisper 模型
        model_choices = [
            Choice(key, f"{info['name']}  {info['size']}") for key, info in WHISPER_MODELS.items()
        ]
        model_name = inquirer.select(
            message="Whisper 模型:",
            choices=model_choices,
            default="small",
        ).execute()

        # 语言
        lang = inquirer.select(
            message="语言:",
            choices=[
                Choice(None, "🌐 自动检测"),
                Choice("zh", "🇨🇳 中文"),
                Choice("en", "🇬🇧 英语"),
                Choice("ja", "🇯🇵 日语"),
            ],
            default=None,
        ).execute()

    if mode == "burn":
        # 烧录模式: 需要指定 .srt 文件
        srt_path = inquirer.filepath(
            message="SRT 字幕文件路径:",
            validate=lambda x: Path(x).is_file() and Path(x).suffix.lower() == ".srt",
        ).execute()

    if mode in ("burn", "oneclick"):
        # 选择字幕样式
        style_choices = [
            Choice(key, info["name"]) for key, info in SUBTITLE_STYLES.items()
        ]
        style = inquirer.select(
            message="字幕样式:",
            choices=style_choices,
            default="default",
        ).execute()

    if mode == "tts":
        # 配音模式: 需要指定 .srt 文件并且选择发音人
        srt_path = inquirer.filepath(
            message="SRT 字幕文件路径 (基于此文件生成配音):",
            validate=lambda x: Path(x).is_file() and Path(x).suffix.lower() == ".srt",
        ).execute()

        voice_choices = [
            Choice(key, info["name"]) for key, info in TTS_VOICES.items()
        ]
        voice_key = inquirer.select(
            message="选择配音角色:",
            choices=voice_choices,
            default="xiaoxiao",
        ).execute()

    # 执行
    for v in videos:
        print(f"\n📹 处理: {v.name}")

        if mode == "auto":
            transcribe_video(v, model_name=model_name, language=lang)

        elif mode == "burn":
            burn_subtitles(v, subtitle_path=srt_path, style=style)

        elif mode == "oneclick":
            # 先识别
            result = transcribe_video(v, model_name=model_name, language=lang)
            srt_file = result["output"]
            # 再烧录
            burn_subtitles(v, subtitle_path=srt_file, style=style)
            
        elif mode == "tts":
            # 视频配音
            dub_video_with_tts(v, srt_path=srt_path, voice_key=voice_key)


def _check_ffmpeg():
    """检测 FFmpeg 是否可用"""
    if not shutil.which(FFMPEG_BIN):
        print("❌ 未检测到 FFmpeg, 请先安装:")
        print(f"   当前配置 FFMPEG_BIN = {FFMPEG_BIN!r}")
        print("   macOS:   brew install ffmpeg")
        print("   Ubuntu:  sudo apt install ffmpeg")
        print("   Windows: https://ffmpeg.org/download.html")
        sys.exit(1)


def main():
    _check_ffmpeg()
    ensure_dirs()

    print("\n🧰 X-TOOLS 视频处理工具箱 v0.2.0\n")

    while True:
        module = inquirer.select(
            message="选择功能模块:",
            choices=[
                Choice("watermark", "💧 去水印 (Watermark)"),
                Choice("add_watermark", "🏷️  增加水印 (Add Watermark)"),
                Choice("upscale", "🆙 高清重置 (Upscale)"),
                Choice("interpolate", "⏯️  帧数补充 (Interpolate)"),
                Choice("convert", "🔄 格式转换 (Convert)"),
                Choice("filter", "🎨 滤镜效果 (Filter)"),
                Choice("crop", "✂️  裁切比例 (Crop)"),
                Choice("concat", "🎬 拼接视频 (Concat)"),
                Choice("subtitle", "📝 字幕 (Subtitle)"),
                Choice("mediainfo", "📊 查看信息 (Media Info)"),
                Separator(),
                Choice("exit", "❌ 退出"),
            ],
            default="watermark",
        ).execute()

        if module == "exit":
            print("Bye!")
            sys.exit(0)

        # 获取输入
        if module in ("add_watermark", "convert", "mediainfo", "filter", "crop", "subtitle"):
            media = get_input_media()
            if media is None:
                continue
            if not media:
                print("❌ 未找到媒体文件")
                continue
            if module == "add_watermark":
                menu_add_watermark(media)
            elif module == "convert":
                menu_convert(media)
            elif module == "mediainfo":
                menu_mediainfo(media)
            elif module == "filter":
                menu_filter(media)
            elif module == "crop":
                menu_crop(media)
            elif module == "subtitle":
                menu_subtitle(media)
        else:
            videos = get_input_videos()
            if videos is None:
                continue  # 用户选择返回上一级
            if not videos:
                print("❌ 未找到视频文件")
                continue

            if module == "watermark":
                menu_watermark(videos)
            elif module == "upscale":
                menu_upscale(videos)
            elif module == "interpolate":
                menu_interpolate(videos)
            elif module == "concat":
                menu_concat(videos)

        print()
        if not inquirer.confirm(message="继续其他操作?", default=True).execute():
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消操作")
        sys.exit(0)
