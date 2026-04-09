"""
x-tools 交互式终端入口 (TUI)

功能:
  - 引导用户配置参数
  - 扫描 input/ 目录或选择单文件
  - 调用 Rich 显示进度
"""
import shutil
import sys
from datetime import datetime
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import ADD_WATERMARK_TEXT, FFMPEG_BIN, INPUT_DIR, OUTPUT_DIR, OUTPUT_PDF_SPLIT, WATERMARK_BRAND_PRESETS, ensure_dirs, MUSIC_DIR
from tools.add_watermark.batch import batch_add_image_watermark, batch_add_text_watermark
from tools.bgm.ffmpeg_bgm import add_bgm_to_video
from tools.common import scan_media, scan_videos
from tools.concat.ffmpeg_concat import TRANSITION_PRESETS, concat_videos, get_available_music
from tools.convert.batch import batch_convert
from tools.convert.ffmpeg_convert import AUDIO_FORMATS, VIDEO_FORMATS
from tools.crop.batch import batch_crop
from tools.filter.batch import batch_filter
from tools.filter.ffmpeg_filter import FILTER_PRESETS, preview_filter
from tools.interpolation.batch import batch_interpolate_ffmpeg, batch_interpolate_rife

from tools.subtitle.ffmpeg_subtitle import SUBTITLE_STYLES, burn_subtitles
from tools.subtitle.whisper_transcribe import WHISPER_MODELS, transcribe_video
from tools.upscale.batch import batch_upscale_ffmpeg, batch_upscale_realesrgan

# 引入各个批量处理去水、超分等函数
from tools.watermark.batch import batch_remove_watermark_delogo, batch_remove_watermark_lama, batch_remove_watermark_opencv
from tools.compress.batch import batch_compress


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


def _prompt_directory() -> tuple[Path, bool]:
    """统一目录输入"""
    path_str = inquirer.filepath(
        message="输入目录路径:",
        default="",
        validate=lambda x: Path(x).is_dir(),
        only_directories=True,
    ).execute()
    return Path(path_str), False


def _maybe_show_file_list(files: list[Path], message: str | None = None) -> None:
    """可选展示待处理文件列表"""
    prompt = message or f"是否查看将要处理的 {len(files)} 个文件列表?"
    if inquirer.confirm(message=prompt, default=False).execute():
        print("\n文件列表:")
        for f in files:
            print(f"  - {f.name}")
        print()


def _confirm_action(message: str, default: bool = True) -> bool:
    """统一确认提示"""
    return inquirer.confirm(message=message, default=default).execute()


def get_input_videos() -> list[Path] | None:
    """获取待处理视频列表, 返回 None 表示用户选择返回"""
    mode = _prompt_input_mode()

    if mode == "back":
        return None

    if mode == "scan":
        videos = scan_videos(INPUT_DIR, recursive=False)
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
        videos, images = scan_media(INPUT_DIR, recursive=False)
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
            Choice("delogo", "⚡ FFmpeg delogo (快速推荐, 适合固定位置 Logo)"),
            Choice("opencv", "🔧 OpenCV (传统算法, 中速)"),
            Choice("lama", "🧠 LaMA (深度学习, 慢, 效果最好)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
        default="delogo",
    ).execute()

    if engine == "back":
        return

    # 选择水印品牌类型 (预设 / 手动)
    brand_choices = [
        Choice("veo", f"🎬 {WATERMARK_BRAND_PRESETS['veo']['label']}"),
        Choice("custom", "✍️  手动选择/输入坐标"),
        Separator(),
        Choice("back", "⬅️  返回上一级"),
    ]
    brand = inquirer.select(
        message="选择水印品牌类型:",
        choices=brand_choices,
    ).execute()

    if brand == "back":
        return

    # 参考分辨率 (仅在鼠标框选/预设时记录)
    ref_width = 0
    ref_height = 0
    x1 = y1 = x2 = y2 = 0

    if brand == "veo":
        preset = WATERMARK_BRAND_PRESETS["veo"]
        ref_width = preset["ref_width"]
        ref_height = preset["ref_height"]
        x1, y1, x2, y2 = preset["regions"][0]
        print(f"✅ 已选择 Veo 预设区域: {x1},{y1},{x2},{y2} (参考分辨率 {ref_width}x{ref_height})")
    else:
        # 手动输入/框选
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
                    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
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

    if _confirm_action(f"确认处理 {len(videos)} 个视频?"):
        if engine == "opencv":
            batch_remove_watermark_opencv(
                videos=videos, regions=[(x1, y1, x2, y2)],
                ref_width=ref_width, ref_height=ref_height,
            )
        elif engine == "delogo":
            batch_remove_watermark_delogo(
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

    _maybe_show_file_list(videos)

    if _confirm_action(f"确认放大 {len(videos)} 个视频?"):
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

    if _confirm_action(f"确认处理 {len(videos)} 个视频?"):
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

        _maybe_show_file_list(media)

        if _confirm_action(f"确认为 {len(media)} 个文件添加文字水印?"):
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

        _maybe_show_file_list(media)

        if _confirm_action(f"确认为 {len(media)} 个文件添加 Logo 水印?"):
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

    _maybe_show_file_list(media)

    if _confirm_action(f"确认转换 {len(media)} 个文件 → .{target_format}?"):
        batch_convert(
            files=media,
            target_format=target_format,
            video_codec=video_codec,
            copy_streams=(mode == "remux"),
            strip_audio=(mode == "strip"),
        )



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

    if _confirm_action(f"确认将 {len(media)} 个文件裁切为 {ratio}?"):
        batch_crop(files=media, ratio=ratio)


def menu_filter(media: list[Path]):
    """滤镜效果菜单"""
    # 直接在后台打开预览窗口
    close_preview_fn = preview_filter(media[0])

    # 构建选项列表: 展示滤镜名称 + 描述 + 对应序号 (方便对照)
    filter_choices = []
    for idx, (key, info) in enumerate(FILTER_PRESETS.items()):
        # idx 对应生成的预览图上的数字: 0->2 (1 是原图)
        filter_choices.append(Choice(key, f"[{idx + 2}] {info['name']}  {info['desc']}"))

    try:
        preset = inquirer.select(
            message="选择滤镜 (可参考弹出的预览窗口):",
            choices=filter_choices,
            default="saturate",
        ).execute()
    except KeyboardInterrupt:
        # 用户取消, 关闭预览窗口后向上传播异常
        if close_preview_fn is not None:
            close_preview_fn()
        raise

    # 用户选择完毕, 关闭预览窗口
    if close_preview_fn is not None:
        close_preview_fn()

    _maybe_show_file_list(media)

    preset_name = FILTER_PRESETS[preset]["name"]
    if _confirm_action(f"确认为 {len(media)} 个文件应用 {preset_name} 滤镜?"):
        batch_filter(files=media, preset=preset)


def _prompt_bgm_options() -> tuple[str, float, bool]:
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
    return music_path, music_volume, keep_audio


def menu_add_bgm(videos: list[Path]):
    """添加背景音乐菜单"""
    if not videos:
        print("❌ 未找到视频文件")
        return

    if len(videos) > 1:
        video_choice = inquirer.select(
            message="选择要添加背景音乐的视频:",
            choices=[Choice(str(v), v.name) for v in videos],
        ).execute()
        video_path = Path(video_choice)
    else:
        video_path = videos[0]

    music_path, music_volume, keep_audio = _prompt_bgm_options()

    # 音频淡入淡出
    audio_fade_choice = inquirer.select(
        message="是否进行音频淡入淡出?",
        choices=[
            Choice("0", "🔊 不处理"),
            Choice("1.0", "🔉 1 秒"),
            Choice("2.0", "🔉 2 秒 (推荐)"),
            Choice("3.0", "🔉 3 秒"),
            Choice("custom", "✏️  自定义"),
        ],
        default="2.0",
    ).execute()

    audio_fade_in = 0.0
    audio_fade_out = 0.0
    if audio_fade_choice == "custom":
        audio_fade_in = float(inquirer.text(message="音频首部淡入 (秒):").execute())
        audio_fade_out = float(inquirer.text(message="音频尾部淡出 (秒):").execute())
    else:
        audio_fade_in = audio_fade_out = float(audio_fade_choice)

    if _confirm_action("确认添加背景音乐?"):
        add_bgm_to_video(
            video_path=video_path,
            music_path=music_path,
            music_volume=music_volume,
            keep_original_audio=keep_audio,
            audio_fade_in=audio_fade_in,
            audio_fade_out=audio_fade_out,
        )


def menu_concat(videos: list[Path]):
    """拼接视频菜单"""
    if len(videos) < 2:
        print("❌ 拼接需要至少 2 个视频")
        return

    print(f"\n将按以下顺序拼接 {len(videos)} 个视频:")
    
    for i, v in enumerate(videos, 1):
        print(f"  {i}. {v.name}")
    print()

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

    # 音频过渡 (避免接缝爆音) 或静音
    mute_audio = False
    audio_fade_choice = inquirer.select(
        message="如何处理音频?",
        choices=[
            Choice("2.0", "🔉 默认 (保留原声，平滑过渡 2 秒，推荐)"),
            Choice("0", "🔊 保持原声不变 (直接拼接)"),
            Choice("1.0", "🔉 平滑过渡 1 秒"),
            Choice("3.0", "🔉 平滑过渡 3 秒"),
            Choice("custom", "✏️  自定义过渡"),
            Choice("mute", "🔇 消除原声 (静音输出)"),
        ],
        default="2.0",
    ).execute()
    
    audio_fade_in = 0.0
    audio_fade_out = 0.0
    if audio_fade_choice == "mute":
        mute_audio = True
    elif audio_fade_choice == "custom":
        audio_fade_in = float(inquirer.text(message="音频首部淡入 (秒):").execute())
        audio_fade_out = float(inquirer.text(message="音频尾部淡出 (秒):").execute())
    else:
        audio_fade_in = audio_fade_out = float(audio_fade_choice)

    if inquirer.confirm(message=f"确认拼接 {len(videos)} 个视频?", default=True).execute():
        concat_videos(
            video_paths=videos,
            transition=transition,
            transition_duration=transition_duration,
            audio_fade_in=audio_fade_in,
            audio_fade_out=audio_fade_out,
            mute_audio=mute_audio,
        )


def menu_subtitle(media: list[Path]):
    """字幕菜单"""
    from tools.subtitle.ffmpeg_subtitle import SUBTITLE_STYLES, burn_subtitles
    from tools.subtitle.tts_dubbing import TTS_VOICES, dub_video_with_tts
    from tools.subtitle.whisper_transcribe import WHISPER_MODELS, transcribe_video

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

        # 双语字幕
        bilingual = inquirer.confirm(message="是否生成双语字幕?", default=True).execute()
        target_lang = None
        if bilingual:
            # 默认翻译目标语言：中文
            target_lang = inquirer.select(
                message="翻译目标语言:",
                choices=[
                    Choice("zh", "🇨🇳 中文"),
                    Choice("en", "🇬🇧 英语"),
                ],
                default="zh",
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
            default="large",
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
            transcribe_video(v, model_name=model_name, language=lang,
                             bilingual=bilingual, target_lang=target_lang)

        elif mode == "burn":
            burn_subtitles(v, subtitle_path=srt_path, style=style)

        elif mode == "oneclick":
            # 先识别
            result = transcribe_video(v, model_name=model_name, language=lang,
                                      bilingual=bilingual, target_lang=target_lang)
            srt_file = result["output"]
            # 再烧录
            burn_subtitles(v, subtitle_path=srt_file, style=style)
            
        elif mode == "tts":
            # 视频配音
            dub_video_with_tts(v, srt_path=srt_path, voice_key=voice_key)


def menu_mv():
    """歌词 MV 生成菜单"""
    from tools.mv import generate_mv
    from tools.subtitle.whisper_transcribe import WHISPER_MODELS, transcribe_video
    
    music_files = list(MUSIC_DIR.glob("*.*"))
    audio_extensions = {".mp3", ".wav", ".flac", ".m4a", ".aac"}
    audio_files = [f for f in music_files if f.suffix.lower() in audio_extensions]
    
    if not audio_files:
         print(f"❌ 未在 {MUSIC_DIR.name}/ 目录中找到音频文件")
         print("请先将音乐放入 music/ 目录中。")
         return
         
    audio_choices = [Choice(f, f.name) for f in audio_files]
    music_path = inquirer.select(
        message="选择音乐文件:",
        choices=audio_choices,
    ).execute()
    
    ratio = inquirer.select(
        message="视频比例:",
        choices=[
            Choice("9:16", "📱 9:16 竖屏 (1080x1920)"),
            Choice("16:9", "🖥️  16:9 横屏 (1920x1080)"),
        ],
        default="9:16",
    ).execute()
    
    if ratio == "16:9":
        width, height = 1920, 1080
    else:
        width, height = 1080, 1920
        
    lyric_mode = inquirer.select(
        message="歌词来源:",
        choices=[
            Choice("auto", "🎙️ Whisper 自动识别 (推荐)"),
            Choice("manual", "📝 加载已有字幕文件 (.srt / .lrc)"),
        ]
    ).execute()
    
    whisper_segments = None
    lyrics_path = None
    
    if lyric_mode == "auto":
        model_choices = [
            Choice(key, f"{item['name']} ({item['size']})")
            for key, item in WHISPER_MODELS.items()
        ]
        model_name = inquirer.select(
            message="选择 Whisper 模型:",
            choices=model_choices,
            default="small",
        ).execute()
        
        print("💡 正在使用 Whisper 识别歌词...")
        result = transcribe_video(music_path, model_name=model_name)
        whisper_segments = result["segments"]
        print(f"✅ 识别完成, 共 {len(whisper_segments)} 句歌词。")
        
    else:
        path_str = inquirer.filepath(
            message="输入字幕文件路径 (.srt, .lrc):",
            validate=lambda x: Path(x).is_file(),
        ).execute()
        lyrics_path = Path(path_str)
        
    bg_choices = [Choice("none", "⬛ 纯黑背景")]
    
    images_dir = Path("images")
    if images_dir.is_dir():
        for img_file in images_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                bg_choices.append(Choice(str(img_file), f"🖼️  {img_file.name}"))
                
    bg_choices.append(Choice("path", "📂 指定其他背景图路径"))

    bg_mode = inquirer.select(
        message="背景图:",
        choices=bg_choices,
        default="none"
    ).execute()
    
    bg_image_path = None
    if bg_mode == "path":
        bg_path_str = inquirer.filepath(
            message="背景图片路径 (.jpg, .png):",
            validate=lambda x: Path(x).is_file(),
        ).execute()
        bg_image_path = Path(bg_path_str)
    elif bg_mode != "none":
        bg_image_path = Path(bg_mode)
        
    if _confirm_action("确认生成歌词 MV?"):
        generate_mv(
            music_path=music_path,
            lyrics_path=lyrics_path,
            whisper_segments=whisper_segments,
            width=width,
            height=height,
            bg_image_path=bg_image_path,
        )


def menu_compress(videos: list[Path]):
    """无损/视觉无损压缩菜单"""
    codec = inquirer.select(
        message="视频编码器:",
        choices=[
            Choice("libx264", "🔧 H.264 (兼容各类主流媒体平台, 稳定推荐)"),
            Choice("libx265", "🚀 H.265 (HEVC, 极致体积, 部分旧设备或环境可能不兼容)"),
        ],
        default="libx264",
    ).execute()
    
    strength = inquirer.select(
        message="压缩强度:",
        choices=[
            Choice(24, "🌟 标准平衡 (视觉几乎无损, 适合上传抖音/视频号等)"),
            Choice(28, "🗜️  体积极限 (画质极轻微降低, 体积最小化)"),
            Choice(18, "💎 极致高保真 (接近原画大小, 适合高清存档)"),
        ],
        default=24,
    ).execute()

    _maybe_show_file_list(videos)
    if _confirm_action(f"确认压缩 {len(videos)} 个视频?"):
        batch_compress(files=videos, codec=codec, crf=strength)


def menu_slideshow(images: list[Path]):
    """幻灯片 (Slideshow) 菜单"""
    from tools.slideshow.generator import (
        build_texts_from_filenames,
        discover_slideshow_groups,
        generate_slideshow,
        load_texts_from_caption_file,
    )
    from tools.concat.ffmpeg_concat import get_available_music
    
    mode = inquirer.select(
        message="幻灯片模式:",
        choices=[
            Choice("hotspot", "⚡ 热点批量模式 (按子目录一键出片)"),
            Choice("custom", "🎛️ 自定义模式 (原有流程)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if mode == "back":
        return

    # 背景音乐（两种模式共用）
    available_music = get_available_music()
    music_choices = [Choice("none", "🚫 不添加音乐")]
    if available_music:
        for m in available_music:
            music_choices.append(Choice(str(m), f"🎵 {m.stem}"))
    music_choices.append(Choice("custom", "📂 指定音乐文件路径"))
    music_choice = inquirer.select(message="选择背景音乐:", choices=music_choices).execute()
    if music_choice == "custom":
        selected_music_path = inquirer.filepath(
            message="音乐文件路径:",
            validate=lambda x: Path(x).is_file(),
        ).execute()
    else:
        selected_music_path = music_choice

    music_volume = 0.3
    if selected_music_path != "none":
        music_volume = float(inquirer.text(message="音乐音量 (0.0~1.0):", default="0.3").execute())

    # 过渡效果（两种模式共用）
    from tools.concat.ffmpeg_concat import TRANSITION_PRESETS
    transition_choices = [
        Choice(key, preset["name"]) for key, preset in TRANSITION_PRESETS.items()
    ]
    transition = inquirer.select(
        message="选择图片间过渡效果:",
        choices=transition_choices,
        default="none",
    ).execute()
    
    transition_duration = 0.0
    if transition != "none":
        transition_duration = float(inquirer.text(message="过渡时长 (秒):", default="1.0").execute())

    if mode == "custom":
        if len(images) < 1:
            print("❌ 未找到图片")
            return

        print(f"\n将按以下顺序将 {len(images)} 张图片拼接为视频:")
        for i, img in enumerate(images, 1):
            print(f"  {i}. {img.name}")
        print()

        res = inquirer.select(
            message="选择目标分辨率比例:",
            choices=[
                Choice((1080, 1920), "📱 9:16 竖屏 (1080x1920)"),
                Choice((1920, 1080), "🖥️  16:9 横屏 (1920x1080)"),
            ],
            default=(1080, 1920),
        ).execute()

        text_input = inquirer.text(message="输入文案 (多句话用 | 分隔，空则不加):").execute()
        texts = [t.strip() for t in text_input.split("|")] if text_input.strip() else []

        if len(texts) > 0 and len(texts) < len(images):
            print(f"💡 提示：只提供了 {len(texts)} 句文案，后面的图片将不带文案。")

        duration = float(inquirer.text(message="每张图片展示时长 (秒):", default="5.0").execute())
        if not _confirm_action(f"确认生成 {len(images)} 张图片的 Slideshow?"):
            return

        generate_slideshow(
            image_paths=images,
            texts=texts,
            resolution=res,
            duration_per_image=duration,
            music_path=selected_music_path if selected_music_path != "none" else None,
            music_volume=music_volume,
            transition=transition,
            transition_duration=transition_duration,
        )
        return

    # 热点批量模式: 按目录分组一键批量生成
    source_dir = INPUT_DIR
    source_mode = inquirer.select(
        message="选择批量图片来源:",
        choices=[
            Choice("input_subdirs", f"📂 使用 {INPUT_DIR.name}/ 下一级子目录"),
            Choice("custom_dir", "📁 指定目录 (按其子目录分组)"),
        ],
        default="input_subdirs",
    ).execute()
    if source_mode == "custom_dir":
        source_dir = Path(inquirer.filepath(
            message="输入目录路径:",
            validate=lambda x: Path(x).is_dir(),
            only_directories=True,
        ).execute())

    groups = discover_slideshow_groups(source_dir, recursive=False)
    if not groups:
        print(f"❌ 在 {source_dir} 下未发现包含图片的子目录")
        print("💡 目录结构建议: 一个热点一个文件夹，每个文件夹放该热点的多张图片")
        return

    print(f"\n已发现 {len(groups)} 个可批量出片目录:")
    for idx, (group_dir, group_images) in enumerate(groups, 1):
        print(f"  {idx}. {group_dir.name} ({len(group_images)} 张)")
    print()

    res = inquirer.select(
        message="目标平台预设:",
        choices=[
            Choice((1080, 1920), "📱 抖音/视频号 9:16 竖屏 (1080x1920)"),
            Choice((1920, 1080), "🖥️ 横屏 16:9 (1920x1080)"),
        ],
        default=(1080, 1920),
    ).execute()

    duration = float(inquirer.text(message="每张图片展示时长 (秒):", default="3.0").execute())
    caption_mode = inquirer.select(
        message="文案来源:",
        choices=[
            Choice("filename", "🧾 使用图片文件名作为文案"),
            Choice("captions_file", "📄 使用各目录 captions.txt (每行一句)"),
            Choice("none", "🚫 不添加文案"),
        ],
        default="filename",
    ).execute()

    if not _confirm_action(f"确认批量生成 {len(groups)} 条热点 Slideshow?"):
        return

    batch_dir = OUTPUT_DIR / "slideshow" / f"hotspot_batch_{datetime.now().strftime('%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    for group_dir, group_images in groups:
        if caption_mode == "filename":
            texts = build_texts_from_filenames(group_images)
        elif caption_mode == "captions_file":
            texts = load_texts_from_caption_file(group_dir / "captions.txt")
            if len(texts) < len(group_images):
                texts = texts + [""] * (len(group_images) - len(texts))
        else:
            texts = []

        output_path = batch_dir / f"{group_dir.name}.mp4"
        try:
            generate_slideshow(
                image_paths=group_images,
                texts=texts,
                resolution=res,
                duration_per_image=duration,
                music_path=selected_music_path if selected_music_path != "none" else None,
                music_volume=music_volume,
                transition=transition,
                transition_duration=transition_duration,
                output_path=output_path,
            )
            success += 1
        except Exception as e:
            failed += 1
            print(f"❌ 生成失败 [{group_dir.name}]: {e}")

    print(f"\n✅ 批量完成: 成功 {success} 条, 失败 {failed} 条")
    print(f"📁 输出目录: {batch_dir}")


def menu_pdf_split():
    """PDF 拆分菜单"""
    from tools.pdf_split.splitter import (
        get_pdf_info,
        split_by_every_page,
        split_by_range,
        split_by_chunk,
        extract_pages,
        _ensure_pypdf,
    )

    if not _ensure_pypdf():
        return

    # 选择 PDF 文件
    pdf_source = inquirer.select(
        message="选择 PDF 来源:",
        choices=[
            Choice("scan", "📂 扫描 input/ 目录"),
            Choice("path", "📄 指定 PDF 文件路径"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if pdf_source == "back":
        return

    pdf_files: list[Path] = []

    if pdf_source == "scan":
        pdf_files = sorted(
            f for f in INPUT_DIR.glob("*.pdf")
            if f.is_file()
        )
        if not pdf_files:
            print(f"❌ 未在 {INPUT_DIR.name}/ 目录中找到 PDF 文件")
            return
    else:
        path_str = inquirer.filepath(
            message="输入 PDF 文件路径:",
            validate=lambda x: Path(x).is_file() and Path(x).suffix.lower() == ".pdf",
        ).execute()
        pdf_files = [Path(path_str)]

    # 如果有多个 PDF, 选择一个
    if len(pdf_files) > 1:
        pdf_choices = [Choice(str(f), f.name) for f in pdf_files]
        selected = inquirer.select(
            message="选择要拆分的 PDF:",
            choices=pdf_choices,
        ).execute()
        pdf_path = Path(selected)
    else:
        pdf_path = pdf_files[0]

    # 显示 PDF 信息
    info = get_pdf_info(pdf_path)
    print(f"\n📄 文件: {pdf_path.name}")
    print(f"   页数: {info['pages']}")
    print(f"   大小: {info['size_mb']:.1f} MB")
    if info.get("title"):
        print(f"   标题: {info['title']}")
    print()

    if info["pages"] < 1:
        print("❌ PDF 文件无有效页面")
        return

    # 选择拆分模式
    mode = inquirer.select(
        message="选择拆分模式:",
        choices=[
            Choice("every", "📑 按页拆分 (每页生成一个 PDF)"),
            Choice("range", "📐 按页码范围拆分 (如 1-5, 6-10)"),
            Choice("chunk", f"📦 按固定页数拆分 (每 N 页一份)"),
            Choice("extract", "🔍 提取指定页码 (合并为一个 PDF)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if mode == "back":
        return

    total_pages = info["pages"]

    if mode == "every":
        if _confirm_action(f"确认将 {total_pages} 页拆分为 {total_pages} 个文件?"):
            result = split_by_every_page(pdf_path)
            print(f"\n✅ 拆分完成: {len(result['output'])} 个文件")
            print(f"📁 输出目录: {OUTPUT_PDF_SPLIT}")

    elif mode == "range":
        print(f"请输入页码范围 (总共 {total_pages} 页)")
        print("格式: 1-5,6-10,11-20  (逗号分隔多个范围)")
        range_str = inquirer.text(
            message="页码范围:",
            default=f"1-{min(total_pages, 10)}",
        ).execute()

        # 解析范围
        ranges = []
        try:
            for part in range_str.split(","):
                part = part.strip()
                if "-" in part:
                    s, e = part.split("-", 1)
                    ranges.append((int(s.strip()), int(e.strip())))
                else:
                    # 单页: "5" -> (5, 5)
                    p = int(part)
                    ranges.append((p, p))
        except ValueError:
            print("❌ 格式错误，请使用 1-5,6-10 格式")
            return

        if not ranges:
            print("❌ 未指定有效范围")
            return

        range_desc = ", ".join(f"{s}-{e}" for s, e in ranges)
        if _confirm_action(f"确认拆分范围 [{range_desc}]?"):
            result = split_by_range(pdf_path, ranges)
            print(f"\n✅ 拆分完成: {len(result['output'])} 个文件")
            print(f"📁 输出目录: {OUTPUT_PDF_SPLIT}")

    elif mode == "chunk":
        chunk_size = int(inquirer.number(
            message=f"每份页数 (总共 {total_pages} 页):",
            default=10,
            min_allowed=1,
            max_allowed=total_pages,
        ).execute())

        import math
        num_files = math.ceil(total_pages / chunk_size)
        if _confirm_action(f"确认将 {total_pages} 页按每 {chunk_size} 页拆分为 {num_files} 份?"):
            result = split_by_chunk(pdf_path, chunk_size)
            print(f"\n✅ 拆分完成: {len(result['output'])} 个文件")
            print(f"📁 输出目录: {OUTPUT_PDF_SPLIT}")

    elif mode == "extract":
        print(f"请输入要提取的页码 (总共 {total_pages} 页)")
        print("格式: 1,3,5,7  或  1-5,8,10-12  (支持范围和单页混合)")
        pages_str = inquirer.text(
            message="页码:",
            default="1",
        ).execute()

        # 解析页码 (支持 1,3,5 和 1-5 混合格式)
        pages = []
        try:
            for part in pages_str.split(","):
                part = part.strip()
                if "-" in part:
                    s, e = part.split("-", 1)
                    pages.extend(range(int(s.strip()), int(e.strip()) + 1))
                else:
                    pages.append(int(part))
        except ValueError:
            print("❌ 格式错误，请使用 1,3,5 或 1-5 格式")
            return

        if not pages:
            print("❌ 未指定有效页码")
            return

        # 去重排序
        pages = sorted(set(pages))
        pages_preview = ", ".join(str(p) for p in pages[:10])
        if len(pages) > 10:
            pages_preview += f" ... (共 {len(pages)} 页)"

        if _confirm_action(f"确认提取页码 [{pages_preview}]?"):
            result = extract_pages(pdf_path, pages)
            if result["output"]:
                print(f"\n✅ 提取完成: {result['pages']} 页")
                print(f"📁 输出: {result['output']}")


def menu_clean():
    """清理 input 和 output 目录菜单"""
    target = inquirer.select(
        message="选择清理范围:",
        choices=[
            Choice("all", "🗑️  清理全部 (input + output)"),
            Choice("input", "📥 仅清理 input"),
            Choice("output", "📤 仅清理 output"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if target == "back":
        return

    dirs_to_clean = []
    if target in ("all", "input"):
        dirs_to_clean.append(INPUT_DIR)
    if target in ("all", "output"):
        dirs_to_clean.append(OUTPUT_DIR)

    # 统计文件数量
    total_files = 0
    total_size = 0
    for d in dirs_to_clean:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    total_files += 1
                    total_size += f.stat().st_size

    if total_files == 0:
        print("✅ 目录已经是空的，无需清理")
        return

    size_mb = total_size / (1024 * 1024)
    print(f"\n⚠️  即将删除 {total_files} 个文件 (共 {size_mb:.1f} MB)")
    for d in dirs_to_clean:
        print(f"   📂 {d}")

    if not inquirer.confirm(message="确认删除? 此操作不可恢复!", default=False).execute():
        print("✅ 已取消")
        return

    deleted = 0
    for d in dirs_to_clean:
        if not d.exists():
            continue
        for item in sorted(d.rglob("*"), reverse=True):
            try:
                if item.is_file():
                    item.unlink()
                    deleted += 1
                elif item.is_dir() and item != d:
                    # 删除空子目录 (保留根目录本身)
                    try:
                        item.rmdir()
                    except OSError:
                        pass  # 目录非空，跳过
            except Exception as e:
                print(f"  ⚠️  无法删除 {item.name}: {e}")

    print(f"\n✅ 清理完成，共删除 {deleted} 个文件")


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
                Choice("compress", "🗜️  无损压缩 (Compress)"),
                Choice("upscale", "🆙 高清重置 (Upscale)"),
                Choice("interpolate", "⏯️  帧数补充 (Interpolate)"),
                Choice("convert", "🔄 格式转换 (Convert)"),
                Choice("filter", "🎨 滤镜效果 (Filter)"),
                Choice("crop", "✂️  裁切比例 (Crop)"),
                Choice("concat", "🎬 拼接视频 (Concat)"),
                Choice("slideshow", "📸 幻灯片 (Slideshow)"),
                Choice("bgm", "🎵 添加背景音乐 (BGM)"),
                Choice("subtitle", "📝 字幕 (Subtitle)"),
                Choice("mv", "🎵 歌词 MV 生成 (Lyric MV)"),
                Choice("pdf_split", "📄 PDF 拆分 (PDF Split)"),
                Separator(),
                Choice("clean", "🧹 清理文件 (Clean)"),
                Choice("exit", "❌ 退出"),
            ],
            default="watermark",
        ).execute()

        if module == "exit":
            print("Bye!")
            sys.exit(0)

        if module == "clean":
            menu_clean()
            print()
            if not inquirer.confirm(message="继续其他操作?", default=True).execute():
                break
            continue

        if module == "mv":
            menu_mv()
            print()
            if not inquirer.confirm(message="继续其他操作?", default=True).execute():
                break
            continue

        if module == "pdf_split":
            menu_pdf_split()
            print()
            if not inquirer.confirm(message="继续其他操作?", default=True).execute():
                break
            continue

        # 获取输入
        if module in ("add_watermark", "convert", "filter", "crop", "subtitle", "slideshow"):
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
            elif module == "filter":
                menu_filter(media)
            elif module == "crop":
                menu_crop(media)
            elif module == "subtitle":
                menu_subtitle(media)
            elif module == "slideshow":
                images = [f for f in media if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".gif"}]
                if not images:
                    print("❌ 未找到图片文件")
                    continue
                menu_slideshow(images)
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
            elif module == "bgm":
                menu_add_bgm(videos)
            elif module == "compress":
                menu_compress(videos)

        print()
        if not inquirer.confirm(message="继续其他操作?", default=True).execute():
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消操作")
        sys.exit(0)
