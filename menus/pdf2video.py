"""
PDF 转视频交互菜单 (TUI)

功能:
  - 扫描 input/ 目录下的 PDF 文件或手动指定路径
  - 挑选 Edge-TTS 声音音色与翻页留白节奏
  - 可选配置背景音乐混音与目标分辨率
  - 默认保持幻灯片画面纯净，同步导出对齐的 .srt 字幕文件
"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import INPUT_DIR, PDF_EXTENSIONS
from tools.common import logger
from tools.concat.ffmpeg_concat import get_available_music
from tools.pdf2video.generator import generate_pdf_video


def _scan_pdfs(directory: Path) -> list[Path]:
    """扫描目录下的 PDF 文件"""
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in PDF_EXTENSIONS
    )


def menu_pdf2video():
    """PDF 转视频主菜单"""
    print("\n" + "=" * 50)
    print("📄  PDF 转视频 (PDF to Video)")
    print("=" * 50)

    # 1. 选择 PDF 输入文件
    candidates = _scan_pdfs(INPUT_DIR)
    choices = [Choice(str(p), f"{p.name} ({p.stat().st_size / 1024 / 1024:.1f} MB)") for p in candidates]
    choices.append(Separator())
    choices.append(Choice("__manual__", "📂 手动输入文件绝对路径"))
    choices.append(Choice("__back__", "🔙 返回主菜单"))

    pdf_choice = inquirer.select(
        message="请选择要转换的 PDF 文件:",
        choices=choices,
    ).execute()

    if pdf_choice == "__back__":
        return

    if pdf_choice == "__manual__":
        manual_path = inquirer.filepath(
            message="请输入 PDF 文件的完整路径:",
            validate=lambda x: Path(x).is_file() and Path(x).suffix.lower() in PDF_EXTENSIONS,
            invalid_message="文件不存在或不是 PDF 格式",
        ).execute()
        pdf_path = Path(manual_path)
    else:
        pdf_path = Path(pdf_choice)

    # 2. 选择语音配音音色
    voice_choices = [
        Choice("zh-CN-YunxiNeural", "云希 (男声, 阳光沉稳解说推荐)"),
        Choice("zh-CN-YunyangNeural", "云扬 (男声, 专业严谨/新闻政企主播推荐)"),
        Choice("zh-CN-XiaoxiaoNeural", "晓晓 (女声, 亲和自然/标准播音推荐)"),
        Choice("zh-CN-YunjianNeural", "云健 (男声, 影视解说/纪录片风)"),
        Choice("zh-CN-XiaoyiNeural", "晓伊 (女声, 温柔甜美/生动)"),
        Choice("zh-CN-liaoning-XiaobeiNeural", "东北晓北 (女声, 东北方言/幽默短视频风)"),
        Choice("zh-CN-shaanxi-XiaoniNeural", "陕西晓妮 (女声, 陕西特色方言)"),
        Choice("zh-TW-HsiaoChenNeural", "晓臻 (台湾腔女声, 柔和甜润)"),
        Choice("zh-HK-HiuMaanNeural", "晓曼 (粤语女声)"),
    ]
    # 如果系统配置里有更多声音，允许额外选择
    voice = inquirer.select(
        message="选择朗读配音音色:",
        choices=voice_choices,
        default="zh-CN-YunxiNeural",
    ).execute()

    # 3. 语速与翻页留白
    page_padding = float(
        inquirer.text(
            message="每页朗读完成后的停顿留白时长 (秒):",
            default="0.8",
            validate=lambda x: x.replace(".", "", 1).isdigit() and float(x) >= 0.0,
            invalid_message="请输入 >= 0 的数字",
        ).execute()
    )

    # 4. 视频分辨率
    resolution_choice = inquirer.select(
        message="视频输出分辨率:",
        choices=[
            Choice("1080p", "1080P 高清 (1920x1080)"),
            Choice("2k", "2K 超清 (2560x1440)"),
            Choice("original", "PDF 原始比例直接等比缩放"),
        ],
        default="1080p",
    ).execute()

    # 5. 背景音乐
    music_files = get_available_music()
    bgm_path = None
    bgm_volume = 0.15

    add_bgm = inquirer.confirm(
        message="是否添加背景音乐 (BGM)?",
        default=False,
    ).execute()

    if add_bgm:
        if music_files:
            bgm_choices = [Choice(str(p), p.name) for p in music_files]
            bgm_choices.append(Separator())
            bgm_choices.append(Choice("__manual__", "📂 手动输入音频路径"))
            chosen_bgm = inquirer.select(
                message="选择背景音乐:",
                choices=bgm_choices,
            ).execute()

            if chosen_bgm == "__manual__":
                manual_bgm = inquirer.filepath(
                    message="请输入音频文件路径:",
                    validate=lambda x: Path(x).is_file(),
                ).execute()
                bgm_path = Path(manual_bgm)
            else:
                bgm_path = Path(chosen_bgm)
        else:
            print("💡 music/ 目录下未发现音乐文件，如需请手动指定")
            manual_bgm = inquirer.filepath(
                message="请输入音频文件路径 (留空跳过):",
            ).execute()
            if manual_bgm and Path(manual_bgm).is_file():
                bgm_path = Path(manual_bgm)

        if bgm_path:
            bgm_vol_str = inquirer.text(
                message="背景音乐音量 (0.05~1.0, 推荐 0.15):",
                default="0.15",
                validate=lambda x: x.replace(".", "", 1).isdigit() and 0.0 < float(x) <= 1.0,
            ).execute()
            bgm_volume = float(bgm_vol_str)

    # 6. 字幕烧录选项
    burn_subtitles = inquirer.confirm(
        message="是否在视频画面底部烧录解说字幕 (推荐是)?",
        default=True,
    ).execute()

    subtitle_style = "white_box"
    subtitle_layout = "split_phrases"
    if burn_subtitles:
        subtitle_style = inquirer.select(
            message="选择字幕颜色类型:",
            choices=[
                Choice("white_box", "⬜ 白字半透明黑底(圆角框) (沉浸质感)"),
                Choice("black_transparent", "🖤 黑字透明底 (简约白边微轮廓)"),
            ],
            default="white_box",
        ).execute()

        subtitle_layout = inquirer.select(
            message="选择字幕长句排版方案:",
            choices=[
                Choice("split_phrases", "✨ 方案 1: 标点短句拆分流转 (推荐，8~18字动态切换，小巧大字)"),
                Choice("double_line", "📑 方案 2: 智能双行折行卡片 (超长句自动对称折两行，紧凑贴合底框)"),
                Choice("single_line_scale", "🔍 方案 3: 纯单行自适应字号 (长句绝对不换行，动态等比缩小字号)"),
            ],
            default="split_phrases",
        ).execute()

    # 7. 确认生成
    style_label = "白字半透明黑底(圆角框)" if subtitle_style == "white_box" else "黑字透明底"
    layout_label_map = {
        "split_phrases": "方案1: 标点短句拆分流转",
        "double_line": "方案2: 智能双行折行卡片",
        "single_line_scale": "方案3: 纯单行自适应字号",
        "auto_scale": "方案3: 纯单行自适应字号",
        "fixed_bar": "方案3: 纯单行自适应字号",
    }
    layout_label = layout_label_map.get(subtitle_layout, "方案1: 标点短句拆分流转")
    burn_desc = f"是 ({style_label} | {layout_label})" if burn_subtitles else "否 (保持画面纯净，输出独立 SRT)"
    print("\n--- 任务配置清单 ---")
    print(f"📄 输入文件: {pdf_path.name}")
    print(f"🎙️  朗读音色: {voice}")
    print(f"⏸️  翻页留白: {page_padding} 秒")
    print(f"🖥️  输出规格: {resolution_choice}")
    print(f"🎵 背景音乐: {bgm_path.name if bgm_path else '无'}")
    print(f"📝 画面烧录: {burn_desc}")
    print("--------------------\n")

    confirm = inquirer.confirm(message="确认开始生成视频吗?", default=True).execute()
    if not confirm:
        print("已取消")
        return

    try:
        result = generate_pdf_video(
            pdf_path=pdf_path,
            voice=voice,
            page_padding=page_padding,
            bgm_path=bgm_path,
            bgm_volume=bgm_volume,
            resolution=resolution_choice,
            burn_subtitles=burn_subtitles,
            subtitle_style=subtitle_style,
            subtitle_layout=subtitle_layout,
        )
        print("\n🎉 处理完成!")
        print(f"   🎬 视频文件: {result['output']}")
        if result.get("srt"):
            print(f"   📝 SRT 字幕: {result['srt']}")
        print(f"   ⏱️  总时长:   {result['duration']} 秒")
        print(f"   📄 总页数:   {result['slides_count']} 页")
        print(f"   📦 文件大小: {result['size_mb']} MB\n")
    except Exception as e:
        logger.error(f"生成 PDF 视频失败: {e}")
        print(f"\n❌ 生成失败: {e}\n")
