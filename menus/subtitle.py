"""字幕菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from config import VIDEO_EXTENSIONS


def menu_subtitle(media: list[Path]):
    """字幕菜单"""
    from tools.subtitle.ffmpeg_subtitle import SUBTITLE_STYLES, burn_subtitles
    from tools.subtitle.tts_dubbing import TTS_VOICES, dub_video_with_tts
    from tools.subtitle.whisper_transcribe import WHISPER_MODELS, transcribe_video

    # 只处理视频
    videos = [f for f in media if f.suffix.lower() in VIDEO_EXTENSIONS]
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

        try:
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
        except Exception as e:
            print(f"  ❌ 处理失败: {v.name} — {e}")
