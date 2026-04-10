"""歌词 MV 生成菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from config import AUDIO_EXTENSIONS, IMAGE_EXTENSIONS, MUSIC_DIR
from menus._prompts import confirm_action


def menu_mv():
    """歌词 MV 生成菜单"""
    from tools.mv import generate_mv
    from tools.subtitle.whisper_transcribe import WHISPER_MODELS, transcribe_video
    
    music_files = list(MUSIC_DIR.glob("*.*"))
    audio_files = [f for f in music_files if f.suffix.lower() in AUDIO_EXTENSIONS]
    
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
            if img_file.is_file() and img_file.suffix.lower() in IMAGE_EXTENSIONS:
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
        
    if confirm_action("确认生成歌词 MV?"):
        generate_mv(
            music_path=music_path,
            lyrics_path=lyrics_path,
            whisper_segments=whisper_segments,
            width=width,
            height=height,
            bg_image_path=bg_image_path,
        )
