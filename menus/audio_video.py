"""BGM 与拼接菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from menus._prompts import confirm_action


def _prompt_bgm_options() -> tuple[str, float, bool]:
    from tools.concat.ffmpeg_concat import get_available_music

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
    from tools.bgm.ffmpeg_bgm import add_bgm_to_video

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

    if confirm_action("确认添加背景音乐?"):
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
    from tools.concat.ffmpeg_concat import TRANSITION_PRESETS, concat_videos

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
