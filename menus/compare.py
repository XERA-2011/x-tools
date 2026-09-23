"""画面对比与分屏拼接菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from config import INPUT_DIR
from menus._prompts import confirm_action
from tools.common import scan_videos
from tools.compare.ffmpeg_compare import (
    AudioMode,
    LayoutType,
    create_video_comparison,
    extract_model_name_from_path,
)


def _select_compare_videos(videos: list[Path] | None = None) -> list[Path] | None:
    """选择参与对比的 2~4 个视频"""
    if not videos:
        # 扫描 input/
        available = scan_videos(INPUT_DIR, recursive=False)
    else:
        available = videos

    if not available:
        print("❌ 未在 input/ 目录找到视频文件，请将对比视频放入 input/ 目录后重试。")
        return None

    if len(available) < 2:
        print("❌ 对比功能需要至少 2 个视频文件。")
        return None

    # 如果正好是 2~4 个，直接展示并询问是否全部加入对比
    if 2 <= len(available) <= 4:
        print(f"\n检测到 {len(available)} 个视频素材:")
        for i, v in enumerate(available, 1):
            print(f"  {i}. {v.name}")
        print()
        if inquirer.confirm(message="使用上述所有视频进行同屏对比?", default=True).execute():
            return available

    # 超过 4 个或用户想挑选: 使用复选框
    choices = [Choice(str(v), v.name) for v in available]
    selected_strs = inquirer.checkbox(
        message="请勾选 2~4 个参与对比的视频 (空格勾选，回车确认):",
        choices=choices,
        validate=lambda result: 2 <= len(result) <= 4,
        invalid_message="请至少勾选 2 个、最多勾选 4 个视频",
    ).execute()

    if not selected_strs:
        return None
    return [Path(s) for s in selected_strs]


def menu_compare(videos: list[Path] | None = None):
    """画面对比与分屏拼接主交互菜单"""
    selected_videos = _select_compare_videos(videos)
    if not selected_videos:
        return

    num = len(selected_videos)
    print(f"\n已选择 {num} 个视频进行同屏对比:")
    for i, v in enumerate(selected_videos, 1):
        print(f"  {i}. {v.name}")
    print()

    # 1. 布局选择
    layout_choice: LayoutType = inquirer.select(
        message="选择同屏排版布局:",
        choices=[
            Choice("auto", "✨ 智能推荐 (横屏视频自动推荐 9:16 竖屏叠排，适合短视频)"),
            Choice("vertical_stack", "📱 竖屏 9:16 (1080x1920) 叠排 (适合抖音/小红书/Shorts)"),
            Choice("horizontal_stack", "🖥️  宽屏 16:9 (1920x1080) 横排"),
            Choice("grid_2x2", "🔲 网格 2x2 四宫格 (宽屏)"),
        ],
        default="auto",
    ).execute()

    # 2. 右上角模型角标设置
    default_labels = [extract_model_name_from_path(p) for p in selected_videos]
    print("\n已自动提取各视频右上角模型角标:")
    for i, (v, lbl) in enumerate(zip(selected_videos, default_labels), 1):
        print(f"  {i}. {v.name} → 🏷️  [{lbl}]")
    print()

    edit_labels = inquirer.confirm(
        message="是否需要手动修改角标文本?",
        default=False,
    ).execute()

    final_labels = list(default_labels)
    if edit_labels:
        for i in range(num):
            new_val = inquirer.text(
                message=f"视频 {i+1} ({selected_videos[i].name}) 角标:",
                default=final_labels[i],
            ).execute()
            if new_val.strip():
                final_labels[i] = new_val.strip()

    # 3. 时长对齐策略
    duration_mode = inquirer.select(
        message="时长对齐策略:",
        choices=[
            Choice("shortest", "⏱️  以最短视频为准 (多出的部分截断，推荐)"),
            Choice("longest", "⏳ 以最长视频为准 (较短的视频末尾静止定格)"),
        ],
        default="shortest",
    ).execute()

    # 4. 音频模式
    audio_choice: AudioMode = inquirer.select(
        message="音频处理方式:",
        choices=[
            Choice("first", "🔊 保留首个视频原声 (推荐)"),
            Choice("mute", "🔇 全部静音 (纯画面对比)"),
            Choice("mix", "🎛️  混合所有视频原声"),
            Choice("bgm", "🎵 自选背景音乐 (BGM)"),
        ],
        default="first",
    ).execute()

    music_path = None
    music_volume = 0.3
    if audio_choice == "bgm":
        from tools.concat.ffmpeg_concat import get_available_music
        available_music = get_available_music()
        music_choices = [Choice(str(m), f"🎵 {m.stem}") for m in available_music]
        music_choices.append(Choice("custom", "📂 指定其他音频文件"))
        chosen = inquirer.select(message="选择背景音乐:", choices=music_choices).execute()
        if chosen == "custom":
            music_path = inquirer.filepath(
                message="音频文件路径:",
                validate=lambda x: Path(x).is_file(),
            ).execute()
        else:
            music_path = chosen
        music_volume = float(inquirer.text(message="BGM 音量 (0.0~1.0):", default="0.3").execute())

    # 5. 确认并执行
    print()
    if not confirm_action("确认开始合成对比视频?"):
        print("已取消操作")
        return

    try:
        res = create_video_comparison(
            video_paths=selected_videos,
            labels=final_labels,
            layout=layout_choice,
            duration_mode=duration_mode,
            audio_mode=audio_choice,
            music_path=music_path,
            music_volume=music_volume,
        )
        print(
            f"\n🎉 画面对比视频合成成功！\n"
            f"📂 输出文件: {res['output']}\n"
            f"⏱️  视频时长: {res['duration']}s | 大小: {res['size_mb']} MB\n"
        )
    except Exception as e:
        print(f"\n❌ 合成失败: {e}\n")
