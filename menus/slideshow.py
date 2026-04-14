"""幻灯片菜单"""
from datetime import datetime
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import IMAGE_EXTENSIONS, INPUT_DIR, OUTPUT_DIR
from menus._prompts import confirm_action, get_input_media


def menu_slideshow():
    """幻灯片 (Slideshow) 菜单"""
    from tools.concat.ffmpeg_concat import TRANSITION_PRESETS, get_available_music
    from tools.slideshow.generator import (
        discover_slideshow_groups,
        generate_slideshow,
        load_texts_from_caption_file,
    )
    
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

    images = None
    if mode == "custom":
        media = get_input_media()
        if media is None:
            return
        images = [f for f in media if f.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            print("❌ 未找到图片文件")
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
        if not confirm_action(f"确认生成 {len(images)} 张图片的 Slideshow?"):
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
    print("  (文案来源：将自动读取各目录中的 captions.txt)\n")

    res = inquirer.select(
        message="目标平台预设:",
        choices=[
            Choice((1080, 1920), "📱 抖音/视频号 9:16 竖屏 (1080x1920)"),
            Choice((1920, 1080), "🖥️ 横屏 16:9 (1920x1080)"),
        ],
        default=(1080, 1920),
    ).execute()

    duration = float(inquirer.text(message="每张图片展示时长 (秒):", default="5.0").execute())


    if not confirm_action(f"确认批量生成 {len(groups)} 条热点 Slideshow?"):
        return

    batch_dir = OUTPUT_DIR / "slideshow" / f"hotspot_batch_{datetime.now().strftime('%m%d_%H%M%S')}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = 0
    for group_dir, group_images in groups:
        texts = load_texts_from_caption_file(group_dir / "captions.txt")
        if len(texts) < len(group_images):
            texts = texts + [""] * (len(group_images) - len(texts))

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
