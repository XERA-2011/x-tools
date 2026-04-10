"""超分菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from menus._prompts import confirm_action, maybe_show_file_list


def menu_upscale(videos: list[Path]):
    """超分菜单"""
    from tools.upscale.batch import batch_upscale_ffmpeg, batch_upscale_realesrgan

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

    maybe_show_file_list(videos)

    if confirm_action(f"确认放大 {len(videos)} 个视频?"):
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
