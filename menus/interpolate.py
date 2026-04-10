"""插帧菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from menus._prompts import confirm_action


def menu_interpolate(videos: list[Path]):
    """插帧菜单"""
    from tools.interpolation.batch import batch_interpolate_ffmpeg, batch_interpolate_rife

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

    if confirm_action(f"确认处理 {len(videos)} 个视频?"):
        if engine == "ffmpeg":
            batch_interpolate_ffmpeg(
                videos=videos, target_fps=target_fps,
                mode=mode_val, mb_size=mb_size_val, search_param=search_param_val
            )
        else:
            batch_interpolate_rife(videos=videos, multiplier=multiplier)
