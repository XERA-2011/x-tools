"""加水印菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import ADD_WATERMARK_TEXT
from menus._prompts import confirm_action, maybe_show_file_list


def menu_add_watermark(media: list[Path]):
    """加水印菜单"""
    from tools.add_watermark.batch import batch_add_image_watermark, batch_add_text_watermark

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

        maybe_show_file_list(media)

        if confirm_action(f"确认为 {len(media)} 个文件添加文字水印?"):
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

        maybe_show_file_list(media)

        if confirm_action(f"确认为 {len(media)} 个文件添加 Logo 水印?"):
            batch_add_image_watermark(
                files=media, watermark_path=logo_path,
                position=position, scale=scale, opacity=opacity,
            )
