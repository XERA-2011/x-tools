"""
TUI 公共 prompt 函数

提供输入源选择、目录选择、文件列表展示、确认等通用交互组件。
"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import IMAGE_EXTENSIONS, INPUT_DIR
from tools.common import scan_media, scan_videos


def prompt_input_mode() -> str:
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


def prompt_directory() -> tuple[Path, bool]:
    """统一目录输入"""
    path_str = inquirer.filepath(
        message="输入目录路径:",
        default="",
        validate=lambda x: Path(x).is_dir(),
        only_directories=True,
    ).execute()
    return Path(path_str), False


def maybe_show_file_list(files: list[Path], message: str | None = None) -> None:
    """可选展示待处理文件列表"""
    prompt = message or f"是否查看将要处理的 {len(files)} 个文件列表?"
    if inquirer.confirm(message=prompt, default=False).execute():
        print("\n文件列表:")
        for f in files:
            print(f"  - {f.name}")
        print()


def confirm_action(message: str, default: bool = True) -> bool:
    """统一确认提示"""
    return inquirer.confirm(message=message, default=default).execute()


def get_input_videos() -> list[Path] | None:
    """获取待处理视频列表, 返回 None 表示用户选择返回"""
    mode = prompt_input_mode()

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
        directory, recursive = prompt_directory()
        videos = scan_videos(directory, recursive=recursive)
        print(f"扫描到 {len(videos)} 个视频")
        return videos

    return []


def get_input_media() -> list[Path] | None:
    """获取待处理媒体文件列表 (视频 + 图片), 返回 None 表示用户选择返回"""
    mode = prompt_input_mode()

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
        directory, recursive = prompt_directory()
        videos, images = scan_media(directory, recursive=recursive)
        files = images + videos
        print(f"扫描到 {len(images)} 个图片, {len(videos)} 个视频")
        return files

    return []
