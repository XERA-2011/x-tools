"""
x-tools 交互式终端入口 (TUI)

功能:
  - 引导用户配置参数
  - 扫描 input/ 目录或选择单文件
  - 调用 Rich 显示进度
"""
import shutil
import sys

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import FFMPEG_BIN, ensure_dirs
from menus._prompts import get_input_media, get_input_videos


def _check_ffmpeg():
    """检测 FFmpeg 是否可用"""
    if not shutil.which(FFMPEG_BIN):
        print("❌ 未检测到 FFmpeg, 请先安装:")
        print(f"   当前配置 FFMPEG_BIN = {FFMPEG_BIN!r}")
        print("   macOS:   brew install ffmpeg")
        print("   Ubuntu:  sudo apt install ffmpeg")
        print("   Windows: https://ffmpeg.org/download.html")
        sys.exit(1)


def main():
    _check_ffmpeg()
    ensure_dirs()

    print("\n🧰 X-TOOLS 视频处理工具箱 v0.2.0\n")

    while True:
        module = inquirer.select(
            message="选择功能模块:",
            choices=[
                Choice("watermark", "💧 去水印 (Watermark)"),
                Choice("add_watermark", "🏷️  增加水印 (Add Watermark)"),
                Choice("compress", "🗜️  无损压缩 (Compress)"),
                Choice("upscale", "🆙 高清重置 (Upscale)"),
                Choice("interpolate", "⏯️  帧数补充 (Interpolate)"),
                Choice("convert", "🔄 格式转换 (Convert)"),
                Choice("filter", "🎨 滤镜效果 (Filter)"),
                Choice("crop", "✂️  裁切比例 (Crop)"),
                Choice("concat", "🎬 拼接视频 (Concat)"),
                Choice("slideshow", "📸 幻灯片 (Slideshow)"),
                Choice("bgm", "🎵 添加背景音乐 (BGM)"),
                Choice("subtitle", "📝 字幕 (Subtitle)"),
                Choice("mv", "🎵 歌词 MV 生成 (Lyric MV)"),
                Choice("tts", "🔊 TTS 音频生成 (Text-to-Speech)"),
                Choice("pdf_split", "📄 PDF 拆分 (PDF Split)"),
                Separator(),
                Choice("clean", "🧹 清理文件 (Clean)"),
                Choice("selftest", "🩺 自检测试 (Self Test)"),
                Choice("exit", "❌ 退出"),
            ],
            default="watermark",
        ).execute()

        if module == "exit":
            print("Bye!")
            sys.exit(0)

        # 无需输入文件的模块
        if module == "clean":
            from menus.clean import menu_clean
            menu_clean()
        elif module == "selftest":
            from menus.selftest import menu_selftest
            menu_selftest()
        elif module == "mv":
            from menus.mv import menu_mv
            menu_mv()
        elif module == "pdf_split":
            from menus.pdf_split import menu_pdf_split
            menu_pdf_split()
        elif module == "slideshow":
            from menus.slideshow import menu_slideshow
            menu_slideshow()
        elif module == "tts":
            from menus.tts import menu_tts
            menu_tts()

        # 需要媒体文件 (视频 + 图片) 的模块
        elif module in ("add_watermark", "convert", "filter", "crop", "subtitle"):
            media = get_input_media()
            if media is None:
                continue
            if not media:
                print("❌ 未找到媒体文件")
                continue

            if module == "add_watermark":
                from menus.add_watermark import menu_add_watermark
                menu_add_watermark(media)
            elif module == "convert":
                from menus.convert import menu_convert
                menu_convert(media)
            elif module == "filter":
                from menus.media_ops import menu_filter
                menu_filter(media)
            elif module == "crop":
                from menus.media_ops import menu_crop
                menu_crop(media)
            elif module == "subtitle":
                from menus.subtitle import menu_subtitle
                menu_subtitle(media)

        # 需要视频文件的模块
        else:
            videos = get_input_videos()
            if videos is None:
                continue  # 用户选择返回上一级
            if not videos:
                print("❌ 未找到视频文件")
                continue

            if module == "watermark":
                from menus.watermark import menu_watermark
                menu_watermark(videos)
            elif module == "upscale":
                from menus.upscale import menu_upscale
                menu_upscale(videos)
            elif module == "interpolate":
                from menus.interpolate import menu_interpolate
                menu_interpolate(videos)
            elif module == "concat":
                from menus.audio_video import menu_concat
                menu_concat(videos)
            elif module == "bgm":
                from menus.audio_video import menu_add_bgm
                menu_add_bgm(videos)
            elif module == "compress":
                from menus.media_ops import menu_compress
                menu_compress(videos)

        print()
        if not inquirer.confirm(message="继续其他操作?", default=True).execute():
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ 用户取消操作")
        sys.exit(0)
