"""TTS 音频生成菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice

from config import INPUT_DIR, OUTPUT_TTS


def menu_tts():
    """TTS 音频生成交互菜单"""
    from tools.tts.tts_generate import TTS_VOICES, batch_generate, generate_audio

    mode = inquirer.select(
        message="TTS 模式:",
        choices=[
            Choice("batch", "📂 批量生成 (扫描 input/ 下的 .txt 文件)"),
            Choice("single", "📝 单条生成 (手动输入文本)"),
        ],
        default="batch",
    ).execute()

    # 选择音色
    voice_choices = [
        Choice(key, info["name"]) for key, info in TTS_VOICES.items()
    ]
    voice_key = inquirer.select(
        message="选择音色:",
        choices=voice_choices,
        default="jenny",
    ).execute()

    if mode == "single":
        # 单条生成
        text = inquirer.text(
            message="输入要朗读的文本:",
            validate=lambda x: len(x.strip()) > 0,
        ).execute()

        filename = inquirer.text(
            message="输出文件名 (不含后缀, 留空自动生成):",
            default="",
        ).execute().strip() or None

        try:
            result = generate_audio(
                text=text,
                voice_key=voice_key,
                filename=filename,
            )
            print(f"\n✅ 音频已生成: {result['output']}")
            print(f"   大小: {result['size_mb']:.2f} MB")
        except Exception as e:
            print(f"\n❌ 生成失败: {e}")

    else:
        # 批量生成
        txt_files = sorted(
            f for f in INPUT_DIR.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        )

        if not txt_files:
            print(f"\n❌ 未在 {INPUT_DIR} 下找到 .txt 文件")
            print("   请将文本文件放入 input/ 目录后重试。")
            return

        print(f"\n📄 找到 {len(txt_files)} 个文本文件:")
        for f in txt_files:
            preview = f.read_text(encoding="utf-8").strip()[:60]
            print(f"   - {f.name}: {preview}...")

        confirm = inquirer.confirm(
            message=f"开始批量生成 {len(txt_files)} 个音频?",
            default=True,
        ).execute()

        if not confirm:
            print("已取消。")
            return

        results = batch_generate(
            voice_key=voice_key,
            files=txt_files,
        )

        # 打印结果
        print(f"\n{'=' * 50}")
        print("📊 生成结果")
        print("=" * 50)
        for r in results:
            name = Path(r["file"]).name
            if r["status"] == "success":
                print(f"  ✅ {name} → {Path(r['output']).name} ({r.get('size_mb', 0):.2f} MB)")
            else:
                print(f"  ❌ {name} — {r.get('error', '未知错误')}")
        print("=" * 50)
        print(f"输出目录: {OUTPUT_TTS}")
