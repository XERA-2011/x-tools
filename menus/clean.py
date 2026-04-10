"""清理文件菜单"""
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import INPUT_DIR, OUTPUT_DIR


def menu_clean():
    """清理 input 和 output 目录菜单"""
    target = inquirer.select(
        message="选择清理范围:",
        choices=[
            Choice("all", "🗑️  清理全部 (input + output)"),
            Choice("input", "📥 仅清理 input"),
            Choice("output", "📤 仅清理 output"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if target == "back":
        return

    dirs_to_clean = []
    if target in ("all", "input"):
        dirs_to_clean.append(INPUT_DIR)
    if target in ("all", "output"):
        dirs_to_clean.append(OUTPUT_DIR)

    # 统计文件数量
    total_files = 0
    total_size = 0
    for d in dirs_to_clean:
        if d.exists():
            for f in d.rglob("*"):
                if f.is_file():
                    total_files += 1
                    total_size += f.stat().st_size

    if total_files == 0:
        print("✅ 目录已经是空的，无需清理")
        return

    size_mb = total_size / (1024 * 1024)
    print(f"\n⚠️  即将删除 {total_files} 个文件 (共 {size_mb:.1f} MB)")
    for d in dirs_to_clean:
        print(f"   📂 {d}")

    if not inquirer.confirm(message="确认删除? 此操作不可恢复!", default=False).execute():
        print("✅ 已取消")
        return

    deleted = 0
    for d in dirs_to_clean:
        if not d.exists():
            continue
        for item in sorted(d.rglob("*"), reverse=True):
            try:
                if item.is_file():
                    item.unlink()
                    deleted += 1
                elif item.is_dir() and item != d:
                    # 删除空子目录 (保留根目录本身)
                    try:
                        item.rmdir()
                    except OSError:
                        pass  # 目录非空，跳过
            except Exception as e:
                print(f"  ⚠️  无法删除 {item.name}: {e}")

    print(f"\n✅ 清理完成，共删除 {deleted} 个文件")
