"""PDF 拆分菜单"""
import math
from pathlib import Path

from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

from config import INPUT_DIR, OUTPUT_PDF_SPLIT
from menus._prompts import confirm_action


def menu_pdf_split():
    """PDF 拆分菜单"""
    from tools.pdf_split.splitter import (
        _ensure_pypdf,
        extract_pages,
        get_pdf_info,
        split_by_chunk,
        split_by_every_page,
        split_by_range,
    )

    if not _ensure_pypdf():
        return

    # 选择 PDF 文件
    pdf_source = inquirer.select(
        message="选择 PDF 来源:",
        choices=[
            Choice("scan", "📂 扫描 input/ 目录"),
            Choice("path", "📄 指定 PDF 文件路径"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if pdf_source == "back":
        return

    pdf_files: list[Path] = []

    if pdf_source == "scan":
        pdf_files = sorted(
            f for f in INPUT_DIR.glob("*.pdf")
            if f.is_file()
        )
        if not pdf_files:
            print(f"❌ 未在 {INPUT_DIR.name}/ 目录中找到 PDF 文件")
            return
    else:
        path_str = inquirer.filepath(
            message="输入 PDF 文件路径:",
            validate=lambda x: Path(x).is_file() and Path(x).suffix.lower() == ".pdf",
        ).execute()
        pdf_files = [Path(path_str)]

    # 如果有多个 PDF, 选择一个
    if len(pdf_files) > 1:
        pdf_choices = [Choice(str(f), f.name) for f in pdf_files]
        selected = inquirer.select(
            message="选择要拆分的 PDF:",
            choices=pdf_choices,
        ).execute()
        pdf_path = Path(selected)
    else:
        pdf_path = pdf_files[0]

    # 显示 PDF 信息
    info = get_pdf_info(pdf_path)
    print(f"\n📄 文件: {pdf_path.name}")
    print(f"   页数: {info['pages']}")
    print(f"   大小: {info['size_mb']:.1f} MB")
    if info.get("title"):
        print(f"   标题: {info['title']}")
    print()

    if info["pages"] < 1:
        print("❌ PDF 文件无有效页面")
        return

    # 选择拆分模式
    mode = inquirer.select(
        message="选择拆分模式:",
        choices=[
            Choice("every", "📑 按页拆分 (每页生成一个 PDF)"),
            Choice("range", "📐 按页码范围拆分 (如 1-5, 6-10)"),
            Choice("chunk", "📦 按固定页数拆分 (每 N 页一份)"),
            Choice("extract", "🔍 提取指定页码 (合并为一个 PDF)"),
            Separator(),
            Choice("back", "⬅️  返回上一级"),
        ],
    ).execute()

    if mode == "back":
        return

    total_pages = info["pages"]

    if mode == "every":
        if confirm_action(f"确认将 {total_pages} 页拆分为 {total_pages} 个文件?"):
            result = split_by_every_page(pdf_path)
            print(f"\n✅ 拆分完成: {len(result['output'])} 个文件")
            print(f"📁 输出目录: {OUTPUT_PDF_SPLIT}")

    elif mode == "range":
        print(f"请输入页码范围 (总共 {total_pages} 页)")
        print("格式: 1-5,6-10,11-20  (逗号分隔多个范围)")
        range_str = inquirer.text(
            message="页码范围:",
            default=f"1-{min(total_pages, 10)}",
        ).execute()

        # 解析范围
        ranges = []
        try:
            for part in range_str.split(","):
                part = part.strip()
                if "-" in part:
                    s, e = part.split("-", 1)
                    ranges.append((int(s.strip()), int(e.strip())))
                else:
                    # 单页: "5" -> (5, 5)
                    p = int(part)
                    ranges.append((p, p))
        except ValueError:
            print("❌ 格式错误，请使用 1-5,6-10 格式")
            return

        if not ranges:
            print("❌ 未指定有效范围")
            return

        range_desc = ", ".join(f"{s}-{e}" for s, e in ranges)
        if confirm_action(f"确认拆分范围 [{range_desc}]?"):
            result = split_by_range(pdf_path, ranges)
            print(f"\n✅ 拆分完成: {len(result['output'])} 个文件")
            print(f"📁 输出目录: {OUTPUT_PDF_SPLIT}")

    elif mode == "chunk":
        chunk_size = int(inquirer.number(
            message=f"每份页数 (总共 {total_pages} 页):",
            default=10,
            min_allowed=1,
            max_allowed=total_pages,
        ).execute())

        num_files = math.ceil(total_pages / chunk_size)
        if confirm_action(f"确认将 {total_pages} 页按每 {chunk_size} 页拆分为 {num_files} 份?"):
            result = split_by_chunk(pdf_path, chunk_size)
            print(f"\n✅ 拆分完成: {len(result['output'])} 个文件")
            print(f"📁 输出目录: {OUTPUT_PDF_SPLIT}")

    elif mode == "extract":
        print(f"请输入要提取的页码 (总共 {total_pages} 页)")
        print("格式: 1,3,5,7  或  1-5,8,10-12  (支持范围和单页混合)")
        pages_str = inquirer.text(
            message="页码:",
            default="1",
        ).execute()

        # 解析页码 (支持 1,3,5 和 1-5 混合格式)
        pages = []
        try:
            for part in pages_str.split(","):
                part = part.strip()
                if "-" in part:
                    s, e = part.split("-", 1)
                    pages.extend(range(int(s.strip()), int(e.strip()) + 1))
                else:
                    pages.append(int(part))
        except ValueError:
            print("❌ 格式错误，请使用 1,3,5 或 1-5 格式")
            return

        if not pages:
            print("❌ 未指定有效页码")
            return

        # 去重排序
        pages = sorted(set(pages))
        pages_preview = ", ".join(str(p) for p in pages[:10])
        if len(pages) > 10:
            pages_preview += f" ... (共 {len(pages)} 页)"

        if confirm_action(f"确认提取页码 [{pages_preview}]?"):
            result = extract_pages(pdf_path, pages)
            if result["output"]:
                print(f"\n✅ 提取完成: {result['pages']} 页")
                print(f"📁 输出: {result['output']}")
