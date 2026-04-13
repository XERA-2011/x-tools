"""
PDF 拆分工具

支持多种拆分模式:
  - 按页拆分 (每页一个文件)
  - 按页码范围拆分
  - 按固定页数拆分 (每 N 页一个文件)
  - 提取指定页
"""
import logging
from pathlib import Path

from config import OUTPUT_PDF_SPLIT

logger = logging.getLogger("x-tools")


def _ensure_pypdf():
    """检查 pypdf 是否可用"""
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        logger.error("pypdf 未安装，请运行: pip install pypdf")
        print("❌ 需要安装 pypdf: pip install pypdf")
        return False


def get_pdf_info(pdf_path: Path) -> dict:
    """
    获取 PDF 基本信息
    
    Returns:
        {"pages": int, "title": str, "size_mb": float}
    """
    if not _ensure_pypdf():
        return {"pages": 0, "title": "", "author": "", "size_mb": 0.0}

    import pypdf
    
    reader = pypdf.PdfReader(str(pdf_path))
    metadata = reader.metadata or {}
    
    return {
        "pages": len(reader.pages),
        "title": metadata.get("/Title", "") or "",
        "author": metadata.get("/Author", "") or "",
        "size_mb": pdf_path.stat().st_size / (1024 * 1024),
    }


def split_by_every_page(pdf_path: Path, output_dir: Path | None = None) -> dict:
    """
    按页拆分: 每页生成一个独立 PDF
    
    Args:
        pdf_path: 源 PDF 文件
        output_dir: 输出目录 (默认 output/pdf_split/)
    
    Returns:
        {"output": list[str], "pages": int}
    """
    if not _ensure_pypdf():
        return {"output": [], "pages": 0}

    import pypdf

    out_dir = output_dir or OUTPUT_PDF_SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)
    
    reader = pypdf.PdfReader(str(pdf_path))
    total = len(reader.pages)
    stem = pdf_path.stem
    outputs = []
    
    for i in range(total):
        writer = pypdf.PdfWriter()
        writer.add_page(reader.pages[i])
        
        out_name = f"{stem}_p{i + 1:04d}.pdf"
        out_path = out_dir / out_name
        
        with open(out_path, "wb") as f:
            writer.write(f)
        
        outputs.append(str(out_path))
        logger.info(f"✅ 第 {i + 1}/{total} 页 → {out_name}")
    
    logger.info(f"拆分完成: {total} 页 → {len(outputs)} 个文件")
    return {"output": outputs, "pages": total}


def split_by_range(
    pdf_path: Path,
    ranges: list[tuple[int, int]],
    output_dir: Path | None = None,
) -> dict:
    """
    按页码范围拆分: 每个范围生成一个 PDF
    
    Args:
        pdf_path: 源 PDF 文件
        ranges: 页码范围列表 [(start, end), ...], 页码从 1 开始 (含首尾)
        output_dir: 输出目录
    
    Returns:
        {"output": list[str], "pages": int}
    """
    if not _ensure_pypdf():
        return {"output": [], "pages": 0}

    import pypdf

    out_dir = output_dir or OUTPUT_PDF_SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)
    
    reader = pypdf.PdfReader(str(pdf_path))
    total = len(reader.pages)
    stem = pdf_path.stem
    outputs = []
    
    for idx, (start, end) in enumerate(ranges, 1):
        # 限制范围
        start = max(1, start)
        end = min(total, end)
        
        if start > end:
            logger.warning(f"⚠️ 跳过无效范围: {start}-{end}")
            continue
        
        writer = pypdf.PdfWriter()
        for page_num in range(start - 1, end):  # 转为 0-indexed
            writer.add_page(reader.pages[page_num])
        
        out_name = f"{stem}_p{start}-{end}.pdf"
        out_path = out_dir / out_name
        
        with open(out_path, "wb") as f:
            writer.write(f)
        
        outputs.append(str(out_path))
        page_count = end - start + 1
        logger.info(f"✅ 范围 {start}-{end} ({page_count} 页) → {out_name}")
    
    logger.info(f"拆分完成: {len(outputs)} 个文件")
    return {"output": outputs, "pages": total}


def split_by_chunk(
    pdf_path: Path,
    chunk_size: int = 10,
    output_dir: Path | None = None,
) -> dict:
    """
    按固定页数拆分: 每 N 页生成一个 PDF
    
    Args:
        pdf_path: 源 PDF 文件
        chunk_size: 每份包含的页数
        output_dir: 输出目录
    
    Returns:
        {"output": list[str], "pages": int}
    """
    if not _ensure_pypdf():
        return {"output": [], "pages": 0}

    import pypdf

    out_dir = output_dir or OUTPUT_PDF_SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)
    
    reader = pypdf.PdfReader(str(pdf_path))
    total = len(reader.pages)
    stem = pdf_path.stem
    outputs = []
    
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        
        writer = pypdf.PdfWriter()
        for page_num in range(start, end):
            writer.add_page(reader.pages[page_num])
        
        # 页码从 1 开始显示
        out_name = f"{stem}_p{start + 1}-{end}.pdf"
        out_path = out_dir / out_name
        
        with open(out_path, "wb") as f:
            writer.write(f)
        
        outputs.append(str(out_path))
        logger.info(f"✅ 第 {start + 1}-{end} 页 → {out_name}")
    
    num_files = len(outputs)
    logger.info(f"拆分完成: {total} 页 → {num_files} 个文件 (每份最多 {chunk_size} 页)")
    return {"output": outputs, "pages": total}


def extract_pages(
    pdf_path: Path,
    pages: list[int],
    output_dir: Path | None = None,
) -> dict:
    """
    提取指定页码 (合并为一个 PDF)
    
    Args:
        pdf_path: 源 PDF 文件
        pages: 要提取的页码列表 (从 1 开始)
        output_dir: 输出目录
    
    Returns:
        {"output": str, "pages": int}
    """
    if not _ensure_pypdf():
        return {"output": "", "pages": 0}

    import pypdf

    out_dir = output_dir or OUTPUT_PDF_SPLIT
    out_dir.mkdir(parents=True, exist_ok=True)
    
    reader = pypdf.PdfReader(str(pdf_path))
    total = len(reader.pages)
    stem = pdf_path.stem
    
    writer = pypdf.PdfWriter()
    valid_pages = []
    
    for p in pages:
        if 1 <= p <= total:
            writer.add_page(reader.pages[p - 1])
            valid_pages.append(p)
        else:
            logger.warning(f"⚠️ 页码 {p} 超出范围 (总共 {total} 页), 已跳过")
    
    if not valid_pages:
        logger.error("❌ 没有有效页码可提取")
        return {"output": "", "pages": 0}
    
    pages_str = ",".join(str(p) for p in valid_pages[:5])
    if len(valid_pages) > 5:
        pages_str += f"...({len(valid_pages)}页)"
    
    out_name = f"{stem}_extract_{pages_str}.pdf"
    out_path = out_dir / out_name
    
    with open(out_path, "wb") as f:
        writer.write(f)
    
    logger.info(f"✅ 提取 {len(valid_pages)} 页 → {out_name}")
    return {"output": str(out_path), "pages": len(valid_pages)}
