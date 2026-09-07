"""
PDF 文案提取模块

功能:
  - 解析 PDF 每页正文提取朗读文本 (自动清洗末尾页码等噪音)
  - 智能关联同名 .txt 或 .json 自定义口播稿 (可选)
  - 支持将文案保存为 JSON/TXT，或加载自定义编辑后的文案
"""
import json
import re
from pathlib import Path

from tools.common import logger


def clean_script_text(text: str) -> str:
    """清理文案中不适合朗读的噪音 (例如末尾的 14 / 62 等页码标识)"""
    if not text:
        return ""
    cleaned = re.sub(r"\n?\b\d+\s*/\s*\d+\b", "", text).strip()
    return cleaned


def find_companion_script(pdf_path: Path) -> Path | None:
    """查找同名自定义文案文件 (.json 或 .txt)"""
    candidate_json = pdf_path.with_suffix(".json")
    if candidate_json.is_file():
        return candidate_json

    candidate_txt = pdf_path.with_suffix(".txt")
    if candidate_txt.is_file():
        return candidate_txt

    return None


def extract_script_from_pdf(pdf_path: Path | str) -> list[dict]:
    """
    从 PDF 提取每页文本
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("未安装 pypdf，请在虚拟环境中执行: pip install pypdf")

    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise FileNotFoundError(f"未找到 PDF 文件: {pdf_path}")

    # 1. 优先检查是否有同名自定义脚本文件
    companion = find_companion_script(pdf_path)
    if companion:
        if companion.suffix.lower() == ".json":
            try:
                script = load_script(companion)
                logger.info(f"发现并加载同名自定义脚本文件: {companion.name} (共 {len(script)} 页)")
                return script
            except Exception as e:
                logger.warning(f"读取同名脚本 {companion.name} 失败: {e}，回退为从 PDF 提取正文")

    # 2. 从 PDF 提取页面文字
    reader = PdfReader(str(pdf_path))
    slides_script = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        t = clean_script_text((page.extract_text() or "").strip())
        slides_script.append({
            "page": page_num,
            "text": t,
            "source": "pdf_text" if t else "empty",
        })

    logger.info(f"成功从 PDF 提取 {len(slides_script)} 页文本 (文件: {pdf_path.name})")
    return slides_script


def save_script(script: list[dict], output_path: Path | str) -> Path:
    """保存文案为 JSON 文件"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def load_script(script_path: Path | str) -> list[dict]:
    """加载文案 JSON 文件"""
    path = Path(script_path)
    if not path.is_file():
        raise FileNotFoundError(f"未找到文案文件: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("文案文件格式不正确，需为页面列表 JSON")
    return data
