"""
tools/pdf_split/splitter.py 单元测试
"""



def test_get_pdf_info_without_pypdf(monkeypatch, tmp_path):
    from tools.pdf_split import splitter

    monkeypatch.setattr(splitter, "_ensure_pypdf", lambda: False)
    fake_pdf = tmp_path / "a.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    result = splitter.get_pdf_info(fake_pdf)
    assert result == {"pages": 0, "title": "", "author": "", "size_mb": 0.0}


def test_split_functions_without_pypdf(monkeypatch, tmp_path):
    from tools.pdf_split import splitter

    monkeypatch.setattr(splitter, "_ensure_pypdf", lambda: False)
    fake_pdf = tmp_path / "a.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4")

    assert splitter.split_by_every_page(fake_pdf) == {"output": [], "pages": 0}
    assert splitter.split_by_range(fake_pdf, [(1, 2)]) == {"output": [], "pages": 0}
    assert splitter.split_by_chunk(fake_pdf, chunk_size=2) == {"output": [], "pages": 0}
    assert splitter.extract_pages(fake_pdf, [1, 2]) == {"output": "", "pages": 0}
