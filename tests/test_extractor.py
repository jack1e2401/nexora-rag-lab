import httpx
import pytest

from app.ingestion import extractor


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        return None


def test_tika_markdown_endpoint_returns_vietnamese_text(tmp_path, monkeypatch):
    document_path = tmp_path / "quy_dinh.txt"
    document_path.write_text("Bộ phận kinh doanh chịu trách nhiệm.", encoding="utf-8")
    request = {}

    def fake_put(url, content, headers, timeout):
        request["url"] = url
        request["content"] = content
        request["headers"] = headers
        request["timeout"] = timeout
        return FakeResponse("Bộ phận kinh doanh chịu trách nhiệm.")

    monkeypatch.setattr(extractor.httpx, "put", fake_put)

    extracted_text = extractor.extract_text(
        document_path,
        "http://localhost:9998/",
    )

    assert request["url"] == "http://localhost:9998/tika/md"
    assert request["content"] == document_path.read_bytes()
    assert request["headers"]["Accept"] == "text/markdown"
    assert extracted_text == "Bộ phận kinh doanh chịu trách nhiệm."


def test_empty_extraction_raises_file_specific_error(tmp_path, monkeypatch):
    document_path = tmp_path / "empty.pdf"
    document_path.write_bytes(b"not important to the fake server")

    def fake_put(url, content, headers, timeout):
        return FakeResponse(" \n\t ")

    monkeypatch.setattr(extractor.httpx, "put", fake_put)

    with pytest.raises(extractor.ExtractionError, match="empty.pdf"):
        extractor.extract_text(document_path, "http://localhost:9998")


def test_tika_connection_error_names_document(tmp_path, monkeypatch):
    document_path = tmp_path / "offline.pdf"
    document_path.write_bytes(b"document bytes")

    def fake_put(url, content, headers, timeout):
        raise httpx.ConnectError("Tika is offline")

    monkeypatch.setattr(extractor.httpx, "put", fake_put)

    with pytest.raises(extractor.ExtractionError, match="offline.pdf"):
        extractor.extract_text(document_path, "http://localhost:9998")


def test_normalize_text_preserves_paragraphs_and_unicode():
    raw_text = "Ca\u0301c mục\r\nquan trọng.  \r\n\r\nMục tiếp theo.  "

    normalized = extractor.normalize_text(raw_text)

    assert normalized == "Các mục\nquan trọng.\n\nMục tiếp theo."


def test_normalize_markdown_preserves_structure_and_list_indentation():
    raw_text = "# Heading  \r\n\r\n- parent\r\n  - child  \r\n\r\n| A | B |  \r\n"

    normalized = extractor.normalize_text(raw_text)

    assert normalized == "# Heading\n\n- parent\n  - child\n\n| A | B |"
