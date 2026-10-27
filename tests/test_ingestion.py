import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.config import Settings
from app.ingestion import pipeline as ingestion
from app.ingestion.pipeline import corpus_fingerprint, load_manifest, validate_vector


def test_manifest_selects_only_corpus_files():
    path = Path("data/benchmark/manifest.json")
    documents = load_manifest(path)

    assert len(documents) == 24
    assert len({document.document_id for document in documents}) == 24
    assert all("BoCauHoi" not in str(document.path) for document in documents)
    assert all(document.path.is_file() for document in documents)


def test_corpus_fingerprint_changes_with_source_bytes(tmp_path):
    source = tmp_path / "01_CORPUS" / "source.txt"
    source.parent.mkdir()
    source.write_text("first", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"domains": [{"board_id": "b", "docs": [
        {"id": "d", "path": "01_CORPUS/source.txt"}
    ]}]}), encoding="utf-8")

    before = corpus_fingerprint(load_manifest(manifest))
    source.write_text("second", encoding="utf-8")
    after = corpus_fingerprint(load_manifest(manifest))

    assert before != after


def test_vector_validation_rejects_wrong_width_and_nan():
    settings = Settings(database_url="postgresql://localhost/test")

    with pytest.raises(ValueError, match="1024"):
        validate_vector([0.1], settings)
    with pytest.raises(ValueError, match="finite"):
        validate_vector([float("nan")] * 1024, settings)


def test_ingestion_rejects_changed_extracted_text_for_existing_document(tmp_path):
    connection = Mock()
    connection.execute.return_value.fetchall.return_value = [("document-1", "old text")]
    source = ingestion.SourceDocument("document-1", "board-1", tmp_path / "source.txt", "source.txt")
    prepared = ingestion.PreparedDocument(source, "same-file-hash", "new text", [], [])

    with pytest.raises(ValueError, match="extracted text changed"):
        ingestion._check_existing_extractions(connection, [prepared])
