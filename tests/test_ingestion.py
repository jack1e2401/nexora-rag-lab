import json
from pathlib import Path

import pytest

from app.config import Settings
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
