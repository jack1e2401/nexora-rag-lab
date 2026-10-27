import pytest

from app.config import Settings
from app.rag.generation import DeepSeekError, request_deepseek_json, validate_generation
from app.rag.pipeline import ask
from app.rag.retrieval import RetrievedChunk, validate_profile


SETTINGS = Settings(database_url="postgresql://localhost/test")


def sample_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="c1", document_id="M-PLAN-02", board_id="board-mercury",
        text="Minh is responsible for payments.", score=0.9, rank=1,
        start_char=10, end_char=43,
    )


def test_profile_rejects_different_embedding_revision():
    with pytest.raises(ValueError, match="revision"):
        validate_profile({"embedding_model": "BAAI/bge-m3", "embedding_revision": "other",
                          "embedding_dimensions": 1024}, SETTINGS)


def test_profile_rejects_different_dimensions():
    with pytest.raises(ValueError, match="dimensions"):
        validate_profile({"embedding_model": SETTINGS.bge_model_name,
                          "embedding_revision": SETTINGS.bge_model_revision,
                          "embedding_dimensions": 768}, SETTINGS)


def test_unknown_citation_label_causes_abstention():
    answer = validate_generation(
        {"answer": "Minh", "source_labels": ["S99"], "abstained": False},
        [sample_chunk()],
    )
    assert answer.abstained is True
    assert answer.citations == []


def test_non_boolean_abstention_flag_causes_abstention():
    answer = validate_generation(
        {"answer": "Minh", "source_labels": ["S1"], "abstained": "false"},
        [sample_chunk()],
    )
    assert answer.abstained is True


def test_valid_citation_uses_stored_source_details():
    answer = validate_generation(
        {"answer": "Minh", "source_labels": ["S1"], "abstained": False},
        [sample_chunk()],
    )
    assert answer.abstained is False
    assert answer.citations[0].document_id == "M-PLAN-02"
    assert answer.citations[0].text == "Minh is responsible for payments."


def test_ask_with_no_hits_does_not_call_deepseek(monkeypatch):
    import app.rag.pipeline as pipeline

    monkeypatch.setattr(pipeline, "search", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "generate_answer", lambda *args, **kwargs: pytest.fail("DeepSeek called"))

    result = ask("Who?", "board-mercury", "baseline", 3, SETTINGS)
    assert result.abstained is True
    assert result.citations == []


def test_search_filters_by_profile_and_board(monkeypatch):
    import app.rag.retrieval as retrieval

    class FakeConnection:
        def __init__(self):
            self.statements = []

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, statement, params):
            self.statements.append((statement, params))
            self.result = self
            return self

        def fetchone(self):
            return {"embedding_model": SETTINGS.bge_model_name,
                    "embedding_revision": SETTINGS.bge_model_revision,
                    "embedding_dimensions": SETTINGS.bge_dimensions}

        def fetchall(self):
            return [{"chunk_id": "c1", "document_id": "M-1", "board_id": "board-a",
                     "text": "source", "start_char": 0, "end_char": 6, "score": 0.8}]

    connection = FakeConnection()
    monkeypatch.setattr(retrieval.psycopg, "connect", lambda *args, **kwargs: connection)
    monkeypatch.setattr(retrieval, "register_vector", lambda *args: None)

    results = retrieval.search("question", "board-a", "profile-a", 3, SETTINGS, [0.1] * 1024)

    query, params = connection.statements[1]
    assert "c.profile_id = %s AND d.board_id = %s" in query
    assert params[1:3] == ("profile-a", "board-a")
    assert results[0].document_id == "M-1"
    assert results[0].rank == 1


def test_deepseek_rejects_truncated_response(monkeypatch):
    import app.rag.generation as generation

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]}

    monkeypatch.setattr(generation.httpx, "post", lambda *args, **kwargs: FakeResponse())
    settings = Settings(database_url="postgresql://localhost/test", deepseek_api_key="test-key")

    with pytest.raises(DeepSeekError, match="incomplete"):
        request_deepseek_json([], "deepseek-flash", settings)
