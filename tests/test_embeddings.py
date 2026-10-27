import sys
from types import ModuleType

import pytest

from app.config import Settings
from app import embeddings


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        token_ids = [ord(character) for character in text]
        if add_special_tokens:
            return [1, *token_ids, 2]
        return token_ids


class FakeSentenceTransformer:
    created_models = []

    def __init__(self, model_name, revision, device):
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.max_seq_length = None
        self.encode_calls = []
        self.created_models.append(self)

    def encode(
        self,
        texts,
        batch_size,
        precision,
        normalize_embeddings,
        convert_to_numpy,
        show_progress_bar,
    ):
        self.encode_calls.append(
            {
                "texts": texts,
                "batch_size": batch_size,
                "precision": precision,
                "normalize_embeddings": normalize_embeddings,
                "convert_to_numpy": convert_to_numpy,
                "show_progress_bar": show_progress_bar,
            }
        )
        return [[0.25] * 1024 for _ in texts]


class FakeAutoTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, model_name, revision, use_fast):
        cls.calls.append(
            {"model_name": model_name, "revision": revision, "use_fast": use_fast}
        )
        return FakeTokenizer()


def install_fake_model_libraries(monkeypatch):
    sentence_transformers_module = ModuleType("sentence_transformers")
    sentence_transformers_module.SentenceTransformer = FakeSentenceTransformer
    transformers_module = ModuleType("transformers")
    transformers_module.AutoTokenizer = FakeAutoTokenizer

    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers_module)
    monkeypatch.setitem(sys.modules, "transformers", transformers_module)
    monkeypatch.setattr(embeddings, "_cached_model", None)
    monkeypatch.setattr(embeddings, "_cached_model_key", None)
    monkeypatch.setattr(embeddings, "_cached_tokenizer", None)
    monkeypatch.setattr(embeddings, "_cached_tokenizer_key", None)
    FakeSentenceTransformer.created_models.clear()
    FakeAutoTokenizer.calls.clear()


def make_settings(**overrides):
    values = {"database_url": "postgresql://postgres:postgres@localhost/nexora_rag"}
    values.update(overrides)
    return Settings(**values)


def test_local_embeddings_use_bge_settings_and_return_1024_values(monkeypatch):
    install_fake_model_libraries(monkeypatch)
    settings = make_settings()

    passage_vectors = embeddings.embed_passages(["Tài liệu tiếng Việt."], settings)
    query_vector = embeddings.embed_query("Câu hỏi tiếng Việt?", settings)

    assert len(passage_vectors) == 1
    assert len(passage_vectors[0]) == 1024
    assert len(query_vector) == 1024
    assert all(value == 0.25 for value in passage_vectors[0])

    model = FakeSentenceTransformer.created_models[0]
    assert model.model_name == "BAAI/bge-m3"
    assert model.revision == "31e47391fcbda65be526abe98e646b3c6cd845a8"
    assert model.device == "cpu"
    assert model.max_seq_length == 1200
    assert len(model.encode_calls) == 2
    assert model.encode_calls[0]["batch_size"] == 4
    assert model.encode_calls[0]["precision"] == "float32"
    assert model.encode_calls[0]["normalize_embeddings"] is True
    assert model.encode_calls[0] == {
        "texts": ["Tài liệu tiếng Việt."],
        "batch_size": 4,
        "precision": "float32",
        "normalize_embeddings": True,
        "convert_to_numpy": True,
        "show_progress_bar": False,
    }
    assert model.encode_calls[1]["texts"] == ["Câu hỏi tiếng Việt?"]
    assert FakeAutoTokenizer.calls == [
        {
            "model_name": "BAAI/bge-m3",
            "revision": "31e47391fcbda65be526abe98e646b3c6cd845a8",
            "use_fast": True,
        }
    ]


def test_embedding_rejects_text_that_would_be_truncated(monkeypatch):
    install_fake_model_libraries(monkeypatch)
    settings = make_settings(bge_max_input_tokens=5)

    with pytest.raises(ValueError, match="5 tokens"):
        embeddings.embed_query("four", settings)

    assert FakeSentenceTransformer.created_models == []


def test_model_cache_changes_when_input_limit_changes(monkeypatch):
    install_fake_model_libraries(monkeypatch)

    first_model = embeddings.get_embedder(make_settings())
    second_model = embeddings.get_embedder(make_settings(bge_max_input_tokens=800))

    assert first_model is not second_model
    assert first_model.max_seq_length == 1200
    assert second_model.max_seq_length == 800
