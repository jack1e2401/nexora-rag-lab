from io import BytesIO
from types import SimpleNamespace

from fastapi import UploadFile

from app import embeddings, lab_api
from app.config import Settings
from app.ingestion import extractor
from app.main import app


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(character) for character in text]

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": self.encode(text, add_special_tokens)}

        if return_offsets_mapping:
            result["offset_mapping"] = [
                (index, index + 1) for index in range(len(text))
            ]

        return result

    def convert_ids_to_tokens(self, token_ids):
        return [chr(token_id) for token_id in token_ids]

    def num_special_tokens_to_add(self, pair=False):
        return 2


def make_settings():
    return Settings(
        database_url="postgresql://postgres:postgres@localhost/nexora_rag"
    )


def get_endpoint(path):
    endpoints = {
        "/ingestion/extract": lab_api.extract_document,
        "/tokenization/inspect": lab_api.inspect_tokens,
        "/chunking/split": lab_api.split_markdown,
        "/embeddings/embed": lab_api.create_embeddings,
    }
    return endpoints[path]


def test_extract_endpoint_returns_markdown_without_saving_file(monkeypatch):
    monkeypatch.setattr(
        extractor,
        "extract_bytes",
        lambda filename, content, tika_url: "# A Markdown heading",
        raising=False,
    )
    endpoint = get_endpoint("/ingestion/extract")
    upload = UploadFile(
        file=BytesIO(b"document bytes"),
        filename="guide.md",
    )
    response = endpoint(upload, make_settings())

    assert response == {
        "filename": "guide.md",
        "character_count": len("# A Markdown heading"),
        "markdown": "# A Markdown heading",
    }


def test_token_inspection_endpoint_returns_real_tokenizer_output(monkeypatch):
    monkeypatch.setattr(embeddings, "get_tokenizer", lambda settings: FakeTokenizer())
    endpoint = get_endpoint("/tokenization/inspect")
    response = endpoint(SimpleNamespace(text="Hi!"), make_settings())

    assert response == {
        "token_count": 3,
        "tokens": [
            {
                "index": 0,
                "token_id": ord("H"),
                "token": "H",
                "start_char": 0,
                "end_char": 1,
                "source_text": "H",
            },
            {
                "index": 1,
                "token_id": ord("i"),
                "token": "i",
                "start_char": 1,
                "end_char": 2,
                "source_text": "i",
            },
            {
                "index": 2,
                "token_id": ord("!"),
                "token": "!",
                "start_char": 2,
                "end_char": 3,
                "source_text": "!",
            },
        ],
    }


def test_chunk_endpoint_returns_chunk_text_and_character_positions(monkeypatch):
    monkeypatch.setattr(embeddings, "get_tokenizer", lambda settings: FakeTokenizer())
    endpoint = get_endpoint("/chunking/split")
    request = SimpleNamespace(
        markdown="abcdefghijklmno",
        strategy="fixed_token",
        chunk_size=8,
        overlap=2,
    )
    response = endpoint(request, make_settings())

    assert response["chunks"] == [
        {
            "order": 0,
            "text": "abcdefgh",
            "token_count": 8,
            "start_char": 0,
            "end_char": 8,
            "fallback_reason": None,
        },
        {
            "order": 1,
            "text": "ghijklmn",
            "token_count": 8,
            "start_char": 6,
            "end_char": 14,
            "fallback_reason": None,
        },
        {
            "order": 2,
            "text": "mno",
            "token_count": 3,
            "start_char": 12,
            "end_char": 15,
            "fallback_reason": None,
        },
    ]


def test_embedding_endpoint_returns_vectors_and_dimension(monkeypatch):
    settings = make_settings()
    monkeypatch.setattr(
        embeddings,
        "embed_passages",
        lambda texts, current_settings: [
            [0.25] * current_settings.bge_dimensions for _ in texts
        ],
    )
    endpoint = get_endpoint("/embeddings/embed")
    request = SimpleNamespace(texts=["first chunk", "second chunk"])
    response = endpoint(request, settings)

    assert response["dimensions"] == 1024
    assert response["count"] == 2
    assert len(response["vectors"]) == 2
    assert len(response["vectors"][0]) == 1024


def test_swagger_lists_each_local_rag_step():
    paths = app.openapi()["paths"]

    assert "/ingestion/extract" in paths
    assert "/tokenization/inspect" in paths
    assert "/chunking/split" in paths
    assert "/embeddings/embed" in paths
