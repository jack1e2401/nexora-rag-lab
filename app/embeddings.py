"""Load the pinned local BGE-M3 model and create dense embeddings."""

from __future__ import annotations

import math
from threading import Lock
from typing import TYPE_CHECKING, Any

from app.config import Settings

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
    from transformers import PreTrainedTokenizerBase


_cached_model: Any = None
_cached_model_key: tuple[str, str, str, int] | None = None
_cached_tokenizer: Any = None
_cached_tokenizer_key: tuple[str, str] | None = None
_model_lock = Lock()
_tokenizer_lock = Lock()


def get_embedder(settings: Settings) -> SentenceTransformer:
    """Cache one model per revision, device, and input limit in this process."""
    global _cached_model
    global _cached_model_key

    model_key = (
        settings.bge_model_name,
        settings.bge_model_revision,
        settings.bge_device,
        settings.bge_max_input_tokens,
    )

    with _model_lock:
        if _cached_model is not None and _cached_model_key == model_key:
            return _cached_model

        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            settings.bge_model_name,
            revision=settings.bge_model_revision,
            device=settings.bge_device,
        )
        model.max_seq_length = settings.bge_max_input_tokens

        _cached_model = model
        _cached_model_key = model_key
        return model


def get_tokenizer(settings: Settings) -> PreTrainedTokenizerBase:
    """Load the fast tokenizer matching the pinned BGE-M3 revision."""
    global _cached_tokenizer
    global _cached_tokenizer_key

    tokenizer_key = (settings.bge_model_name, settings.bge_model_revision)

    with _tokenizer_lock:
        if _cached_tokenizer is not None and _cached_tokenizer_key == tokenizer_key:
            return _cached_tokenizer

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            settings.bge_model_name,
            revision=settings.bge_model_revision,
            use_fast=True,
        )

        _cached_tokenizer = tokenizer
        _cached_tokenizer_key = tokenizer_key
        return tokenizer


def _check_input_length(text: str, settings: Settings) -> None:
    tokenizer = get_tokenizer(settings)
    token_ids = tokenizer.encode(text, add_special_tokens=True)

    if len(token_ids) > settings.bge_max_input_tokens:
        raise ValueError(
            "Text has "
            f"{len(token_ids)} tokens including special tokens; "
            f"the model input limit is {settings.bge_max_input_tokens} tokens"
        )


def _convert_vectors(vectors: Any, settings: Settings) -> list[list[float]]:
    converted_vectors = []

    for vector in vectors:
        values = [float(value) for value in vector]

        if len(values) != settings.bge_dimensions:
            raise ValueError(
                f"Expected {settings.bge_dimensions} embedding values, got {len(values)}"
            )

        if not all(math.isfinite(value) for value in values):
            raise ValueError("Embedding contains a non-finite value")

        converted_vectors.append(values)

    return converted_vectors


def embed_passages(texts: list[str], settings: Settings) -> list[list[float]]:
    """Create normalized float32 vectors for document passages."""
    if not texts:
        return []

    for text in texts:
        _check_input_length(text, settings)

    model = get_embedder(settings)
    vectors = model.encode(
        texts,
        batch_size=settings.bge_batch_size,
        precision=settings.bge_precision,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    converted_vectors = _convert_vectors(vectors, settings)

    if len(converted_vectors) != len(texts):
        raise ValueError("The embedding model returned a different number of vectors")

    return converted_vectors


def embed_query(text: str, settings: Settings) -> list[float]:
    """Create one query vector with the same settings used for passages."""
    return embed_passages([text], settings)[0]
