"""Helpers for turning source text ranges into checked chunks."""

from typing import Any

from app.ingestion.chunking.chunk_types import ChunkSlice, ChunkingSettings
from app.ingestion.chunking.utils.token_utils import count_special_tokens, count_tokens


def create_chunk(
    order: int,
    text: str,
    start_char: int,
    end_char: int,
    tokenizer: Any,
    settings: ChunkingSettings,
    fallback_reason: str | None = None,
) -> ChunkSlice | None:
    """Build one chunk and verify its token and model input limits."""
    chunk_text = text[start_char:end_char]

    if not chunk_text.strip():
        return None

    count = count_tokens(chunk_text, tokenizer)

    if count > settings.chunk_size:
        raise ValueError("Chunk is longer than the configured chunk_size")

    if count + count_special_tokens(tokenizer) > settings.max_input_tokens:
        raise ValueError("Chunk plus special tokens exceeds max_input_tokens")

    return ChunkSlice(
        order=order,
        text=chunk_text,
        token_count=count,
        start_char=start_char,
        end_char=end_char,
        fallback_reason=fallback_reason,
    )


def create_chunks_from_character_ranges(
    text: str,
    character_ranges: list[tuple[int, int]],
    tokenizer: Any,
    settings: ChunkingSettings,
    fallback_reason: str | None = None,
    order_start: int = 0,
) -> list[ChunkSlice]:
    """Create checked chunks from start/end character positions."""
    chunks = []

    for start_char, end_char in character_ranges:
        chunk = create_chunk(
            order=order_start + len(chunks),
            text=text,
            start_char=start_char,
            end_char=end_char,
            tokenizer=tokenizer,
            settings=settings,
            fallback_reason=fallback_reason,
        )

        if chunk is not None:
            chunks.append(chunk)

    return chunks
