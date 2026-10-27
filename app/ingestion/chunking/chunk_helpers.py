"""Shared token and source-span helpers used by chunking strategies."""

from typing import Any

from app.ingestion.chunking.chunk_types import ChunkSlice, ChunkingSettings


def token_offsets(text: str, tokenizer: Any) -> list[tuple[int, int]]:
    """Return source offsets for every non-special BGE token."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    valid_offsets = []

    for start, end in offsets:
        if start < 0 or end > len(text) or end < start:
            raise ValueError("Tokenizer returned invalid source offsets")

        if end > start:
            valid_offsets.append((start, end))

    if text.strip() and len(valid_offsets) != len(token_ids):
        raise ValueError("Tokenizer did not return source offsets for every token")

    return valid_offsets


def token_count(text: str, tokenizer: Any) -> int:
    """Count content tokens without adding model special tokens."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def special_token_count(tokenizer: Any) -> int:
    """Return how many special tokens the model adds to one text input."""
    count_special_tokens = getattr(tokenizer, "num_special_tokens_to_add", None)

    if count_special_tokens is None:
        return 0

    return count_special_tokens(pair=False)


def make_chunk(
    ordinal: int,
    text: str,
    start_offset: int,
    end_offset: int,
    tokenizer: Any,
    settings: ChunkingSettings,
    fallback_reason: str | None = None,
) -> ChunkSlice | None:
    """Build one chunk and verify its token and model input limits."""
    chunk_text_value = text[start_offset:end_offset]

    if not chunk_text_value.strip():
        return None

    count = token_count(chunk_text_value, tokenizer)

    if count > settings.chunk_size:
        raise ValueError("Chunk is longer than the configured chunk_size")

    if count + special_token_count(tokenizer) > settings.max_input_tokens:
        raise ValueError("Chunk plus special tokens exceeds max_input_tokens")

    return ChunkSlice(
        ordinal=ordinal,
        text=chunk_text_value,
        token_count=count,
        start_offset=start_offset,
        end_offset=end_offset,
        fallback_reason=fallback_reason,
    )


def fixed_token_ranges(
    text: str,
    tokenizer: Any,
    settings: ChunkingSettings,
) -> list[tuple[int, int]]:
    """Return source ranges for fixed-size token windows."""
    offsets = token_offsets(text, tokenizer)

    if not offsets:
        return []

    ranges = []
    token_start = 0

    while token_start < len(offsets):
        token_end = min(token_start + settings.chunk_size, len(offsets))
        start_offset = 0 if token_start == 0 else offsets[token_start][0]
        end_offset = len(text) if token_end == len(offsets) else offsets[token_end][0]

        while token_end > token_start:
            piece = text[start_offset:end_offset]

            if token_count(piece, tokenizer) <= settings.chunk_size:
                break

            token_end -= 1
            end_offset = len(text) if token_end == len(offsets) else offsets[token_end][0]

        if token_end == token_start:
            raise ValueError("Could not split text within the configured chunk_size")

        ranges.append((start_offset, end_offset))

        if token_end == len(offsets):
            break

        token_start = max(token_start + 1, token_end - settings.overlap)

    return ranges


def chunks_from_ranges(
    text: str,
    ranges: list[tuple[int, int]],
    tokenizer: Any,
    settings: ChunkingSettings,
    fallback_reason: str | None = None,
    ordinal_start: int = 0,
) -> list[ChunkSlice]:
    """Build checked chunks from ordered source ranges."""
    chunks = []

    for start_offset, end_offset in ranges:
        chunk = make_chunk(
            ordinal=ordinal_start + len(chunks),
            text=text,
            start_offset=start_offset,
            end_offset=end_offset,
            tokenizer=tokenizer,
            settings=settings,
            fallback_reason=fallback_reason,
        )

        if chunk is not None:
            chunks.append(chunk)

    return chunks
