"""Find character positions where chunks should start and end in source text."""

from typing import Any

from app.ingestion.chunking.chunk_types import ChunkingSettings
from app.ingestion.chunking.utils.token_utils import count_tokens


def get_token_character_offsets(
    text: str,
    tokenizer: Any,
) -> list[tuple[int, int]]:
    """Return each token's start and end character positions in the source text."""
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    token_offsets = encoded["offset_mapping"]
    character_offsets = []

    for start_char, end_char in token_offsets:
        if start_char < 0 or end_char > len(text) or end_char < start_char:
            raise ValueError("Tokenizer returned invalid character positions")

        if end_char > start_char:
            character_offsets.append((start_char, end_char))

    if text.strip() and len(character_offsets) != len(token_ids):
        raise ValueError("Tokenizer did not return positions for every token")

    return character_offsets


def get_fixed_token_chunk_character_ranges(
    text: str,
    tokenizer: Any,
    settings: ChunkingSettings,
) -> list[tuple[int, int]]:
    """Find character ranges for chunks whose maximum size is measured in tokens."""
    token_character_offsets = get_token_character_offsets(text, tokenizer)

    if not token_character_offsets:
        return []

    character_ranges = []
    token_start = 0

    while token_start < len(token_character_offsets):
        token_end = min(token_start + settings.chunk_size, len(token_character_offsets))

        if token_start == 0:
            start_char = 0
        else:
            start_char = token_character_offsets[token_start][0]

        if token_end == len(token_character_offsets):
            end_char = len(text)
        else:
            end_char = token_character_offsets[token_end][0]

        while token_end > token_start:
            text_piece = text[start_char:end_char]

            if count_tokens(text_piece, tokenizer) <= settings.chunk_size:
                break

            token_end -= 1
            if token_end == len(token_character_offsets):
                end_char = len(text)
            else:
                end_char = token_character_offsets[token_end][0]

        if token_end == token_start:
            raise ValueError("Could not split text within the configured chunk_size")

        character_ranges.append((start_char, end_char))

        if token_end == len(token_character_offsets):
            break

        token_start = max(token_start + 1, token_end - settings.overlap)

    return character_ranges
