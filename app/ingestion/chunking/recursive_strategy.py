"""Recursive separator-aware chunking strategy."""

import re
from typing import Any

from app.ingestion.chunking.utils.chunk_utils import (
    create_chunks_from_character_ranges,
)
from app.ingestion.chunking.utils.token_utils import count_tokens
from app.ingestion.chunking.utils.text_ranges import (
    get_fixed_token_chunk_character_ranges,
)
from app.ingestion.chunking.chunk_types import (
    ChunkSlice,
    ChunkingSettings,
    ChunkingStrategy,
)


def split_oversized_text_range(
    text: str,
    start_char: int,
    end_char: int,
    tokenizer: Any,
    settings: ChunkingSettings,
) -> list[tuple[int, int]]:
    """Recursively split a range at the strongest available separator."""
    piece = text[start_char:end_char]

    if not piece.strip():
        return []

    if count_tokens(piece, tokenizer) <= settings.chunk_size:
        return [(start_char, end_char)]

    separators = [
        r"\n\s*\n",
        r"\n",
        r"(?<=[.!?。！？])\s+",
        r"\s+",
    ]

    for separator in separators:
        matches = list(re.finditer(separator, piece))
        if not matches:
            continue

        smaller_ranges = []
        part_start_char = start_char

        for match in matches:
            part_end_char = start_char + match.end()
            if part_end_char > part_start_char and part_end_char < end_char:
                smaller_ranges.append((part_start_char, part_end_char))
                part_start_char = part_end_char

        if part_start_char < end_char and smaller_ranges:
            smaller_ranges.append((part_start_char, end_char))

        if len(smaller_ranges) > 1:
            result = []
            for smaller_start, smaller_end in smaller_ranges:
                result.extend(
                    split_oversized_text_range(
                        text,
                        smaller_start,
                        smaller_end,
                        tokenizer,
                        settings,
                    )
                )
            return result

    fallback_settings = ChunkingSettings(
        chunk_size=settings.chunk_size,
        overlap=0,
        max_input_tokens=settings.max_input_tokens,
    )
    local_character_ranges = get_fixed_token_chunk_character_ranges(
        piece,
        tokenizer,
        fallback_settings,
    )
    return [
        (start_char + local_start, start_char + local_end)
        for local_start, local_end in local_character_ranges
    ]


def pack_text_ranges(
    text: str,
    units: list[tuple[int, int]],
    tokenizer: Any,
    settings: ChunkingSettings,
) -> list[tuple[int, int]]:
    """Pack adjacent small units together and keep whole-unit overlap."""
    if not units:
        return []

    packed_ranges = []
    unit_start = 0

    while unit_start < len(units):
        unit_end = unit_start
        chunk_end = units[unit_start][0]

        while unit_end < len(units):
            candidate_end = units[unit_end][1]
            candidate_start = units[unit_start][0]
            candidate_text = text[candidate_start:candidate_end]

            if count_tokens(candidate_text, tokenizer) > settings.chunk_size:
                break

            chunk_end = candidate_end
            unit_end += 1

        if unit_end == unit_start:
            raise ValueError("A recursive chunk unit is longer than chunk_size")

        chunk_start = units[unit_start][0]
        packed_ranges.append((chunk_start, chunk_end))

        if unit_end == len(units):
            break

        next_start = unit_end
        if settings.overlap:
            for candidate_start in range(unit_end - 1, unit_start - 1, -1):
                overlap_start = units[candidate_start][0]
                overlap_text = text[overlap_start:chunk_end]
                if count_tokens(overlap_text, tokenizer) <= settings.overlap:
                    next_start = candidate_start
                else:
                    break

        unit_start = next_start if next_start > unit_start else unit_end

    return packed_ranges


class RecursiveStrategy(ChunkingStrategy):
    """Prefer paragraph, line, and sentence boundaries before token fallback."""

    strategy_id = "recursive"

    def split(
        self,
        text: str,
        tokenizer: Any,
        settings: ChunkingSettings,
    ) -> list[ChunkSlice]:
        text_ranges = split_oversized_text_range(text, 0, len(text), tokenizer, settings)
        packed_ranges = pack_text_ranges(text, text_ranges, tokenizer, settings)
        return create_chunks_from_character_ranges(
            text,
            packed_ranges,
            tokenizer,
            settings,
        )
