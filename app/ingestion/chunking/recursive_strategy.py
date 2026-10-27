"""Recursive separator-aware chunking strategy."""

import re
from typing import Any

from app.ingestion.chunking.chunk_helpers import (
    chunks_from_ranges,
    fixed_token_ranges,
    token_count,
)
from app.ingestion.chunking.chunk_types import ChunkSlice, ChunkingSettings, ChunkingStrategy


def split_oversized_range(
    text: str,
    start_offset: int,
    end_offset: int,
    tokenizer: Any,
    settings: ChunkingSettings,
) -> list[tuple[int, int]]:
    """Recursively split a range at the strongest available separator."""
    piece = text[start_offset:end_offset]

    if not piece.strip():
        return []

    if token_count(piece, tokenizer) <= settings.chunk_size:
        return [(start_offset, end_offset)]

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
        part_start = start_offset

        for match in matches:
            part_end = start_offset + match.end()
            if part_end > part_start and part_end < end_offset:
                smaller_ranges.append((part_start, part_end))
                part_start = part_end

        if part_start < end_offset and smaller_ranges:
            smaller_ranges.append((part_start, end_offset))

        if len(smaller_ranges) > 1:
            result = []
            for smaller_start, smaller_end in smaller_ranges:
                result.extend(
                    split_oversized_range(
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
    return [
        (start_offset + start, start_offset + end)
        for start, end in fixed_token_ranges(piece, tokenizer, fallback_settings)
    ]


def pack_ranges(
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

            if token_count(candidate_text, tokenizer) > settings.chunk_size:
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
                if token_count(overlap_text, tokenizer) <= settings.overlap:
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
        units = split_oversized_range(text, 0, len(text), tokenizer, settings)
        ranges = pack_ranges(text, units, tokenizer, settings)
        return chunks_from_ranges(text, ranges, tokenizer, settings)
