"""Fixed-token chunking strategy."""

from typing import Any

from app.ingestion.chunking.chunk_helpers import chunks_from_ranges, fixed_token_ranges
from app.ingestion.chunking.chunk_types import ChunkSlice, ChunkingSettings, ChunkingStrategy


class FixedTokenStrategy(ChunkingStrategy):
    """Split text into fixed BGE-token windows with token overlap."""

    strategy_id = "fixed_token"

    def split(
        self,
        text: str,
        tokenizer: Any,
        settings: ChunkingSettings,
    ) -> list[ChunkSlice]:
        ranges = fixed_token_ranges(text, tokenizer, settings)
        return chunks_from_ranges(text, ranges, tokenizer, settings)


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    tokenizer: Any,
) -> list[ChunkSlice]:
    """Keep the original helper as a wrapper around fixed-token chunking."""
    settings = ChunkingSettings(chunk_size=chunk_size, overlap=overlap)
    return FixedTokenStrategy().split(text, tokenizer, settings)
