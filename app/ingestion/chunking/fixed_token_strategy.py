"""Fixed-token chunking strategy."""

from typing import Any

from app.ingestion.chunking.chunk_types import (
    ChunkSlice,
    ChunkingSettings,
    ChunkingStrategy,
)
from app.ingestion.chunking.utils.chunk_utils import (
    create_chunks_from_character_ranges,
)
from app.ingestion.chunking.utils.text_ranges import (
    get_fixed_token_chunk_character_ranges,
)


class FixedTokenStrategy(ChunkingStrategy):
    """Split text into fixed BGE-token windows with token overlap."""

    strategy_id = "fixed_token"

    def split(
        self,
        text: str,
        tokenizer: Any,
        settings: ChunkingSettings,
    ) -> list[ChunkSlice]:
        character_ranges = get_fixed_token_chunk_character_ranges(
            text,
            tokenizer,
            settings,
        )
        return create_chunks_from_character_ranges(
            text,
            character_ranges,
            tokenizer,
            settings,
        )


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    tokenizer: Any,
) -> list[ChunkSlice]:
    """Keep the original helper as a wrapper around fixed-token chunking."""
    settings = ChunkingSettings(chunk_size=chunk_size, overlap=overlap)
    return FixedTokenStrategy().split(text, tokenizer, settings)
