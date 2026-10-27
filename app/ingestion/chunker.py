"""Compatibility imports for the original chunker module."""

from app.ingestion.chunking import (
    ChunkSlice,
    ChunkingSettings,
    ChunkingStrategy,
    FixedTokenStrategy,
    chunk_text,
    get_chunking_strategy,
)

__all__ = [
    "ChunkSlice",
    "ChunkingSettings",
    "ChunkingStrategy",
    "FixedTokenStrategy",
    "chunk_text",
    "get_chunking_strategy",
]
