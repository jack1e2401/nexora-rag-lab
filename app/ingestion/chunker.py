"""Compatibility imports for the original chunker module."""

from app.ingestion.chunking import (
    ChunkSlice,
    ChunkingSettings,
    ChunkingStrategy,
    DocumentStructureStrategy,
    FixedTokenStrategy,
    RecursiveStrategy,
    chunk_text,
    get_chunking_strategy,
)

__all__ = [
    "ChunkSlice",
    "ChunkingSettings",
    "ChunkingStrategy",
    "DocumentStructureStrategy",
    "FixedTokenStrategy",
    "RecursiveStrategy",
    "chunk_text",
    "get_chunking_strategy",
]
