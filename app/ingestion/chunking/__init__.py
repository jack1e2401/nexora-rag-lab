"""Public chunking types and the available strategy selection."""

from app.ingestion.chunking.chunk_types import ChunkSlice, ChunkingSettings, ChunkingStrategy
from app.ingestion.chunking.fixed_token_strategy import FixedTokenStrategy, chunk_text


def get_chunking_strategy(strategy_id: str) -> ChunkingStrategy:
    """Return the fixed-token strategy currently available in the lab."""
    if strategy_id == FixedTokenStrategy.strategy_id:
        return FixedTokenStrategy()

    raise ValueError(f"Unknown chunking strategy: {strategy_id}")


__all__ = [
    "ChunkSlice",
    "ChunkingSettings",
    "ChunkingStrategy",
    "FixedTokenStrategy",
    "chunk_text",
    "get_chunking_strategy",
]
