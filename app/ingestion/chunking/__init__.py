"""Public chunking types and Phase 1 strategy selection."""

from app.ingestion.chunking.chunk_types import ChunkSlice, ChunkingSettings, ChunkingStrategy
from app.ingestion.chunking.fixed_token_strategy import FixedTokenStrategy, chunk_text
from app.ingestion.chunking.markdown_structure_strategy import DocumentStructureStrategy
from app.ingestion.chunking.recursive_strategy import RecursiveStrategy


def get_chunking_strategy(strategy_id: str) -> ChunkingStrategy:
    """Return one of the explicitly supported Phase 1 strategies."""
    strategies = {
        FixedTokenStrategy.strategy_id: FixedTokenStrategy,
        RecursiveStrategy.strategy_id: RecursiveStrategy,
        DocumentStructureStrategy.strategy_id: DocumentStructureStrategy,
    }

    try:
        strategy_class = strategies[strategy_id]
    except KeyError as error:
        raise ValueError(f"Unknown chunking strategy: {strategy_id}") from error

    return strategy_class()


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
