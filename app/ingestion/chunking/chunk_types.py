"""Shared data types for the chunking strategies."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkingSettings:
    """Token limits shared by chunking strategies."""

    chunk_size: int
    overlap: int = 0
    max_input_tokens: int = 1200

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be at least zero and smaller than chunk_size")

        if self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be greater than zero")


@dataclass(frozen=True)
class ChunkSlice:
    """One chunk and its character range in the canonical source text."""

    order: int
    text: str
    token_count: int
    start_char: int
    end_char: int
    fallback_reason: str | None = None


class ChunkingStrategy:
    """Base interface implemented by each selectable chunking strategy."""

    strategy_id = "base"
    strategy_version = "1"

    def split(
        self,
        text: str,
        tokenizer: Any,
        settings: ChunkingSettings,
    ) -> list[ChunkSlice]:
        raise NotImplementedError
