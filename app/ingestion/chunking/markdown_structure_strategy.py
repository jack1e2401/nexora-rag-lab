"""Markdown document-structure-aware chunking strategy."""

import re
from typing import Any

from app.ingestion.chunking.chunk_types import (
    ChunkSlice,
    ChunkingSettings,
    ChunkingStrategy,
)
from app.ingestion.chunking.recursive_strategy import (
    pack_text_ranges,
    split_oversized_text_range,
)
from app.ingestion.chunking.utils.chunk_utils import (
    create_chunks_from_character_ranges,
)
from app.ingestion.chunking.utils.token_utils import count_tokens


class DocumentStructureStrategy(ChunkingStrategy):
    """Keep Markdown headings with their sections and split large sections."""

    strategy_id = "document_structure"
    heading_pattern = re.compile(r"(?m)^ {0,3}#{1,6}[ \t]+")

    def split(
        self,
        text: str,
        tokenizer: Any,
        settings: ChunkingSettings,
    ) -> list[ChunkSlice]:
        headings = list(self.heading_pattern.finditer(text))

        if not headings:
            text_ranges = split_oversized_text_range(
                text,
                0,
                len(text),
                tokenizer,
                settings,
            )
            packed_ranges = pack_text_ranges(text, text_ranges, tokenizer, settings)
            return create_chunks_from_character_ranges(
                text,
                packed_ranges,
                tokenizer,
                settings,
                fallback_reason="no_markdown_headings",
            )

        section_starts = sorted(set([0] + [heading.start() for heading in headings]))
        sections = []

        for index, section_start in enumerate(section_starts):
            if index + 1 < len(section_starts):
                section_end = section_starts[index + 1]
            else:
                section_end = len(text)

            if section_end > section_start:
                sections.append((section_start, section_end))

        chunks = []
        for section_start, section_end in sections:
            section_text = text[section_start:section_end]
            fallback_reason = None

            if count_tokens(section_text, tokenizer) > settings.chunk_size:
                fallback_reason = "oversized_section"

            text_ranges = split_oversized_text_range(
                text,
                section_start,
                section_end,
                tokenizer,
                settings,
            )
            packed_ranges = pack_text_ranges(text, text_ranges, tokenizer, settings)
            section_chunks = create_chunks_from_character_ranges(
                text,
                packed_ranges,
                tokenizer,
                settings,
                fallback_reason=fallback_reason,
                order_start=len(chunks),
            )
            chunks.extend(section_chunks)

        return chunks
