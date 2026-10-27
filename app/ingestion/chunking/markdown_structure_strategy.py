"""Markdown document-structure-aware chunking strategy."""

import re
from typing import Any

from app.ingestion.chunking.chunk_helpers import chunks_from_ranges, token_count
from app.ingestion.chunking.chunk_types import ChunkSlice, ChunkingSettings, ChunkingStrategy
from app.ingestion.chunking.recursive_strategy import pack_ranges, split_oversized_range


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
            units = split_oversized_range(text, 0, len(text), tokenizer, settings)
            ranges = pack_ranges(text, units, tokenizer, settings)
            return chunks_from_ranges(
                text,
                ranges,
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

            if token_count(section_text, tokenizer) > settings.chunk_size:
                fallback_reason = "oversized_section"

            units = split_oversized_range(
                text,
                section_start,
                section_end,
                tokenizer,
                settings,
            )
            ranges = pack_ranges(text, units, tokenizer, settings)
            section_chunks = chunks_from_ranges(
                text,
                ranges,
                tokenizer,
                settings,
                fallback_reason=fallback_reason,
                ordinal_start=len(chunks),
            )
            chunks.extend(section_chunks)

        return chunks
