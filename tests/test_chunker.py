import importlib

import pytest

from app.ingestion.chunker import (
    ChunkingSettings,
    DocumentStructureStrategy,
    FixedTokenStrategy,
    RecursiveStrategy,
    chunk_text,
    get_chunking_strategy,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        token_ids = [ord(character) for character in text]
        if add_special_tokens:
            return [1, *token_ids, 2]
        return token_ids

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        characters = []
        for token_id in token_ids:
            if token_id not in (1, 2):
                characters.append(chr(token_id))
        return "".join(characters)

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        result = {"input_ids": self.encode(text, add_special_tokens)}

        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(text))]

        return result


class UnknownTokenTokenizer(CharacterTokenizer):
    def encode(self, text, add_special_tokens=False):
        if text in ("☃", "<unk>"):
            return [0]
        return super().encode(text, add_special_tokens=add_special_tokens)

    def decode(
        self,
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        if token_ids == [0]:
            return "" if skip_special_tokens else "<unk>"
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )


class MissingOffsetTokenizer(CharacterTokenizer):
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        return {"input_ids": [1, 2], "offset_mapping": [(0, 1)]}


def test_unicode_chunks_are_stable_and_overlap():
    tokenizer = CharacterTokenizer()
    text = "Tiếng Việt có dấu: trách nhiệm, điều khoản và bằng chứng. " * 2

    first_run = chunk_text(text, chunk_size=32, overlap=5, tokenizer=tokenizer)
    second_run = chunk_text(text, chunk_size=32, overlap=5, tokenizer=tokenizer)

    assert first_run == second_run
    assert len(first_run) > 1

    for chunk in first_run:
        assert 0 < chunk.token_count <= 32
        assert "\ufffd" not in chunk.text

    assert first_run[1].text.startswith(first_run[0].text[-5:])


def test_fixed_strategy_preserves_original_text_for_unknown_token():
    chunks = chunk_text("☃", chunk_size=10, overlap=0, tokenizer=UnknownTokenTokenizer())

    assert [chunk.text for chunk in chunks] == ["☃"]


def test_larger_experiment_chunks_fit_model_input_limit():
    tokenizer = CharacterTokenizer()
    text = "A" * 2500

    chunks = chunk_text(text, chunk_size=1000, overlap=100, tokenizer=tokenizer)

    for chunk in chunks:
        token_count_with_special_tokens = (
            chunk.token_count + tokenizer.num_special_tokens_to_add(pair=False)
        )
        assert token_count_with_special_tokens < 1200


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_size_or_overlap_raises(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size, overlap, CharacterTokenizer())


def test_fixed_strategy_chunks_keep_exact_source_spans():
    text = "abcdefghijklmno"
    strategy = FixedTokenStrategy()

    chunks = strategy.split(
        text,
        CharacterTokenizer(),
        ChunkingSettings(chunk_size=8, overlap=2),
    )

    assert [chunk.text for chunk in chunks] == ["abcdefgh", "ghijklmn", "mno"]
    assert [chunk.order for chunk in chunks] == [0, 1, 2]

    for chunk in chunks:
        assert text[chunk.start_char : chunk.end_char] == chunk.text
        assert chunk.token_count <= 8


def test_recursive_strategy_splits_at_paragraph_boundary():
    text = "one two.\n\nthree four."
    strategy = RecursiveStrategy()

    chunks = strategy.split(
        text,
        CharacterTokenizer(),
        ChunkingSettings(chunk_size=15, overlap=0),
    )

    assert len(chunks) == 2
    assert chunks[0].text.startswith("one two.")
    assert "three" not in chunks[0].text
    assert chunks[1].text.startswith("three four.")


def test_document_structure_strategy_keeps_heading_with_table_and_list():
    text = "# First\nIntro\n\n# Second\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n- item one\n- item two"
    strategy = DocumentStructureStrategy()

    chunks = strategy.split(
        text,
        CharacterTokenizer(),
        ChunkingSettings(chunk_size=75, overlap=0),
    )

    first_section = next(chunk for chunk in chunks if chunk.text.startswith("# First"))
    second_section = next(chunk for chunk in chunks if chunk.text.startswith("# Second"))

    assert "Intro" in first_section.text
    assert "| 1 | 2 |" in second_section.text
    assert "- item one\n- item two" in second_section.text
    assert text[first_section.start_char : first_section.end_char] == first_section.text
    assert text[second_section.start_char : second_section.end_char] == second_section.text
    assert first_section.fallback_reason is None
    assert second_section.fallback_reason is None


def test_document_structure_reports_recursive_fallback_without_headings():
    chunks = DocumentStructureStrategy().split(
        "A paragraph without a Markdown heading.",
        CharacterTokenizer(),
        ChunkingSettings(chunk_size=18, overlap=0),
    )

    assert chunks
    assert all(chunk.fallback_reason == "no_markdown_headings" for chunk in chunks)


def test_all_strategies_cover_every_non_whitespace_character():
    text = "Heading\n\nA sentence with punctuation.\nA second sentence."
    strategies = [
        FixedTokenStrategy(),
        RecursiveStrategy(),
        DocumentStructureStrategy(),
    ]

    for strategy in strategies:
        chunks = strategy.split(
            text,
            CharacterTokenizer(),
            ChunkingSettings(chunk_size=18, overlap=3),
        )
        covered_positions = set()

        for chunk in chunks:
            assert text[chunk.start_char : chunk.end_char] == chunk.text
            assert chunk.token_count <= 18
            assert chunk.token_count + 2 <= 1200
            covered_positions.update(range(chunk.start_char, chunk.end_char))

        for index, character in enumerate(text):
            if not character.isspace():
                assert index in covered_positions


def test_fixed_strategy_rejects_missing_token_character_positions():
    with pytest.raises(ValueError, match="positions for every token"):
        FixedTokenStrategy().split(
            "text",
            MissingOffsetTokenizer(),
            ChunkingSettings(chunk_size=3),
        )


def test_strategy_lookup_returns_explicit_strategies():
    assert isinstance(get_chunking_strategy("fixed_token"), FixedTokenStrategy)
    assert isinstance(get_chunking_strategy("recursive"), RecursiveStrategy)
    assert isinstance(
        get_chunking_strategy("document_structure"),
        DocumentStructureStrategy,
    )

    with pytest.raises(ValueError, match="Unknown chunking strategy"):
        get_chunking_strategy("semantic")


def test_chunking_package_and_compatibility_module_share_public_classes():
    chunking_package = importlib.import_module("app.ingestion.chunking")

    assert chunking_package.ChunkSlice is type(
        FixedTokenStrategy().split(
            "source",
            CharacterTokenizer(),
            ChunkingSettings(chunk_size=10),
        )[0]
    )
    assert chunking_package.get_chunking_strategy is get_chunking_strategy


def test_chunking_modules_have_descriptive_names():
    module_names = [
        "app.ingestion.chunking.chunk_types",
        "app.ingestion.chunking.utils.token_utils",
        "app.ingestion.chunking.utils.text_ranges",
        "app.ingestion.chunking.utils.chunk_utils",
        "app.ingestion.chunking.fixed_token_strategy",
        "app.ingestion.chunking.recursive_strategy",
        "app.ingestion.chunking.markdown_structure_strategy",
    ]

    for module_name in module_names:
        importlib.import_module(module_name)


def test_chunking_utility_functions_have_clear_names():
    token_utils = importlib.import_module(
        "app.ingestion.chunking.utils.token_utils"
    )
    chunk_utils = importlib.import_module(
        "app.ingestion.chunking.utils.chunk_utils"
    )
    text_ranges = importlib.import_module(
        "app.ingestion.chunking.utils.text_ranges"
    )

    assert callable(token_utils.count_tokens)
    assert callable(token_utils.count_special_tokens)
    assert callable(text_ranges.get_token_character_offsets)
    assert callable(text_ranges.get_fixed_token_chunk_character_ranges)
    assert callable(chunk_utils.create_chunk)
    assert callable(chunk_utils.create_chunks_from_character_ranges)
