# Plain-Text Extraction and Two Chunkers Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Stop after each task for the user's test and code review. Do not commit until the user requests it.

**Goal:** Simplify the lab to Tika plain text and two chunking approaches: fixed-token and DeepSeek LLM.

**Architecture:** Tika supplies one normalized plain-text string per document. The chunking interface keeps source character spans; fixed-token is available now, and LLM chunking is added in the next task.

**Tech Stack:** Python, FastAPI, Apache Tika 4.1, BGE-M3, DeepSeek.

**Spec:** [Plain-Text Extraction and Two Chunkers](../specs/2026-10-27-plain-text-two-chunkers-design.md)

## Global Constraints

- Keep code readable for a Python learner familiar with C++.
- Keep the `fixed_token` ID and BGE token units for chunk size and overlap.
- Keep exact character spans against the normalized plain text.
- Stop after one task, explain the code, and give manual test steps.
- Preserve existing uncommitted work; do not commit without a later user request.

## Review Focus

- Tika may flatten DOCX paragraphs or PDF layout; inspect real corpus outputs before ingestion.
- Swagger must return `text`, not a misleading `markdown` field.
- Removed strategies must not remain selectable or imported.
- Fixed-token chunks must cover all non-whitespace source characters within token limits.
- The future LLM chunker must reject invented, missing, overlapping, or oversized spans.

## Task 1: Plain text and fixed-token-only checkpoint

**Files:** Modify `app/ingestion/extractor.py`, `app/lab_api.py`, `app/ingestion/chunking/__init__.py`, `app/ingestion/chunker.py`, `tests/test_extractor.py`, `tests/test_lab_api.py`, and `tests/test_chunker.py`; remove `app/ingestion/chunking/recursive_strategy.py` and `app/ingestion/chunking/markdown_structure_strategy.py`.

**Interfaces:** `extract_text` and `extract_bytes` return normalized plain text from `PUT /tika/text`; `/ingestion/extract` returns `text`; `/chunking/split` accepts `text` and only `strategy="fixed_token"`.

- [x] Update tests for the plain-text request and response, fixed-token-only strategy lookup, source span coverage, and removed Swagger choices.
- [x] Run focused tests and confirm they fail for the old Markdown/multiple-strategy behavior.
- [x] Make the smallest code changes, removing obsolete strategy modules and imports.
- [x] Run the full test suite and `git diff --check`.
- [x] Smoke-test one real DOCX against local Tika, then stop for user review without committing.

The DOCX smoke test used `English_8_15min_Teenagers.docx` and returned 3,546 characters of normalized plain text. The full suite passed with 33 tests.
The user also inspected two fixed-token chunks at 500 tokens with 50-token overlap. The first chunk ended mid-sentence; the second contained that sentence in full. A local `/embeddings/embed` request returned a 1,024-value BGE-M3 vector. These checks validate the current preview workflow, not retrieval quality or database persistence.

## Task 2: DeepSeek LLM chunking

**Files:** Create `app/ingestion/chunking/llm_strategy.py`; modify the strategy selection, Swagger chunk endpoint, settings if required, and focused tests. Keep the LLM-specific code in one focused module.

**Interfaces:** DeepSeek proposes boundaries over the canonical plain text; Python validates exact source spans, full meaningful-text coverage, deterministic ordering, and BGE token/input limits. Return chunk text and character spans in the same `ChunkSlice` shape as fixed-token.

- [ ] Define the DeepSeek request/response contract, prompt version, maximum input size, and failure behavior before code.
- [ ] Use the observed sentence split near `Although Nam is shy, he enjoys joining school clubs.` as a review example for more meaningful boundaries.
- [ ] Add a failing test for valid boundaries and separate tests for invented, missing, duplicate, and oversized spans.
- [ ] Implement the LLM chunker and expose `strategy="llm"` in Swagger.
- [ ] Run full tests and a controlled real DeepSeek smoke test; report latency and token use.
- [ ] Stop for user review without committing.
