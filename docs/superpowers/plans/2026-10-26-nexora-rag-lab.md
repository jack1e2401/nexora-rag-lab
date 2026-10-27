# Nexora RAG Lab Implementation Plan

> Extraction and chunking tasks in this plan are superseded by [Plain-Text Extraction and Two Chunkers](2026-10-27-plain-text-two-chunkers.md).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible local RAG lab that ingests the supplied 24-document benchmark, answers questions with verified citations, and compares chunking strategies on the same labeled cases.

**Architecture:** One synchronous ingestion CLI runs local BGE-M3 and writes strategy-specific chunks to PostgreSQL/pgvector. FastAPI exposes health, search, and DeepSeek-backed ask; an evaluation CLI invokes the same RAG pipeline directly, records paired strategy comparisons, and publishes Langfuse experiments. The approved chunking design defines the three Phase 1 strategies and the shared comparison controls.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, psycopg 3, pgvector, sentence-transformers/PyTorch, `httpx` for DeepSeek generation and judging, Apache Tika 4.1.0, PostgreSQL 17, Langfuse Python SDK v5, pytest.

**Working Style:** Prefer straightforward Python with descriptive names, small functions, explicit control flow, and familiar syntax. Pause after each task for the user to run the suggested checks and review the code. Do not commit until all tasks are complete.

**Spec:** [Nexora RAG Lab design](../specs/2026-10-26-nexora-rag-lab-design.md)

**Chunking spec:** [Chunking strategy comparison design](../specs/2026-10-27-nexora-chunking-strategies-design.md)

## Global Constraints

- Keep the code to `app/main.py`, `config.py`, `database.py`, `embeddings.py`, three ingestion modules, three RAG modules, and two scripts unless a real duplication warrants another file.
- Ingest only files listed under `01_CORPUS/` in the supplied manifest. Never index the question files, answers, README, or manifest.
- Verify the supplied benchmark ZIP SHA-256 is `9820F73D97E46856DB8D5940A67C404557E435A55C53D254FB34F4D137CE8A37` before staging.
- Embedding model is `BAAI/bge-m3` at revision `31e47391fcbda65be526abe98e646b3c6cd845a8`, dense 1024-dimensional vectors. Use CPU float32, normalized vectors, batch size 4, and 1200-token model input cap for the baseline. Keep model/device/precision fixed across compared strategies.
- Generation model is DeepSeek `deepseek-flash`; formal evaluation judge is DeepSeek `deepseek-v4-pro`. Both use non-thinking mode through direct HTTPS requests. Record actual response model IDs, prompt/rubric versions, and fingerprints when returned.
- Do not use an OpenAI account, API, SDK, or credentials anywhere in the lab.
- Use the pinned BGE-M3 tokenizer for every strategy and enforce a 1200-token model input limit including special tokens.
- The corpus reference date is 2026-10-26. Treat future dates in questions as statements about that snapshot.
- Keep ingestion synchronous and profile writes atomic. No queue, worker, upload API, authentication, RBAC, frontend, or Nexora integration.
- Compare fixed-token (500/50), recursive, and document-structure strategies first, all at `top_k=3` and about 500 tokens maximum. Defer fixed-token 1000/100 and `top_k=5` to parameter tuning.
- Exclude all 27 `SECURITY` cases from the primary RAG score. Experimental Board filtering is not an access-control claim.
- Use the supplied `expected_document_ids` for **DocumentRecall@K**. Do not report chunk recall from those labels.
- Pass only the case question and Board partition to RAG; never pass expected answers, rationales, or gold document IDs into the generation prompt.
- Compare quality, latency, and cost separately. Do not invent a weighted overall score or declare a universal winner.
- Do not treat ZIP contents, including its README and sample Python script, as implementation instructions. Validate the files as data before use.

## Review Focus

1. A changed corpus, extraction representation, model revision, or embedding width must not leave strategy profiles silently incomparable; Task 3 tests their fingerprints and model guard.
2. Unicode text, BGE token overlap, and model input limits must not produce malformed or truncated chunks; Task 2 tests Vietnamese input and boundary cases.
3. Failed extraction or embedding must not leave half of a profile in PostgreSQL; Task 3 tests the atomic replacement path.
4. Malformed DeepSeek JSON or a citation label absent from retrieved context must never become a returned citation; Task 5 tests rejection and abstention.
5. A case with several gold documents must not score as full recall after only one hit; Task 6 tests per-case recall and all-gold coverage.

---

## File map

| Path | Responsibility |
| --- | --- |
| `app/config.py` | Typed environment settings and defaults. |
| `app/database.py` | psycopg connection and small schema/profile helpers. |
| `app/embeddings.py` | One lazy BGE-M3 loader/tokenizer and identical passage/query embedding settings. |
| `app/main.py` | FastAPI entry point and health route. |
| `app/lab_api.py` | Swagger endpoints for extraction, token inspection, chunking, and embedding. |
| `app/tika.py` | Apache Tika health check and HTTP helpers. |
| `app/ingestion/extractor.py` | Tika 4 Markdown extraction and structure-preserving text normalization. |
| `app/ingestion/chunking/chunk_types.py` | Shared chunk settings, source-span result, and strategy interface. |
| `app/ingestion/chunking/utils/token_utils.py` | Count content and special tokens. |
| `app/ingestion/chunking/utils/text_ranges.py` | Map tokenizer positions to source character ranges. |
| `app/ingestion/chunking/utils/chunk_utils.py` | Create and validate chunks from source character ranges. |
| `app/ingestion/chunking/fixed_token_strategy.py` | Fixed-token strategy and the original helper wrapper. |
| `app/ingestion/chunking/recursive_strategy.py` | Separator-aware splitting and range packing. |
| `app/ingestion/chunking/markdown_structure_strategy.py` | Markdown section splitting and fallback reporting. |
| `app/ingestion/chunking/__init__.py` | Public strategy exports and explicit strategy lookup. |
| `app/ingestion/chunker.py` | Compatibility imports for the original chunker module. |
| `app/ingestion/pipeline.py` | Manifest-driven synchronous ingest, local embedding, atomic profile replacement, report. |
| `app/rag/retrieval.py` | Local query embedding and exact cosine search scoped by Board/profile. |
| `app/rag/generation.py` | Grounded DeepSeek prompt and validated JSON response. |
| `app/rag/pipeline.py` | Search → generate → verify citations → final answer. |
| `scripts/ingest.py` | Ingestion CLI. |
| `scripts/evaluate.py` | Benchmark validation, scoring, Langfuse runs, comparison output. |
| `sql/schema.sql` | `documents`, `profiles`, `chunks`, pgvector extension and constraints. |
| `data/benchmark/` | Staged original `manifest.json`, `01_CORPUS/`, and `BoCauHoi/questions_300.jsonl`. |
| `experiments/` | Versioned human-readable experiment summaries; raw runs under ignored `data/results/`. |

### Task 1: Foundation and health

**Files:** Create `pyproject.toml`, `.env.example`, `.gitignore`, `docker-compose.yml`, `sql/schema.sql`, `app/__init__.py`, `app/config.py`, `app/database.py`, `app/tika.py`, `app/main.py`; test `tests/test_health.py`.

**Interfaces:** `get_settings() -> Settings`; `get_connection(settings: Settings) -> psycopg.Connection`; `GET /health -> {"database":"ok","tika":"ok"}` or HTTP 503 with the failing dependency named. Schema fields: `documents(document_id,board_id,source_path,content_sha256,extracted_text)`, `profiles(profile_id,chunk_size,overlap,embedding_model,embedding_revision,embedding_dimensions,embedding_settings jsonb,corpus_fingerprint)`, `chunks(chunk_id,profile_id,document_id,ordinal,text,token_count,embedding vector(1024))`. `embedding_settings` records device, precision, normalization, tokenizer, batch size, and model input cap.

- [x] **Step 1: Write failing health/config tests.** `test_health_ok`: assert both dependency fields equal `"ok"`; `test_health_names_failed_dependency`: assert HTTP 503 identifies the failed dependency; `test_settings_require_database_url`: assert a validation error.
- [x] **Step 2: Run `python -m pytest tests/test_health.py -q`.** Expected: failures because the app/config modules are absent.
- [x] **Step 3: Implement files above.** Use `pgvector/pgvector:pg17` and `apache/tika:4.1.0` in Compose, bound to localhost. Check PostgreSQL with `SELECT 1` and Tika with `GET /version`; load schema explicitly from `sql/schema.sql` in setup instructions. Include one `DEEPSEEK_API_KEY` plus configurable DeepSeek generation/judge model IDs in `.env.example`; do not require the key for health.
- [x] **Step 4: Run `python -m pytest tests/test_health.py -q` and `docker compose config`.** Expected: all tests pass and Compose validates; verify the Tika helper is in `app/tika.py`.
- [x] **Step 5: Initialize Git on `main` if absent and create `feat/nexora-rag-lab`; show the changes and stop for the user's test/review checkpoint. Do not commit.**

### Task 2: Tika extraction, local BGE-M3 embeddings, and chunking strategies

**Files:** Create `app/embeddings.py`, `app/ingestion/__init__.py`, `app/ingestion/extractor.py`, and the `app/ingestion/chunking/` package; keep `app/ingestion/chunker.py` as a compatibility module. Test `tests/test_extractor.py`, `tests/test_chunker.py`, and `tests/test_embeddings.py`.

**Interfaces:** `extract_text(path: Path, tika_url: str) -> str` and `extract_bytes(filename: str, document_bytes: bytes, tika_url: str) -> str` use `PUT /tika/md`; `normalize_text(text: str) -> str`; `get_embedder(settings: Settings) -> SentenceTransformer` returns a process-cached CPU model; `get_tokenizer(settings: Settings) -> PreTrainedTokenizerBase`; `embed_passages(texts: list[str], settings: Settings) -> list[list[float]]`; `embed_query(text: str, settings: Settings) -> list[float]`; `ChunkSlice(order: int, text: str, token_count: int, start_char: int, end_char: int, fallback_reason: str | None)`; `ChunkingSettings(chunk_size: int, overlap: int, max_input_tokens: int = 1200)`; `ChunkingStrategy.split(text: str, tokenizer: PreTrainedTokenizerBase, settings: ChunkingSettings) -> list[ChunkSlice]`; explicit strategies `FixedTokenStrategy`, `RecursiveStrategy`, and `DocumentStructureStrategy`, selected by `get_chunking_strategy(strategy_id: str) -> ChunkingStrategy`.

- [x] **Step 1: Write failing tests.** Existing tests cover Tika extraction, Vietnamese fixed-token overlap, embedding dimensions, and invalid overlap. Add strategy tests for exact source spans, maximum token count, deterministic output, recursive separator preference, and heading/list/table preservation.
- [x] **Step 2: Run `python -m pytest tests/test_extractor.py tests/test_chunker.py tests/test_embeddings.py -q`.** Expected: failures for missing functions.
- [x] **Step 3: Implement extraction, BGE tokenizer chunking, and local embeddings.** Normalize line endings and Unicode NFC, collapse excessive whitespace without erasing paragraph separation. Load `SentenceTransformer("BAAI/bge-m3", revision="31e47391fcbda65be526abe98e646b3c6cd845a8", device="cpu")` lazily; set `max_seq_length=1200`; encode normalized dense float32 vectors in batches of 4. Use the same pinned tokenizer for chunks and verify model inputs do not exceed 1200 tokens.
- [x] **Step 3a: Add failing source-span and strategy tests.** Assert all returned spans slice the canonical text to the exact chunk text, chunk token counts stay within the configured maximum, and all canonical non-whitespace text is covered. Assert recursive splitting prefers paragraphs before sentences and document-structure splitting keeps headings with sections and intact tables/lists when they fit.
- [x] **Step 3b: Implement the strategy interface and three Phase 1 strategies.** Keep the existing `chunk_text` as a compatibility wrapper for `FixedTokenStrategy`. Use tokenizer offset mappings to preserve source spans. Use Tika Markdown as the shared canonical extraction, preserve headings/lists/tables, and recursively split oversized structural units. Record when document-structure chunking falls back because Markdown headings are absent or a section exceeds the chunk limit. Do not implement semantic or LLM strategies in this task.
- [ ] **Step 3c: Run focused tests and inspect Tika Markdown on representative PDF, DOCX, and TXT documents.** Verify the returned representation preserves useful structure and that normalization keeps offsets aligned. If the local Tika service is unavailable, leave the endpoint smoke check for the user's local run and report it explicitly.
  Preliminary DOCX check with `English_8_15min_Teenagers.docx` (outside the benchmark corpus): raw Tika Markdown preserves the text and tables but includes escaped pipes, an HTML space entity, and no `#` headings. The DOCX's custom `SectionHeading` style appears as `class="sectionHeading"` in Tika HTML, but as plain text in Tika Markdown. These artifacts are present before `normalize_text`. This DOCX causes `DocumentStructureStrategy` to use its `no_markdown_headings` fallback. A first pass over one benchmark file per format found three Markdown headings in a DOCX, none in a PDF or TXT, and escaped pipes in all three. Keep the full corpus audit open before changing the canonical extraction format.
- [ ] **Step 4: Run the three test files, then one real Vietnamese passage/query smoke encode.** Expected: tests pass and both actual vectors have 1024 finite values. Record first-download size, cold-load time, and CPU encode time; do not assume a GPU is usable.
  Automated tests for extraction and embeddings previously passed. The actual model download and smoke encode are pending in the user's local environment because the sandbox cannot reach PyPI/Hugging Face.
- [ ] **Step 5: Show the changes and stop for the user's test/review checkpoint. Do not commit.**

### Task 3: Atomic, manifest-driven ingestion

**Files:** Create `app/ingestion/pipeline.py`, `scripts/ingest.py`; modify `app/database.py`, `sql/schema.sql`; test `tests/test_ingestion.py`. Stage the benchmark files under `data/benchmark/` while preserving their internal paths.

**Interfaces:** `ingest_manifest(manifest_path: Path, profile_id: str, strategy_id: str, strategy_params: dict[str, object], settings: Settings, reset_corpus: bool = False) -> IngestReport`. Report includes corpus/extraction fingerprints, document/chunk totals, fallback counts, errors, and profile ID; CLI accepts `--manifest`, `--profile`, `--strategy`, JSON strategy settings, and explicit `--reset-corpus`. `profiles` records strategy ID/version/JSON parameters and extraction fingerprint; `chunks` stores canonical character offsets.

- [ ] **Step 1: Write failing tests.** `test_manifest_limits_corpus`: assert 24 IDs and zero `BoCauHoi/` paths; `test_profile_rerun_isolated`: assert B rows survive replacing A; `test_changed_corpus_requires_reset`, `test_embedding_revision_guard`, and `test_embedding_dimension_guard`: assert clear errors; `test_failed_ingest_is_atomic`: assert old rows remain after extract/embed failure.
- [ ] **Step 2: Run `python -m pytest tests/test_ingestion.py -q`.** Expected: failures for missing pipeline.
- [ ] **Step 3: Implement strategy-profile ingestion.** Stage only the listed ZIP members, preserving the original structure. Hash sorted `(document_id, relative_path, file_sha256)` tuples plus canonical extraction/version. Extract and batch-embed passages locally before the replacement transaction; validate 1024 finite values per vector; write documents/profile/chunks with strategy metadata and source offsets in one transaction. Save model revision, CPU/float32 mode, tokenizer, and 1200-token cap in profile metadata. For a changed corpus or extraction, require `--reset-corpus` and clear all profiles atomically. Generate stable chunk IDs from profile/document/ordinal/content hash. Print and save a JSON report including per-file failures, extraction quality, structural fallbacks, model load time, embed time, and chunks/second.
- [ ] **Step 4: Run `python -m pytest tests/test_ingestion.py -q`; then ingest fixed-token, recursive, and document-structure profiles against local containers and verify SQL counts and matching source fingerprints.** Expected: tests pass, 24 documents, positive chunk counts for each profile, no failed files.
- [ ] **Step 5: Show the changes and stop for the user's test/review checkpoint. Do not commit.**

### Task 4: Semantic retrieval and search API

**Files:** Create `app/rag/__init__.py`, `app/rag/retrieval.py`; modify `app/main.py`; test `tests/test_retrieval.py`, `tests/test_search_api.py`.

**Interfaces:** `search(question: str, board_id: str, profile_id: str, top_k: int, settings: Settings, query_vector: list[float] | None = None) -> list[RetrievedChunk]`; each chunk includes `chunk_id`, `document_id`, `board_id`, `text`, `score`, and `rank`. `POST /rag/search` accepts the four public inputs and returns ordered chunks; the optional vector exists only for internal evaluation reuse.

- [ ] **Step 1: Write failing tests.** `test_search_scopes_board_and_profile`: assert SQL uses `<=>` plus both predicates and returns ordered hits; `test_search_embedding_matches_profile`: assert local BGE model/revision/1024 width; `test_precomputed_query_vector`: assert supplied vector bypasses a second encode; `test_top_k_bounds`: assert values outside 1–10 fail; `test_missing_profile`: assert an actionable error.
- [ ] **Step 2: Run `python -m pytest tests/test_retrieval.py tests/test_search_api.py -q`.** Expected: failures for missing retrieval/API.
- [ ] **Step 3: Implement exact pgvector search and request validation.** Use `embed_query` from `app/embeddings.py` when no vector is supplied; validate an internal precomputed vector is finite and 1024 elements. Keep the Board value an explicit experiment partition parameter; document that it is caller supplied and offers no authorization.
- [ ] **Step 4: Run the two test files and one Swagger search against the fixed-token profile.** Expected: passing tests and retrieved chunks tagged with manifest document IDs.
- [ ] **Step 5: Show the changes and stop for the user's test/review checkpoint. Do not commit.**

### Task 5: Answering and verified citations

**Files:** Create `app/rag/generation.py`, `app/rag/pipeline.py`; modify `app/main.py`; test `tests/test_ask.py`.

**Interfaces:** `request_deepseek_json(messages: list[dict[str, str]], model: str, settings: Settings) -> DeepSeekResponse` performs the shared raw HTTP call and returns parsed `payload`, `model`, `system_fingerprint`, and `usage`; `generate_answer(question: str, chunks: list[RetrievedChunk], settings: Settings) -> GeneratedAnswer` with `answer`, `source_labels`, `abstained`; `ask(question: str, board_id: str, profile_id: str, top_k: int, settings: Settings, query_vector: list[float] | None = None) -> AskResult` with `answer`, `citations`, `abstained`, `retrieved_chunks`. `POST /rag/ask` uses the same public request fields as search.

- [ ] **Step 1: Write failing tests.** `test_no_hits_abstains`: assert zero DeepSeek calls; `test_citation_mapping`: assert `S1` maps to the retrieved IDs; `test_invalid_label_abstains`: assert no fabricated citation; `test_malformed_json_abstains`: assert invalid/truncated JSON cannot produce a claimed answer; `test_source_text_is_untrusted`: assert an injection string stays inside the evidence message, not system instructions.
- [ ] **Step 2: Run `python -m pytest tests/test_ask.py -q`.** Expected: failures for missing generation/pipeline.
- [ ] **Step 3: Implement a fixed grounded prompt and validated DeepSeek JSON output.** Use `httpx` to POST to `https://api.deepseek.com/chat/completions` with bearer `DEEPSEEK_API_KEY`, model `deepseek-flash`, `thinking={"type":"disabled"}`, and `response_format={"type":"json_object"}`. Keep the HTTP/JSON helper reusable by the evaluator. Instruct JSON in the prompt, reject truncated/malformed responses, mark retrieved text as untrusted data, include the benchmark reference date, distinguish proposals from later explicit decisions/status, require cited source labels, and verify labels against the retrieved set in code. Return citation snippets copied from stored chunks, never model-authored source text.
- [ ] **Step 4: Run `python -m pytest tests/test_ask.py -q` and manually ask one answerable plus one unanswerable smoke case through Swagger.** Expected: valid citation or abstention, respectively.
- [ ] **Step 5: Show the changes and stop for the user's test/review checkpoint. Do not commit.**

### Task 6: Benchmark loader and deterministic scores

**Files:** Create `scripts/evaluate.py`; test `tests/test_evaluate.py`.

**Interfaces:** `load_cases(path: Path, split: str, include_security: bool = False) -> list[BenchmarkCase]`; `score_case(case: BenchmarkCase, result: AskResult) -> dict[str, float | bool | None]`; CLI accepts `--cases`, `--split`, `--configs`, `--dry-run`, and `--output-dir`.

- [ ] **Step 1: Write failing tests.** `test_benchmark_counts`: assert 300 unique IDs, 60/120/180 split sizes, 27 SECURITY exclusions, and 51/111/162 primary sizes; `test_multidoc_recall`: assert `recall == 1/3` and `all_gold is False` for one of three hits; `test_abstention_score`: assert expected ABSTAIN is scored; `test_bad_cases`: assert empty selection/duplicate IDs fail.
- [ ] **Step 2: Run `python -m pytest tests/test_evaluate.py -q`.** Expected: failures for missing loader/scorer.
- [ ] **Step 3: Implement loader, validation, and local scoring.** Derive cumulative regression from split membership or verify the supplied 120-file IDs; preserve `scenario_id` and category for grouped reports. Report DocumentRecall@K, any-hit, all-gold, citation validity, abstention accuracy, latency, and usage when available. Treat `evidence_quotes` as advisory and do not claim chunk recall.
- [ ] **Step 4: Run `python -m pytest tests/test_evaluate.py -q` and `python scripts/evaluate.py --dry-run --split smoke`.** Expected: tests pass; dry run lists 51 scored and 9 skipped SECURITY cases without model calls.
- [ ] **Step 5: Show the changes and stop for the user's test/review checkpoint. Do not commit.**

### Task 7: Langfuse experiments and paired comparison

**Files:** Modify `scripts/evaluate.py`, `app/rag/pipeline.py`; create `experiments/README.md`; test `tests/test_langfuse_eval.py`.

**Interfaces:** Evaluation CLI syncs selected cases to a versioned Langfuse dataset, precomputes one BGE-M3 vector per unique question, runs each strategy with the current SDK experiment runner at the fixed Phase 1 settings, attaches scores and profile/run metadata, and writes `data/results/<run-id>/` JSONL plus `summary.md`.

- [ ] **Step 1: Write failing tests with a fake Langfuse client.** `test_strategies_use_same_cases`: assert identical case IDs, corpus/extraction fingerprint, `top_k=3`, and BGE settings for fixed-token, recursive, and document-structure profiles; `test_query_vectors_reused`: assert one local encode per unique question and the same vector supplied to every strategy; `test_skips_security`: assert 27 skip records; `test_trace_metadata`: assert strategy/profile metadata, BGE revision/device/precision, DeepSeek response model/mode, and prompt/rubric/dataset versions; `test_generation_input_excludes_gold`: assert no expected answer or gold ID reaches `ask`.
- [ ] **Step 2: Run `python -m pytest tests/test_langfuse_eval.py -q`.** Expected: failures for missing run integration.
- [ ] **Step 3: Implement current Langfuse SDK v5 dataset/experiment calls and tracing.** Precompute query vectors once before timing strategies, then separate query-embedding, SQL retrieval, DeepSeek generation, and DeepSeek judge times/costs. Flush traces at script exit. Produce per-category aggregates and a paired per-question strategy-delta table; show the worst regressions with retrieved document IDs and citation IDs. Use `request_deepseek_json` with fixed `deepseek-v4-pro` and a fixed rubric for correctness/groundedness in formal regression/holdout runs; save raw judgments and allow smoke/dry runs to skip them. Treat judge grades as secondary to deterministic scores and manual audit. Present quality, latency, and cost separately without a weighted winner.
- [ ] **Step 4: Run tests and one smoke experiment.** Expected: 51 principal RAG cases for the selected strategy, linked traces/scores visible in Langfuse, and a local summary with run IDs. Then run regression; run holdout once after settings are frozen.
- [ ] **Step 5: Show the changes and stop for the user's test/review checkpoint. Do not commit.**

### Task 8: Reproducible handoff and final acceptance

**Files:** Create `README.md`; modify `.env.example`, `experiments/README.md`, and any task files needed by acceptance findings; test only where a real finding requires it.

**Interfaces:** README provides commands to start dependencies, load schema, ingest each Phase 1 strategy profile, serve Swagger, run smoke/regression/holdout, and read Langfuse plus local reports.

- [ ] **Step 1: Write the README from a clean checkout.** Include ZIP provenance/checksum, exact staging location, BGE-M3's one-time model download and CPU-first setup, the single `DEEPSEEK_API_KEY`, expected 24-document count, Phase 1 strategy matrix, corpus reset, cost-bearing DeepSeek generation/judging steps, and the SECURITY exclusion.
- [ ] **Step 2: Follow the README on a clean local database.** Start containers, apply schema, ingest the three Phase 1 profiles, run `GET /health`, one search, one ask, then the dry-run evaluator.
- [ ] **Step 3: Run `python -m pytest -q` and `docker compose config`.** Expected: all automated tests pass and Compose validates; fix any finding before proceeding.
- [ ] **Step 4: Run paired regression and holdout evaluations with fixed models, prompt, judge rubric, corpus, and dataset version.** Expected: local summaries plus Langfuse experiment links, per-category metrics, and at least three explained cases where the Phase 1 strategies differ. Manually audit at least ten low-scoring or disputed judge outputs against cited text. State that the holdout is workflow-reserved, not blind, because labels ship in the ZIP. If API credentials are unavailable, record that external acceptance remains pending rather than claiming completion.
- [ ] **Step 5: Review the full diff and commit all project work once.** Suggested message: `feat: build nexora rag experiment lab`.

## Execution order and gates

Milestone 1 = Task 1; Milestone 2 = Tasks 2–3; Milestone 3 = Tasks 4–5; Milestone 4 = Tasks 6–8. Stop after every task for user testing and review. Commit the completed project only after Task 8. The holdout set is evaluated only after prompts, models, judge rubric, and Phase 1 strategy settings are frozen. The benchmark's SECURITY cases stay visible as skipped records and never enter primary scores.
