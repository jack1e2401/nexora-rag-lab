# Nexora RAG Lab — Design

**Date:** 2026-10-26
**Status:** implementation planning baseline

**Chunking update:** The approved [2026-10-27 chunking strategy comparison design](2026-10-27-nexora-chunking-strategies-design.md) supersedes this document's A/B/C experiment matrix and chunking/evaluation contract. The other stack and scope decisions still apply.

## Intent

Build a small, inspectable Python lab for learning and measuring retrieval-augmented generation on a real document corpus. A successful run explains, with item-level evidence, why a chunking or retrieval configuration did better or worse on the same questions.

## Scope

Three independent entry points:

1. `python scripts/ingest.py`: synchronously extract, chunk, embed, and store a local corpus.
2. FastAPI `GET /health`, `POST /rag/search`, and `POST /rag/ask`: inspect retrieval and ask grounded questions through Swagger.
3. `python scripts/evaluate.py`: run a fixed question set against configurations and save local results plus Langfuse experiment traces and scores.

Use Apache Tika 4.1.0, local BAAI/bge-m3 dense embeddings, DeepSeek API for both generation and judging, PostgreSQL 17 with pgvector, and Langfuse Cloud by default. Call DeepSeek directly with `httpx`; do not use an OpenAI account, API, or SDK. Target Python 3.11 or newer and Langfuse Python SDK v5. Keep Redis, workers, upload APIs, authentication, frontend, Nexora integration, and production deployment outside this repo.

## Approach and decisions

The user selected a synchronous lab with one FastAPI process and two scripts. A production service with queued ingestion would obscure the RAG experiment, and an in-memory vector store would omit the requested pgvector learning objective.

- Keep the proposed small `app/ingestion` and `app/rag` structure. Add no generic repository/service layer.
- Start with exact pgvector cosine search. An approximate index adds a retrieval variable that the first experiments do not intend to test.
- Pin one local embedding model and vector width per database: `BAAI/bge-m3` at Hugging Face revision `31e47391fcbda65be526abe98e646b3c6cd845a8`, dense vectors, 1024 dimensions. Use the model-card-supported `sentence-transformers` loader. Reject a model, revision, or dimension mismatch with an actionable error. Changing the embedding model or width requires a fresh schema/corpus build.
- Put BGE-M3 loading, its tokenizer, passage embedding, and query embedding in one small `app/embeddings.py` module because both pipelines must use the same model and settings. Load once per Python process, lazily; keep the model cache outside Git. Use CPU float32 with normalized vectors as the comparable baseline, batch size 4, and a fixed model input cap of 1200 tokens. GPU is an explicit later option; record device/precision/model revision.
- Measure chunk size and overlap with the BGE-M3 tokenizer, not tiktoken. Check that chunks plus model special tokens fit below the 1200-token model input cap so B's 1000-token chunks are not silently truncated. Use dense vectors only; sparse and ColBERT modes are outside the first experiment. A and B produce separate persisted chunk profiles; C reuses A's chunks and changes only `top_k`.
- Set initial generation model to DeepSeek `deepseek-flash` and judge model to DeepSeek `deepseek-v4-pro`, both in non-thinking mode via direct HTTPS requests. Request JSON output, validate it in Python, and record the returned model/fingerprint if available. Keep the two model settings fixed across A/B/C.
- Use the supplied `Nexora_RAG_Benchmark_BoCauHoi.zip` as the v1 benchmark source: 24 documents (DOCX, PDF, TXT) in MERCURY, AURORA, and UNITECH; 300 cases with `smoke_60`, cumulative `regression_120`, and `holdout_180` subsets. Stage only `01_CORPUS/` documents for ingestion; preserve manifest and question files as evaluation data, never retrievable context.
- Keep the original `questions_300.jsonl` fields rather than inventing a new case format. Sync selected cases to a versioned Langfuse dataset before experiment runs. Use smoke for wiring, regression for iteration, and holdout for final comparison; the 105 scenario groups do not cross these splits.
- Keep expected answers, rationales, and evaluation metadata out of generation inputs. The holdout is a reserved workflow split, not a blind set: its labels are present in the supplied ZIP, so final conclusions must state that limit.
- Use Langfuse Cloud so this lab needs no extra observability containers. Support a configurable Langfuse base URL if an existing compatible instance is available.
- A local corpus has one fingerprint derived from sorted relative source paths and file hashes. Re-ingestion of the same corpus replaces only the selected profile. A changed corpus requires an explicit reset and rebuild of all profiles; never compare profiles from different corpora.
- `document_id` is the ID in `manifest.json` (for example `M-PLAN-02`); also store its Board and relative path. Match `expected_document_ids` against retrieved chunks' document IDs. `evidence_quotes` are advisory snippets, sometimes joined with ` | `, so they are unsuitable as an exact full-string chunk ground truth. Inspect/normalize individual snippets before any evidence coverage metric.
- Use the case's Board as an explicit retrieval partition for experiments. This is corpus scoping, not authentication or a security boundary. Exclude the 27 `SECURITY` cases from the principal RAG score, as the user requested. Record them as skipped with a clear reason; do not claim access-control results from a caller-supplied Board filter.
- Return citations only for chunks actually retrieved. The answer model refers to temporary labels such as `S1`; code maps these to stored chunk IDs and source IDs and verifies them. This proves citation provenance, not that every factual claim is supported; groundedness is scored separately.
- Fix the generation model and mode, prompt version, embedding model/revision/device/precision, corpus fingerprint, dataset version, and judge rubric across compared runs. Record all in run metadata.
- Tell generation to treat source documents as evidence rather than instructions and to distinguish proposals from later explicit decisions/status. When sources conflict without a clear resolution, state the uncertainty and cite the conflicting sources.

## Experiment matrix

| Config | Persisted profile | Chunk size (tokens) | Overlap (tokens) | Top-K |
| --- | --- | ---: | ---: | ---: |
| A — baseline | A | 500 | 50 | 3 |
| B — larger chunks | B | 1000 | 100 | 3 |
| C — more retrieval | A | 500 | 50 | 5 |

These values are hypotheses to test, not presumed optima.

## Data and evaluation contract

Each supplied JSONL item includes `id`, `question`, `board_id`, `expected_answer`, `expected_behavior`, `expected_document_ids`, `evidence_quotes`, `category`, `split`, and `scenario_id`. The benchmark includes 237 `ANSWER`, 36 `ABSTAIN`, and 27 `REFUSE_UNAUTHORIZED` cases. Keep the benchmark's fixed reference date of 2026-10-26; future dates in questions refer to this synthetic snapshot.

The script reports DocumentRecall@K (with both any-hit and all-gold-document coverage for multi-document cases), citation validity, abstention accuracy, latency, and token usage/cost when available. It must not label `expected_document_ids` as chunk recall. A fixed DeepSeek judge grades answer correctness and groundedness against the expected answer and cited context for formal regression/holdout comparisons; dry runs and smoke checks can skip judging. Save the rubric and raw judgments for audit, and manually inspect a sample of disputed or low-scoring judgments. Since generation and judging use the same provider, judge grades are secondary evidence beside deterministic metrics and manual review. Aggregate metrics are paired with per-question deltas and examples of regressions. The synthetic dataset supports comparison inside this corpus, not a universal ranking of configurations.

Report local model cold-load time, CPU ingest time, chunks/second, and query-embedding time separately from retrieval SQL and DeepSeek generation latency. Precompute each unique benchmark question embedding once before timed A/B/C comparison; pass the same vector to every configuration. Do not silently mix CPU and GPU results in the same comparison. The current host reports 15.4 GiB RAM and a GTX 1650 with 4 GiB VRAM, but CPU execution remains the baseline and CUDA usability is unverified.

Do not collapse these metrics into one weighted score or announce a universal winner. Compare A/B/C as a vector of quality, latency, and cost tradeoffs; a conclusion names the metric and case category for which a configuration improved or regressed.

## Acceptance

- `GET /health` confirms both PostgreSQL and Tika are reachable.
- Ingestion reports document/chunk counts and per-file errors, and commits no partial profile on a failed run.
- Ingestion stores 1024-element vectors from BGE-M3. Query embedding comes from the same pinned BGE-M3 revision and numerical mode; mismatches fail before retrieval.
- Search returns ordered chunks with source IDs and scores. Ask returns an answer with verified citations or an explicit insufficient-information response.
- A, B, and C run against one corpus fingerprint and one question-set version; Langfuse shows linked traces/scores and local comparison artifacts explain the differences.

## Remaining input

The benchmark ZIP is at `C:/Users/ACER/Downloads/Nexora_RAG_Benchmark_BoCauHoi.zip` (SHA-256 `9820F73D97E46856DB8D5940A67C404557E435A55C53D254FB34F4D137CE8A37`). Keep it accessible or stage its contents when implementing. The principal RAG subsets contain 51 smoke, 111 cumulative regression, and 162 holdout cases after excluding `SECURITY`.

## Primary references checked on 2026-10-26

- [pgvector exact and approximate search](https://github.com/pgvector/pgvector)
- [Apache Tika 4 server endpoint behavior](https://tika.apache.org/docs/4.0.x/using-tika/server/index.html)
- [BGE-M3 model dimensions and usage](https://huggingface.co/BAAI/bge-m3)
- [Pinned BGE-M3 model snapshot](https://huggingface.co/BAAI/bge-m3/tree/31e47391fcbda65be526abe98e646b3c6cd845a8)
- [Sentence Transformers model revision and encoding options](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [DeepSeek API quick start](https://api-docs.deepseek.com/quick_start/agent_integrations/)
- [DeepSeek Chat Completions JSON output](https://api-docs.deepseek.com/api/create-chat-completion/)
- [DeepSeek thinking-mode control](https://api-docs.deepseek.com/guides/thinking_mode/)
- [Langfuse SDK compatibility](https://langfuse.com/docs/compatibility)
- [Langfuse experiments via SDK](https://langfuse.com/docs/evaluation/experiments/experiments-via-sdk)
