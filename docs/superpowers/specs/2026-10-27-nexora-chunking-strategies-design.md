# Nexora RAG Lab — Chunking Strategy Comparison Design

**Date:** 2026-10-27
**Status:** Approved for implementation
**Relation:** This document updates the chunking and evaluation decisions in the [original RAG Lab design](2026-10-26-nexora-rag-lab-design.md). Its other stack and scope decisions still apply.

## Intent

Use the lab to choose a document chunking strategy for Nexora's project-management corpus. The comparison must show which questions each strategy helps or hurts, and why. The main Nexora application may later reuse tested algorithms and findings; its integration will be designed separately.

Use one embedding model: local BGE-M3 with 1024-dimensional dense vectors. Every document chunk and query uses the same model and revision. DeepSeek is the only API for answer generation and judging. Keep the small FastAPI app, Tika, PostgreSQL/pgvector, Langfuse, and synchronous scripts. Do not add a model adapter, queue, worker, RBAC, frontend, or production deployment.

## Approaches considered

| Approach | Benefit | Tradeoff |
| --- | --- | --- |
| Build all five strategies at once | Every candidate is available immediately | Extraction, algorithm, and API failures are hard to separate; semantic and LLM chunking add time and cost before a trustworthy baseline exists |
| Use only Tika 4.1's chunker | Can reuse Tika's `tk:chunks` and locators | `tika-inference` remains experimental; its `MarkdownChunker` uses character limits, while this lab needs BGE-token limits and a shared pgvector pipeline |
| **Stage three strategies, then consider two more** | Compare deterministic methods first and add costly methods only when results justify them | Phase 1 cannot conclude anything about semantic or LLM chunking |

The user chose the **3 + 2** approach: compare fixed-token, recursive separator-aware, and document-structure-aware chunking first. Consider semantic and LLM chunking after Phase 1 if its results show a reason to try them. The linked [article describing five strategies](https://viblo.asia/p/toi-uu-hoa-rag-kham-pha-5-chien-luoc-chunking-hieu-qua-ban-can-biet-EvbLbPGW4nk) supplies a taxonomy, not evidence that one strategy is best for this corpus.

## Shared data contract

The pipeline remains `Tika → extracted representation → strategy → BGE-M3 → pgvector`. Extract each document once for a given corpus/extraction version, then pass the **same canonical extracted representation** to every strategy. Unicode and line-ending normalization must match. Store the source-file hash, canonical-text hash, extraction format, and extractor version so profiles built from different inputs cannot be compared silently.

The current extractor calls `PUT /tika/text` and receives plain text. This may lose headings, lists, and table structure. Before implementing document-structure chunking, audit Tika 4.1 Markdown/metadata output on this corpus's PDF, DOCX, and TXT files. If the canonical representation changes, **reingest every profile**, including fixed-token, from that representation. If Markdown does not preserve useful structure for a file, apply a paragraph-aware fallback on the shared representation and record the affected file and fallback. No strategy may use richer extraction than the others within one comparison.

A strategy takes normalized text and the BGE tokenizer and returns ordered chunks. Keep the interface small: `split(text, tokenizer, settings) -> list[ChunkSlice]`, either as a function or a class method if state is needed. Select by explicit ID (`fixed_token`, `recursive`, `document_structure`); do not add a dynamic plugin loader. Each `ChunkSlice` records `ordinal`, `text`, `token_count`, `start_offset`, and `end_offset` in **character offsets of the saved canonical representation**. Its text must be traceable to that source span. Chunks may overlap, but meaningful canonical text must remain covered. If a unit exceeds the limit, split it further with a checked token fallback; never silently truncate it.

Every chunk, including special tokens, must fit the configured BGE input limit (currently 1200 tokens). Tests must cover Vietnamese text, punctuation, whitespace, and tables; tokenizer decode output must not be assumed to preserve character offsets. Citations retain `document_id`, `chunk_id`, and source span in the saved canonical representation. A valid span proves provenance; separate checks or judging assess whether it supports the answer.

## Strategies

| Phase | Strategy | Main rule | Initial settings |
| --- | --- | --- | --- |
| 1 | Fixed-token | Existing BGE-token window with fixed overlap | 500 tokens, overlap 50 |
| 1 | Recursive | Prefer section, paragraph, then sentence boundaries; fall back to words and tokens for oversized units | About 500 tokens maximum; overlap neighboring units and measure actual overlap in tokens |
| 1 | Document-structure | Keep a heading with its section; avoid splitting lists or tables when the representation supports them; use recursive splitting for oversized sections | About 500 tokens maximum; report structural fallbacks |
| 2 | Semantic | Embed sentence/paragraph units with the same BGE-M3 and split/group by similarity subject to the token limit | Lock the threshold or split rule before holdout; report extra ingest time |
| 2 | LLM | Ask DeepSeek to propose boundaries between units with source offsets; validate spans and token limits in Python | Run only on a shortlist; report tokens, cost, and API failures |

The 500/50 baseline is an experimental setting, not a proven optimum. Recursive and structure-aware overlap follows unit boundaries; report its actual token overlap rather than treating a configured value of 50 as equivalent across strategies. Phase 2 is not automatically part of Phase 1. After reviewing the first three strategies, decide whether semantic or LLM methods are warranted, identify the failure cases to investigate, and freeze their rules before holdout.

## Profiles, schema, and ingestion

A profile records `strategy_id`, `strategy_version`, `strategy_params`, extraction and corpus fingerprints, and BGE model/revision/dimensions/device/precision/input cap. Changing the strategy or its settings creates a separate chunk profile. One profile can still be queried at several `top_k` values. Keep `VECTOR(1024)` because the embedding model does not change.

The existing `profiles(chunk_size, overlap)` schema describes only fixed-token chunking. Implementation will replace those strategy-specific columns with explicit strategy metadata, such as `strategy_params JSONB`, and add source offsets to `chunks`. Since the lab has not ingested a corpus yet, a documented local schema reset is acceptable. Never silently delete a user's existing profiles.

Ingestion stays synchronous per profile: validate settings, chunk and embed all 24 documents, then commit a complete profile. An error must not leave a half-written profile. Report document and chunk counts, actual chunk-size distribution, fallbacks, and per-document errors. Chunk identity and order must be deterministic for the same input and profile.

## Fair comparison

Hold the 24 documents, canonical extraction, BGE revision/device/precision, query embedding, Board filter, exact cosine retrieval, `top_k=3`, DeepSeek model/mode/prompt, and benchmark version fixed. Precompute each question vector once and reuse it for every profile. Move the previous plan's `top_k=5` and fixed-token 1000/100 profiles to **later parameter tuning**; they do not belong to the primary strategy comparison.

1. Use smoke cases to check the pipeline and evidence alignment.
2. Run retrieval-only comparison for all three Phase 1 strategies on regression cases, excluding `SECURITY`. Choose finalists using paired case-level results and ingest cost, not just a single average.
3. Run DeepSeek generation and judging for finalists with the same prompt and context budget. Store Langfuse traces and scores plus local artifacts containing profile, case ID, and raw outputs.
4. Freeze profiles and rubric before one final holdout run. The holdout labels ship in the ZIP, so this is a **workflow holdout**, not a blind dataset.

`expected_document_ids` support DocumentRecall@K, not chunk recall. Benchmark `evidence_quotes` are dictionaries of `{document_id: text}`; a value can contain snippets joined by ` | `. The 27 `SECURITY` cases have no evidence quote. Before computing EvidenceHit/Coverage@K, split snippets, normalize them by a fixed rule, and align them to the canonical extracted representation. Report **snippet and case alignment coverage** and sample alignment failures. Compute evidence metrics only on reliably aligned cases; do not count an unaligned quote as a retrieval miss. Exclude the 27 `SECURITY` cases from the primary RAG score as agreed.

At minimum, report DocumentRecall@3 (any-hit and all-gold-document coverage), EvidenceHit/Coverage@3 on aligned cases, alignment rate, citation validity, answer correctness/groundedness or abstention for finalists, chunk counts, token-size distribution, total context tokens, ingest time, query-embedding time, SQL retrieval latency, and DeepSeek latency and token/cost when available. Compare the same questions pairwise and show improvements and regressions by DIRECT, MULTIHOP, CONFLICT, NUMERIC, UNANSWERABLE, INJECTION, and TEMPORAL category. Do not collapse these into a single universal winner.

## Acceptance and limits

- The three Phase 1 strategies run on one canonical extraction and corpus fingerprint. Every chunk fits BGE's input limit and has a valid source span.
- Meaningful canonical text remains covered across a document's chunks. Every fallback is recorded.
- A case-level retrieval report includes evidence-alignment coverage and latency/ingest cost. Langfuse stores traces and scores for finalist generation runs.
- Concrete cases explain where a strategy improved or regressed on this corpus. The result does not claim a universally optimal strategy.
- Access control is outside scope; Board filtering only partitions this benchmark corpus.

## Decisions left to the implementer

Choose a function or class for the small interface, a simple tested Vietnamese sentence-splitting rule, profile naming, table representation in canonical text, and local artifact format. These choices must preserve traceable source offsets and common input across profiles. If the Tika audit shows unstable Markdown output, record the result and use the paragraph-aware fallback above.

## References

- [Five chunking strategies on Viblo](https://viblo.asia/p/toi-uu-hoa-rag-kham-pha-5-chien-luoc-chunking-hieu-qua-ban-can-biet-EvbLbPGW4nk)
- [Apache Tika 4.1: chunk strategies for search engines](https://tika.apache.org/docs/4.1.x/migration-to-4x/chunk-strategies.html)
- [Apache Tika 4.1: inference engines and bindings](https://tika.apache.org/docs/4.1.x/configuration/inference.html)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
