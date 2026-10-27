# Plain-Text Extraction and Two Chunkers

> Superseded: LLM chunking is out of scope. The current implementation uses fixed-token chunking only; see [the current RAG flow design](2026-10-27-rag-flow-design.md).

**Decision:** Use Tika plain text as the canonical document representation. Compare fixed-token chunking with DeepSeek-based LLM chunking. This decision replaces the extraction and chunking choices in the earlier chunking strategy design.

## Intent

Keep the lab small enough to understand each step. Every strategy must receive the same saved, normalized plain text for a document. Keep BGE-M3 as the only embedding model and DeepSeek as the only remote LLM.

## Extraction

Use `PUT /tika/text` for PDF, DOCX, and TXT. Preserve Unicode, line breaks, and paragraph breaks where Tika provides them. Return the result as `text` in the Swagger extraction endpoint. Do not infer Markdown headings or repair document layout with format-specific rules. Record the extraction format and version when ingestion is implemented; changing it requires reingestion.

## Chunking

- `fixed_token` remains the deterministic baseline. Its size and overlap are measured in BGE tokens. Each chunk retains exact start and end character positions in the canonical plain text.
- `llm` will ask DeepSeek for meaningful boundaries in the same plain text. Its proposed boundaries must be validated against the original text and BGE input limit before any chunk is accepted. Record model, prompt version, input/output, latency, and failures for evaluation.
- Remove `recursive` and `document_structure` from the code and Swagger choices. Do not keep compatibility aliases for them.

The LLM chunker is a separate implementation task. Until it is implemented, Swagger offers only `fixed_token`; it must not advertise a nonfunctional LLM option.

## Comparison

Use the same corpus, canonical extraction, BGE revision, query set, and retrieval settings for both chunkers. Report extraction loss, chunk coverage, chunk sizes, LLM cost/time, retrieval metrics, and case-level answer quality. Plain text can lose layout information, so citation spans refer to the saved plain text rather than the source file's visual pages.
