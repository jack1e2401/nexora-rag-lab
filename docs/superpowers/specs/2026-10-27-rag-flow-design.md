# RAG flow design

This lab completes the existing plain-text, fixed-token, local BGE-M3 path. The only chunking strategy used for ingestion and experiments is `fixed_token`. DeepSeek generates answers; no OpenAI service or model is used.

The supplied benchmark manifest defines the 24-file corpus. The question JSONL defines 300 evaluation cases. The 27 `SECURITY` cases are excluded from the primary RAG score because this lab has no authentication or authorization. `board_id` is still a required retrieval partition, but it is supplied by the caller and is not an access control mechanism.

Ingestion runs synchronously from a script. It extracts every listed file with Tika, chunks with the pinned BGE tokenizer, embeds with local BGE-M3, and replaces one profile atomically in PostgreSQL. A corpus change requires an explicit reset because profiles must share the same source text. Store the model revision, chunk settings, and source character offsets for reproducibility.

`/rag/search` embeds the question with the same model and performs exact cosine search within one profile and Board. `/rag/ask` sends numbered retrieved passages to DeepSeek. The model returns JSON with source labels; code accepts only labels belonging to the retrieved passages and constructs citations from stored source data. Empty or invalid evidence yields an abstention.

Evaluation compares two fixed-token profiles on the same questions and top-K. It reports retrieval and citation/abstention metrics, per-case results, paired differences, and traces/scores in Langfuse when configured. Reference answers stay out of generation input. DeepSeek calls are explicit, cost-bearing experiment steps. The holdout split is workflow-reserved, though its labels are available in the supplied ZIP.

The code stays in small modules with direct function calls, no workers, no HTTP ingestion endpoint, no RBAC, and no generic adapter framework for one strategy.
