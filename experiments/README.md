# RAG experiments

The evaluator compares `fixed_500_50` and `fixed_1000_100` on the same selected benchmark cases. Both profiles must have the same corpus fingerprint and BGE-M3 revision. Query embeddings are created once per distinct question and reused across profiles. Top-K is shared across both runs.

The primary deterministic metrics are document recall, any/all expected document retrieval, citation validity/precision, and abstention accuracy. Document recall is measured by expected document IDs; it is not chunk recall. `--judge` adds DeepSeek correctness and groundedness scores after answer generation. Human review is still needed to interpret borderline answers.

Langfuse credentials are optional. With `--langfuse`, the Python SDK runs each profile as a local dataset experiment and records item traces and deterministic scores. The experiment input contains only case ID, question, and Board ID. Expected answer and expected document IDs are attached as evaluation output; they are never used to build the answer-generation request. DeepSeek judge requests receive the reference answer only after generation.

Start with `python -m scripts.evaluate --split smoke --mode retrieval --run --limit 2`. Add `--langfuse` to inspect traces in Langfuse. Answer generation and judging call DeepSeek and may incur API charges; set `DEEPSEEK_API_KEY` before using `--mode ask` or `--judge`.

Each run writes one JSONL file per profile and `summary.json` beneath `data/results/<run-id>/`. Keep the corpus, BGE revision, profile sizes, top-K, split, generation model, and prompt stable when comparing results. The 27 SECURITY cases remain outside the primary score. The benchmark is synthetic and its gold labels are included in the supplied ZIP, including the workflow-reserved holdout.
