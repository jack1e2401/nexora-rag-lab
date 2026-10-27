# Nexora RAG Lab

A small experiment lab for seeing and measuring a complete RAG workflow. It uses Apache Tika for plain-text extraction, fixed-token chunking, local BGE-M3 embeddings, PostgreSQL with pgvector, DeepSeek for answer generation, and Langfuse for experiment traces and scores.

## What is implemented

| Step | How to try it |
| --- | --- |
| Extract | Swagger `POST /ingestion/extract` or synchronous corpus ingestion |
| Tokenize and chunk | Swagger `POST /tokenization/inspect` and `POST /chunking/split` |
| Embed | Swagger `POST /embeddings/embed` or corpus ingestion |
| Store | `python -m scripts.ingest` writes documents, fixed-token chunks, and 1024-value vectors |
| Search | Swagger `POST /rag/search` |
| Answer | Swagger `POST /rag/ask`, with citation labels checked against retrieved chunks |
| Evaluate | `python -m scripts.evaluate` compares two fixed-token profiles on the same benchmark questions |

Only `fixed_token` chunking is used. The experiment profiles are `fixed_500_50` and `fixed_1000_100`, both use BGE-M3 and Top-K 3 by default. There is no LLM chunking path.

The provided corpus manifest selects 24 documents. The 300-case question set is stored under `data/benchmark/BoCauHoi/`. SECURITY cases are excluded from primary scores because this lab does not implement access control. `board_id` is a required retrieval partition supplied by the caller; it does not authorize access.

## Setup on Windows

Run these commands in PowerShell from the repository root:

```powershell
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d
```

Set `DEEPSEEK_API_KEY` in `.env` before using `/rag/ask` or an ask evaluation. Add `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` if you want experiments uploaded to Langfuse. `LANGFUSE_BASE_URL` defaults to `https://cloud.langfuse.com`.

Start the API with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/health` for `{"database":"ok","tika":"ok"}`. Open `http://127.0.0.1:8000/docs` to use Swagger. Model weights download to the host the first time BGE-M3 is loaded; Docker runs PostgreSQL and Tika.

## Ingest the two profiles

Ingestion applies `sql/schema.sql` automatically. The first profile initializes the corpus and the second profile adds an alternate chunk size:

```powershell
.\.venv\Scripts\python.exe -m scripts.ingest --profile fixed_500_50 --chunk-size 500 --overlap 50 --reset-corpus
.\.venv\Scripts\python.exe -m scripts.ingest --profile fixed_1000_100 --chunk-size 1000 --overlap 100
```

Each run prints the corpus fingerprint and document/chunk totals. Ingestion extracts with Tika, chunks in BGE tokens, embeds locally, then stores a complete profile in one database transaction. If a later run detects changed source files, use `--reset-corpus` to replace all profiles together. Reingesting an unchanged corpus into one profile replaces only that profile.

`--reset-corpus` deletes every stored profile, its chunks, and the current documents before saving the selected corpus. Use it only when intentionally switching the corpus.

## Try search and ask

Example `/rag/search` or `/rag/ask` body:

```json
{
  "question": "Who is responsible for payment integration?",
  "board_id": "board-mercury",
  "profile_id": "fixed_500_50",
  "top_k": 3
}
```

`/rag/search` returns ranked chunks, similarity scores, and source character positions. `/rag/ask` returns an answer, the exact retrieved chunks, and citations copied from stored source text. If there are no results or DeepSeek returns an invalid source label, the pipeline abstains.

## Evaluate both profiles

See the selected case counts without loading BGE or calling any API:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --split smoke
```

Run retrieval-only comparison (local embedding and database required, no DeepSeek call):

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --split smoke --mode retrieval --run
```

Run generation comparison and optionally judge answers with DeepSeek:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate --split smoke --mode ask --run --judge
```

To send local-dataset experiment traces and item scores to Langfuse, add `--langfuse`. Try a paid/API smoke run with `--limit 2` first. Each case runs once under each profile. The report contains one JSONL file per profile and `summary.json` with averages plus per-question retrieval differences under `data/results/`.

Splits are cumulative for the first two stages: `smoke` selects 51 cases, `regression` selects the first 120 cases (111 after SECURITY exclusion), and `holdout` selects 162 cases after exclusion. The supplied answer labels are included in the ZIP, so holdout is workflow-reserved rather than blind. Evaluation never places expected answers or relevant document IDs in the answer-generation request; the optional DeepSeek judge receives them only after generation.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

If Docker reports it cannot connect to `dockerDesktopLinuxEngine`, start Docker Desktop, then run `docker compose up -d` again.
