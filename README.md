# Nexora RAG Lab

A small, local lab for understanding and evaluating a RAG workflow. The current runnable path is **Tika plain text → fixed-token chunks → local BGE-M3 embeddings**. Each step has a FastAPI endpoint so its output can be inspected in Swagger

## Current status

| Step | Available now | What it does |
| --- | --- | --- |
| Extract | Yes | Tika `PUT /tika/text` returns normalized plain text from a PDF, DOCX, or TXT file. |
| Tokenize | Yes | Shows BGE-M3 token IDs and their character positions. |
| Chunk | Fixed-token only | Returns chunk text, token counts, overlap, and exact character positions in the extracted text. |
| Embed | Yes | Returns local BGE-M3 vectors with 1,024 values each. |
| Store and search | No | The pgvector schema exists, but ingestion and retrieval code have not been implemented. |
| LLM chunking | No | DeepSeek-based chunking is the next separate implementation task. |

The extraction and embedding preview endpoints do **not** save documents, chunks, or vectors. The database health check tests connectivity only. The schema in `sql/schema.sql` is not applied by `docker compose up`.

## Run locally on Windows

From the repository root in PowerShell:

```powershell
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
docker compose up -d
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/health` and expect `{"database":"ok","tika":"ok"}`. Open `http://127.0.0.1:8000/docs` for the step-by-step endpoints. Run Uvicorn with the `.venv` Python shown above so it can find the dependencies declared in `pyproject.toml`.

## Try the workflow in Swagger

1. **POST `/ingestion/extract`:** upload a PDF, DOCX, or TXT file. Copy the returned `text` value.
2. **POST `/tokenization/inspect` (optional):** send a short excerpt as `{"text":"..."}` to see how BGE-M3 tokenizes it.
3. **POST `/chunking/split`:** put the extracted content in `text`. For example:

   ```json
   {
     "text": "A first sentence. A second sentence.",
     "strategy": "fixed_token",
     "chunk_size": 80,
     "overlap": 10
   }
   ```

   `chunk_size` and `overlap` are measured in **BGE tokens**. `start_char` and `end_char` are positions in the extracted plain text. Fixed-token chunks may end in the middle of a sentence; overlap can repeat that sentence in the next chunk. A boundary at a sentence is not guaranteed.
4. **POST `/embeddings/embed`:** copy up to four chunk `text` values into `texts`:

   ```json
   {"texts": ["A first chunk.", "A second chunk."]}
   ```

   The response contains one 1,024-dimensional vector per input. The first call may download and load BGE-M3 on the host CPU; it can take time and substantial memory. Docker runs Tika and PostgreSQL, not BGE-M3.

For a whole document pasted into Swagger's JSON editor, preserve JSON escapes such as `\n` for newlines and `\"` for quotation marks.

If `/chunking/split` raises `ModuleNotFoundError: transformers`, install the declared dependencies using the `.venv` Python command above. If Swagger reports **Failed to fetch**, first check `http://127.0.0.1:8000/health` and the Uvicorn terminal: the server may have stopped while loading the local model.

## Check the code

- `app/ingestion/extractor.py`: Tika request and text normalization.
- `app/ingestion/chunking/fixed_token_strategy.py`: fixed-token chunk selection.
- `app/ingestion/chunking/utils/`: token counts, character ranges, and checked chunk creation.
- `app/embeddings.py`: lazy BGE-M3 tokenizer/model loading and vector generation.
- `app/lab_api.py`: Swagger endpoints for the four inspectable steps.
- `sql/schema.sql`: database schema prepared for later ingestion.

Run the automated tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The [current chunking design](docs/superpowers/specs/2026-10-27-plain-text-two-chunkers-design.md) and [implementation plan](docs/superpowers/plans/2026-10-27-plain-text-two-chunkers.md) describe the remaining LLM chunking task. Earlier Markdown, recursive, and document-structure plans are historical and marked as superseded.
