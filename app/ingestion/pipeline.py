"""Synchronous manifest-driven ingestion into PostgreSQL and pgvector."""

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from app.config import Settings
from app.embeddings import embed_passages, get_tokenizer
from app.ingestion.chunking import ChunkSlice, ChunkingSettings, FixedTokenStrategy
from app.ingestion.extractor import extract_text


EXTRACTION_VERSION = "tika-text-v1"


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    board_id: str
    path: Path
    relative_path: str


@dataclass(frozen=True)
class PreparedDocument:
    source: SourceDocument
    content_sha256: str
    text: str
    chunks: list[ChunkSlice]
    vectors: list[list[float]]


@dataclass(frozen=True)
class IngestReport:
    profile_id: str
    corpus_fingerprint: str
    documents: int
    chunks: int


def load_manifest(manifest_path: Path) -> list[SourceDocument]:
    """Select only the source files explicitly listed in the manifest."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent.resolve()
    documents = []
    seen_ids = set()

    for domain in manifest["domains"]:
        for item in domain["docs"]:
            relative_path = item["path"]
            path = (root / relative_path).resolve()
            if not path.is_relative_to(root) or not relative_path.startswith("01_CORPUS/"):
                raise ValueError(f"Document path is outside the corpus: {relative_path}")
            if item["id"] in seen_ids:
                raise ValueError(f"Duplicate document ID: {item['id']}")
            if not path.is_file():
                raise FileNotFoundError(path)
            seen_ids.add(item["id"])
            documents.append(SourceDocument(item["id"], domain["board_id"], path, relative_path))

    if not documents:
        raise ValueError("Manifest contains no source documents")
    return documents


def corpus_fingerprint(documents: list[SourceDocument]) -> str:
    """Hash source identities and bytes so profiles can share one corpus safely."""
    digest = sha256(EXTRACTION_VERSION.encode("utf-8"))
    for document in sorted(documents, key=lambda item: item.document_id):
        file_hash = sha256(document.path.read_bytes()).hexdigest()
        row = [document.document_id, document.board_id, document.relative_path, file_hash]
        digest.update(json.dumps(row, ensure_ascii=False).encode("utf-8"))
    return digest.hexdigest()


def validate_vector(vector: list[float], settings: Settings) -> None:
    if len(vector) != settings.bge_dimensions:
        raise ValueError(f"Expected {settings.bge_dimensions} embedding values")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("Embedding values must be finite")


def _prepare_documents(
    documents: list[SourceDocument], chunk_size: int, overlap: int, settings: Settings
) -> list[PreparedDocument]:
    tokenizer = get_tokenizer(settings)
    chunk_settings = ChunkingSettings(chunk_size, overlap, settings.bge_max_input_tokens)
    strategy = FixedTokenStrategy()
    prepared = []

    for source in documents:
        source_bytes = source.path.read_bytes()
        text = extract_text(source.path, settings.tika_url)
        chunks = strategy.split(text, tokenizer, chunk_settings)
        if not chunks:
            raise ValueError(f"No chunks produced for {source.document_id}")
        vectors = embed_passages([chunk.text for chunk in chunks], settings)
        if len(vectors) != len(chunks):
            raise ValueError(f"Embedding count differs from chunk count for {source.document_id}")
        for vector in vectors:
            validate_vector(vector, settings)
        prepared.append(PreparedDocument(source, sha256(source_bytes).hexdigest(), text, chunks, vectors))

    return prepared


def _initialize_schema(connection: psycopg.Connection) -> None:
    schema_path = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"
    schema = schema_path.read_text(encoding="utf-8")
    for statement in schema.split(";"):
        if statement.strip():
            connection.execute(statement)
    register_vector(connection)


def _check_existing_extractions(
    connection: psycopg.Connection, prepared: list[PreparedDocument]
) -> None:
    """Keep stored source text stable while older profiles still reference it."""
    rows = connection.execute("SELECT document_id, extracted_text FROM documents").fetchall()
    stored_text = {document_id: text for document_id, text in rows}
    for item in prepared:
        document_id = item.source.document_id
        if document_id in stored_text and stored_text[document_id] != item.text:
            raise ValueError(
                f"{document_id}: extracted text changed; rerun with --reset-corpus "
                "to replace all profiles"
            )


def ingest_manifest(
    manifest_path: Path,
    profile_id: str,
    chunk_size: int,
    overlap: int,
    settings: Settings,
    reset_corpus: bool = False,
) -> IngestReport:
    """Prepare before writing, then replace one profile in a DB transaction."""
    if not profile_id.strip():
        raise ValueError("profile_id cannot be empty")
    ChunkingSettings(chunk_size, overlap, settings.bge_max_input_tokens)
    documents = load_manifest(manifest_path)
    fingerprint = corpus_fingerprint(documents)

    with psycopg.connect(settings.database_url) as connection:
        _initialize_schema(connection)
        existing = connection.execute("SELECT DISTINCT corpus_fingerprint FROM profiles").fetchall()
    if existing and any(row[0] != fingerprint for row in existing) and not reset_corpus:
        raise ValueError("Corpus changed; rerun with --reset-corpus to replace all profiles")

    prepared = _prepare_documents(documents, chunk_size, overlap, settings)
    chunk_total = sum(len(item.chunks) for item in prepared)

    with psycopg.connect(settings.database_url) as connection:
        register_vector(connection)
        existing = connection.execute("SELECT DISTINCT corpus_fingerprint FROM profiles").fetchall()
        if existing and any(row[0] != fingerprint for row in existing) and not reset_corpus:
            raise ValueError("Corpus changed; rerun with --reset-corpus to replace all profiles")

        if reset_corpus:
            connection.execute("DELETE FROM profiles")
            connection.execute("DELETE FROM documents")
        else:
            _check_existing_extractions(connection, prepared)
            connection.execute("DELETE FROM profiles WHERE profile_id = %s", (profile_id,))

        for item in prepared:
            connection.execute(
                """INSERT INTO documents
                   (document_id, board_id, source_path, content_sha256, extracted_text)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (document_id) DO UPDATE SET
                     board_id = EXCLUDED.board_id, source_path = EXCLUDED.source_path,
                     content_sha256 = EXCLUDED.content_sha256,
                     extracted_text = EXCLUDED.extracted_text""",
                (item.source.document_id, item.source.board_id, item.source.relative_path,
                 item.content_sha256, item.text),
            )

        embedding_settings = {
            "strategy": FixedTokenStrategy.strategy_id,
            "strategy_version": FixedTokenStrategy.strategy_version,
            "device": settings.bge_device,
            "precision": settings.bge_precision, "max_input_tokens": settings.bge_max_input_tokens,
        }
        connection.execute(
            """INSERT INTO profiles
               (profile_id, chunk_size, overlap, embedding_model, embedding_revision,
                embedding_dimensions, embedding_settings, corpus_fingerprint, extraction_version)
               VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)""",
            (profile_id, chunk_size, overlap, settings.bge_model_name,
             settings.bge_model_revision, settings.bge_dimensions,
             json.dumps(embedding_settings), fingerprint, EXTRACTION_VERSION),
        )

        for item in prepared:
            for chunk, vector in zip(item.chunks, item.vectors, strict=True):
                chunk_key = f"{profile_id}:{item.source.document_id}:{chunk.order}:{chunk.text}"
                chunk_id = sha256(chunk_key.encode("utf-8")).hexdigest()
                connection.execute(
                    """INSERT INTO chunks
                       (chunk_id, profile_id, document_id, ordinal, text, token_count,
                        start_char, end_char, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                    (chunk_id, profile_id, item.source.document_id, chunk.order, chunk.text,
                     chunk.token_count, chunk.start_char, chunk.end_char, Vector(vector)),
                )

    return IngestReport(profile_id, fingerprint, len(prepared), chunk_total)
