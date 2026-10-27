"""Exact cosine search within one benchmark Board and one ingestion profile."""

from dataclasses import dataclass
import math

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from app.config import Settings
from app.embeddings import embed_query


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    board_id: str
    text: str
    score: float
    rank: int
    start_char: int
    end_char: int


def validate_profile(profile: dict, settings: Settings) -> None:
    if profile["embedding_model"] != settings.bge_model_name:
        raise ValueError("Profile embedding model differs from current BGE model")
    if profile["embedding_revision"] != settings.bge_model_revision:
        raise ValueError("Profile embedding revision differs from current BGE revision")
    if profile["embedding_dimensions"] != settings.bge_dimensions:
        raise ValueError("Profile embedding dimensions differ from current BGE dimensions")


def _validate_query_vector(vector: list[float], settings: Settings) -> None:
    has_expected_width = len(vector) == settings.bge_dimensions
    has_only_finite_values = all(math.isfinite(value) for value in vector)
    if not has_expected_width or not has_only_finite_values:
        raise ValueError(f"Query vector must contain {settings.bge_dimensions} finite values")


def search(
    question: str,
    board_id: str,
    profile_id: str,
    top_k: int,
    settings: Settings,
    query_vector: list[float] | None = None,
) -> list[RetrievedChunk]:
    """Return the nearest stored chunks; Board is a caller-selected partition."""
    if not question.strip() or not board_id.strip() or not profile_id.strip():
        raise ValueError("question, board_id, and profile_id are required")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")

    with psycopg.connect(settings.database_url, row_factory=dict_row) as connection:
        register_vector(connection)
        profile = connection.execute(
            """SELECT embedding_model, embedding_revision, embedding_dimensions
               FROM profiles WHERE profile_id = %s""",
            (profile_id,),
        ).fetchone()
        if profile is None:
            raise LookupError(f"Profile '{profile_id}' does not exist; run scripts/ingest.py first")
        validate_profile(profile, settings)

        vector = query_vector if query_vector is not None else embed_query(question, settings)
        _validate_query_vector(vector, settings)
        rows = connection.execute(
            """SELECT c.chunk_id, c.document_id, d.board_id, c.text,
                      c.start_char, c.end_char,
                      1 - (c.embedding <=> %s) AS score
               FROM chunks AS c
               JOIN documents AS d ON d.document_id = c.document_id
               WHERE c.profile_id = %s AND d.board_id = %s
               ORDER BY c.embedding <=> %s, c.chunk_id
               LIMIT %s""",
            (Vector(vector), profile_id, board_id, Vector(vector), top_k),
        ).fetchall()

    return [
        RetrievedChunk(
            chunk_id=row["chunk_id"], document_id=row["document_id"],
            board_id=row["board_id"], text=row["text"], score=float(row["score"]),
            rank=index, start_char=row["start_char"], end_char=row["end_char"],
        )
        for index, row in enumerate(rows, start=1)
    ]
