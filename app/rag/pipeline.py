"""Combine retrieval and generation for one question."""

from dataclasses import dataclass

from app.config import Settings
from app.rag.generation import Citation, abstain, generate_answer
from app.rag.retrieval import RetrievedChunk, search


@dataclass(frozen=True)
class AskResult:
    answer: str
    citations: list[Citation]
    abstained: bool
    retrieved_chunks: list[RetrievedChunk]
    model: str | None
    usage: dict | None


def ask(
    question: str,
    board_id: str,
    profile_id: str,
    top_k: int,
    settings: Settings,
    query_vector: list[float] | None = None,
) -> AskResult:
    chunks = search(question, board_id, profile_id, top_k, settings, query_vector)
    generated = generate_answer(question, chunks, settings) if chunks else abstain()
    return AskResult(
        generated.answer,
        generated.citations,
        generated.abstained,
        chunks,
        generated.model,
        generated.usage,
    )
