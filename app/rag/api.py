"""Swagger-friendly search and ask routes."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.rag.generation import DeepSeekError
from app.rag.pipeline import ask
from app.rag.retrieval import search


router = APIRouter(prefix="/rag", tags=["RAG Query"])


class RagRequest(BaseModel):
    question: str = Field(min_length=1)
    board_id: str = Field(min_length=1)
    profile_id: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=10)


@router.post("/search")
def search_route(request: RagRequest, settings: Settings = Depends(get_settings)) -> dict:
    try:
        chunks = search(request.question, request.board_id, request.profile_id,
                        request.top_k, settings)
    except (ValueError, LookupError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"chunks": [chunk.__dict__ for chunk in chunks]}


@router.post("/ask")
def ask_route(request: RagRequest, settings: Settings = Depends(get_settings)) -> dict:
    try:
        result = ask(request.question, request.board_id, request.profile_id,
                     request.top_k, settings)
    except (ValueError, LookupError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except DeepSeekError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return {
        "answer": result.answer, "abstained": result.abstained,
        "citations": [citation.__dict__ for citation in result.citations],
        "retrieved_chunks": [chunk.__dict__ for chunk in result.retrieved_chunks],
        "model": result.model, "usage": result.usage,
    }
