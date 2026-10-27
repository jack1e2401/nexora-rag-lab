"""Small endpoints for trying each implemented RAG lab step in Swagger."""

from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, model_validator

from app import embeddings
from app.config import Settings, get_settings
from app.ingestion import extractor
from app.ingestion.chunking import ChunkingSettings, get_chunking_strategy


router = APIRouter()


class TextRequest(BaseModel):
    text: str = Field(min_length=1)


class ChunkRequest(BaseModel):
    markdown: str = Field(min_length=1)
    strategy: Literal["fixed_token", "recursive", "document_structure"] = "fixed_token"
    chunk_size: int = Field(default=500, gt=0)
    overlap: int = Field(default=50, ge=0)

    @model_validator(mode="after")
    def overlap_must_be_smaller_than_chunk_size(self):
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        return self


class EmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=4)


@router.post("/ingestion/extract", tags=["Step 1 - Extract"])
def extract_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
) -> dict[str, str | int]:
    """Extract one uploaded document to Markdown without saving the file."""
    filename = file.filename or "uploaded-document"
    document_bytes = file.file.read()

    if not document_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")

    try:
        markdown = extractor.extract_bytes(filename, document_bytes, settings.tika_url)
    except extractor.ExtractionError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "filename": filename,
        "character_count": len(markdown),
        "markdown": markdown,
    }


@router.post("/tokenization/inspect", tags=["Step 2 - Tokenize"])
def inspect_tokens(
    request: TextRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Show how the real BGE tokenizer splits text and maps tokens to characters."""
    tokenizer = embeddings.get_tokenizer(settings)
    encoded = tokenizer(
        request.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = encoded["input_ids"]
    character_offsets = encoded["offset_mapping"]

    if len(token_ids) != len(character_offsets):
        raise HTTPException(
            status_code=500,
            detail="The tokenizer did not return one character range per token",
        )

    token_texts = tokenizer.convert_ids_to_tokens(token_ids)
    tokens = []

    for index, token_id in enumerate(token_ids):
        start_char, end_char = character_offsets[index]
        tokens.append(
            {
                "index": index,
                "token_id": token_id,
                "token": token_texts[index],
                "start_char": start_char,
                "end_char": end_char,
                "source_text": request.text[start_char:end_char],
            }
        )

    return {"token_count": len(token_ids), "tokens": tokens}


@router.post("/chunking/split", tags=["Step 3 - Chunk"])
def split_markdown(
    request: ChunkRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Split Markdown and return every chunk with its size and source positions."""
    tokenizer = embeddings.get_tokenizer(settings)
    strategy = get_chunking_strategy(request.strategy)
    chunk_settings = ChunkingSettings(
        chunk_size=request.chunk_size,
        overlap=request.overlap,
        max_input_tokens=settings.bge_max_input_tokens,
    )
    chunks = strategy.split(request.markdown, tokenizer, chunk_settings)

    return {
        "strategy": request.strategy,
        "chunk_size": request.chunk_size,
        "overlap": request.overlap,
        "chunk_count": len(chunks),
        "chunks": [
            {
                "order": chunk.order,
                "text": chunk.text,
                "token_count": chunk.token_count,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "fallback_reason": chunk.fallback_reason,
            }
            for chunk in chunks
        ],
    }


@router.post("/embeddings/embed", tags=["Step 4 - Embed"])
def create_embeddings(
    request: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
) -> dict[str, object]:
    """Create local BGE-M3 vectors for up to four text passages."""
    try:
        vectors = embeddings.embed_passages(request.texts, settings)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "model": settings.bge_model_name,
        "count": len(vectors),
        "dimensions": settings.bge_dimensions,
        "vectors": vectors,
    }
