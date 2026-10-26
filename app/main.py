"""FastAPI entry point for health checks and the later RAG routes."""

from fastapi import FastAPI, HTTPException

from app.config import Settings, get_settings
from app.database import check_database
from app.tika import check_tika


app = FastAPI(title="Nexora RAG Lab")


@app.get("/health")
def health() -> dict[str, str]:
    """Check PostgreSQL and Tika and name any dependency that is unavailable."""
    settings = get_settings()
    database_status = "ok" if check_database(settings) else "error"
    tika_status = "ok" if check_tika(settings) else "error"
    result = {"database": database_status, "tika": tika_status}

    if database_status == "error" or tika_status == "error":
        raise HTTPException(status_code=503, detail=result)

    return result
