"""Read application settings from the environment and the local .env file."""

import os

from dotenv import dotenv_values
from pydantic import BaseModel, Field


class Settings(BaseModel):
    database_url: str = Field(min_length=1)
    tika_url: str = "http://localhost:9998"

    bge_model_name: str = "BAAI/bge-m3"
    bge_model_revision: str = "31e47391fcbda65be526abe98e646b3c6cd845a8"
    bge_dimensions: int = 1024
    bge_device: str = "cpu"
    bge_precision: str = "float32"
    bge_batch_size: int = 4
    bge_max_input_tokens: int = 1200

    deepseek_api_key: str | None = None
    deepseek_api_url: str = "https://api.deepseek.com"
    deepseek_generation_model: str = "deepseek-flash"
    deepseek_judge_model: str = "deepseek-v4-pro"

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_base_url: str = "https://cloud.langfuse.com"


def get_settings() -> Settings:
    """Load .env values, then let real environment variables take precedence."""
    raw_values = dotenv_values(".env")
    values = {}

    for name, value in raw_values.items():
        if value is not None:
            values[name] = value

    for name, value in os.environ.items():
        values[name] = value

    return Settings(
        database_url=values.get("DATABASE_URL", ""),
        tika_url=values.get("TIKA_URL", "http://localhost:9998"),
        bge_model_name=values.get("BGE_MODEL_NAME", "BAAI/bge-m3"),
        bge_model_revision=values.get(
            "BGE_MODEL_REVISION",
            "31e47391fcbda65be526abe98e646b3c6cd845a8",
        ),
        bge_dimensions=values.get("BGE_DIMENSIONS", 1024),
        bge_device=values.get("BGE_DEVICE", "cpu"),
        bge_precision=values.get("BGE_PRECISION", "float32"),
        bge_batch_size=values.get("BGE_BATCH_SIZE", 4),
        bge_max_input_tokens=values.get("BGE_MAX_INPUT_TOKENS", 1200),
        deepseek_api_key=values.get("DEEPSEEK_API_KEY"),
        deepseek_api_url=values.get("DEEPSEEK_API_URL", "https://api.deepseek.com"),
        deepseek_generation_model=values.get(
            "DEEPSEEK_GENERATION_MODEL", "deepseek-flash"
        ),
        deepseek_judge_model=values.get(
            "DEEPSEEK_JUDGE_MODEL", "deepseek-v4-pro"
        ),
        langfuse_public_key=values.get("LANGFUSE_PUBLIC_KEY"),
        langfuse_secret_key=values.get("LANGFUSE_SECRET_KEY"),
        langfuse_base_url=values.get(
            "LANGFUSE_BASE_URL", "https://cloud.langfuse.com"
        ),
    )
