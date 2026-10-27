"""Extract and normalize plain document text through Apache Tika."""

import mimetypes
import unicodedata
from pathlib import Path

import httpx


class ExtractionError(Exception):
    """A document could not be read or Tika returned no usable text."""


def normalize_text(text: str) -> str:
    """Normalize Unicode and line endings while keeping text line breaks."""
    normalized_text = unicodedata.normalize("NFC", text)
    normalized_text = normalized_text.replace("\r\n", "\n").replace("\r", "\n")

    normalized_lines = []

    for line in normalized_text.split("\n"):
        clean_line = line.rstrip()

        if clean_line.strip():
            normalized_lines.append(clean_line)
        elif normalized_lines and normalized_lines[-1] != "":
            normalized_lines.append("")

    while normalized_lines and normalized_lines[-1] == "":
        normalized_lines.pop()

    return "\n".join(normalized_lines)


def extract_text(path: Path, tika_url: str) -> str:
    """Read one document and return normalized plain text from Tika."""
    try:
        document_bytes = path.read_bytes()
    except OSError as error:
        raise ExtractionError(f"Could not read '{path.name}': {error}") from error

    return extract_bytes(path.name, document_bytes, tika_url)


def extract_bytes(filename: str, document_bytes: bytes, tika_url: str) -> str:
    """Send in-memory document bytes to Tika and return normalized plain text."""
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    request_url = tika_url.rstrip("/") + "/tika/text"

    try:
        response = httpx.put(
            request_url,
            content=document_bytes,
            headers={"Content-Type": content_type},
            timeout=60.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise ExtractionError(f"Tika could not extract '{filename}': {error}") from error

    extracted_text = normalize_text(response.text)

    if not extracted_text:
        raise ExtractionError(f"Tika returned no text for '{filename}'")

    return extracted_text


# https://tika.apache.org/docs/4.1.x/using-tika/server/index.html
# Tika 4.1 supports PUT /tika/text and returns plain document text.
