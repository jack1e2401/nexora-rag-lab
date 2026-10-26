"""HTTP helpers for the local Apache Tika server."""

import httpx

from app.config import Settings


def check_tika(settings: Settings) -> bool:
    """Return True when Tika responds to its version endpoint."""
    version_url = settings.tika_url.rstrip("/") + "/version"

    try:
        response = httpx.get(version_url, timeout=2.0)
        return response.is_success
    except httpx.HTTPError:
        return False
