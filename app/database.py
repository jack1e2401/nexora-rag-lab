"""Small PostgreSQL connectivity check used by the health endpoint."""

from app.config import Settings


def check_database(settings: Settings) -> bool:
    """Return True when PostgreSQL accepts a simple query."""
    try:
        import psycopg
    except ImportError:
        return False

    try:
        with psycopg.connect(settings.database_url, connect_timeout=2) as connection:
            connection.execute("SELECT 1")
        return True
    except psycopg.Error:
        return False
