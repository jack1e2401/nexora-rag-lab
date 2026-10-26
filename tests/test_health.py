import importlib
import importlib.util

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from app import main
from app.config import Settings


def make_settings():
    return Settings(
        database_url="postgresql://postgres:postgres@localhost:5432/nexora_rag"
    )


def test_settings_require_database_url():
    with pytest.raises(ValidationError):
        Settings()


def test_tika_check_uses_version_endpoint(monkeypatch):
    try:
        module_spec = importlib.util.find_spec("app.tika")
    except ModuleNotFoundError:
        module_spec = None

    assert module_spec is not None, "app.tika has not been implemented"
    tika = importlib.import_module("app.tika")
    settings = make_settings()
    request = {}

    class Response:
        is_success = True

    def fake_get(url, timeout):
        request["url"] = url
        request["timeout"] = timeout
        return Response()

    monkeypatch.setattr(tika.httpx, "get", fake_get)

    assert tika.check_tika(settings) is True
    assert request == {
        "url": "http://localhost:9998/version",
        "timeout": 2.0,
    }


def test_health_ok(monkeypatch):
    settings = make_settings()

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "check_database", lambda current_settings: True)
    monkeypatch.setattr(main, "check_tika", lambda current_settings: True)

    response = main.health()

    assert response == {"database": "ok", "tika": "ok"}


def test_health_names_failed_dependency(monkeypatch):
    settings = make_settings()

    monkeypatch.setattr(main, "get_settings", lambda: settings)
    monkeypatch.setattr(main, "check_database", lambda current_settings: False)
    monkeypatch.setattr(main, "check_tika", lambda current_settings: True)

    with pytest.raises(HTTPException) as error:
        main.health()

    assert error.value.status_code == 503
    assert error.value.detail == {"database": "error", "tika": "ok"}
