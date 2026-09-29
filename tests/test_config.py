import pytest

from config import Settings, get_settings


def test_get_settings_returns_settings_when_key_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.google_api_key == "abc123"


def test_get_settings_is_cached(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_settings_constructs_fine_with_no_api_key(monkeypatch, tmp_path):
    # Settings() must never require GOOGLE_API_KEY to construct -- code that only reads
    # unrelated fields (e.g. CORS origins, at module import time) shouldn't need one, and
    # importing backend.main in a clean environment (like CI) must not blow up.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    assert settings.google_api_key is None


def test_require_google_api_key_raises_a_clear_error_when_missing(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    settings = Settings()
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY") as exc_info:
        settings.require_google_api_key()
    assert ".env.example" in str(exc_info.value)


def test_require_google_api_key_returns_the_key_when_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    assert Settings().require_google_api_key() == "abc123"


def test_cors_origins_list_splits_and_strips_the_default():
    settings = Settings(_env_file=None)
    assert settings.cors_origins_list == [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://howardtaken.github.io",
    ]


def test_cors_origins_list_respects_override(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", " https://example.com , https://other.example.com ")
    get_settings.cache_clear()
    assert get_settings().cors_origins_list == ["https://example.com", "https://other.example.com"]
