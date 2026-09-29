import pytest
from pydantic import ValidationError

from config import Settings, _missing_env_error, get_settings


def test_get_settings_returns_settings_when_key_present(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.google_api_key == "abc123"


def test_get_settings_is_cached(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "abc123")
    get_settings.cache_clear()
    assert get_settings() is get_settings()


def test_missing_api_key_raises_validation_error_without_env_file(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_missing_env_error_names_the_variable_and_points_at_example(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)
    err = _missing_env_error(exc_info.value)
    assert isinstance(err, RuntimeError)
    assert "google_api_key" in str(err)
    assert ".env.example" in str(err)


def test_get_settings_wraps_missing_key_as_runtime_error(monkeypatch, tmp_path):
    # Run from an empty directory so no .env file is picked up either.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match=".env.example"):
        get_settings()
