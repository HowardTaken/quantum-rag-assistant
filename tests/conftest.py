import pytest

from config import get_settings


@pytest.fixture(autouse=True)
def _isolated_settings(monkeypatch):
    """Every test gets a dummy API key and a fresh, uncached Settings instance,
    so the suite never depends on a real .env file or a real key being present."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-api-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
