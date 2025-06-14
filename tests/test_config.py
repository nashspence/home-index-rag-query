from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    monkeypatch.setenv("EMBED_MODEL_NAME", "test-embed")
    monkeypatch.setenv("FILES_DOMAIN", "https://example.com")
    s = Settings()
    assert s.llm_model_name == "test-model"
    assert s.embed_model_name == "test-embed"
    assert s.files_domain == "https://example.com"
