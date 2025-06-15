from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import llm as llm_module
from app.config import settings
from app.llm import load_llm


def test_load_llm_small_model():
    llm = load_llm("sshleifer/tiny-gpt2")
    text = llm.invoke("Hello world")
    assert isinstance(text, str)


def test_load_llm_cached_default(monkeypatch):
    monkeypatch.setattr(llm_module, "_cached_llm", None, raising=False)
    monkeypatch.setattr(llm_module, "_cached_model_name", None, raising=False)
    monkeypatch.setattr(settings, "llm_model_name", "sshleifer/tiny-gpt2")

    llm1 = load_llm()
    llm2 = load_llm()

    assert llm1 is llm2


def test_load_llm_cached_same_model(monkeypatch):
    monkeypatch.setattr(llm_module, "_cached_llm", None, raising=False)
    monkeypatch.setattr(llm_module, "_cached_model_name", None, raising=False)

    llm1 = load_llm("sshleifer/tiny-gpt2")
    llm2 = load_llm("sshleifer/tiny-gpt2")

    assert llm1 is llm2
