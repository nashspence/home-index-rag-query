from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm import load_llm


def test_load_llm_small_model():
    llm = load_llm("sshleifer/tiny-gpt2")
    text = llm.invoke("Hello world")
    assert isinstance(text, str)
