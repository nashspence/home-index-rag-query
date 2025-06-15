from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import (
    get_meili_client,
    get_meta_retriever,
    CanonicalURLRetriever,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


def test_get_meili_client_instance():
    client = get_meili_client()
    assert client
    assert hasattr(client, "http")


def test_meta_retriever():
    retriever = get_meta_retriever()
    assert hasattr(retriever, "get_relevant_documents")


class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, run_manager=None):
        meta = {
            "paths": {"a.txt": 1.0, "b.txt": 2.0},
            "mtime": 2.0,
        }
        return [Document(page_content="", metadata=meta)]


def test_canonical_retriever():
    dummy = DummyRetriever()
    wrapper = CanonicalURLRetriever(dummy, base_url="https://domain")
    docs = wrapper.invoke("q")
    assert docs[0].metadata["url"].endswith("b.txt")
    assert docs[0].metadata["paths"][0].startswith("https://domain")
    assert isinstance(docs[0].metadata["mtime"], str)
