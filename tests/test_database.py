from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database import get_meili_client, get_meta_retriever


def test_get_meili_client_instance():
    client = get_meili_client()
    assert client
    assert hasattr(client, 'http')


def test_meta_retriever():
    retriever = get_meta_retriever()
    assert hasattr(retriever, "get_relevant_documents")
