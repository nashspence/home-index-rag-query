from __future__ import annotations

from meilisearch import Client
from langchain_community.vectorstores import Meilisearch as MeiliVector
from langchain_community.embeddings import HuggingFaceEmbeddings

from .config import settings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

_client = None


def get_meili_client() -> Client:
    """Create or return a cached MeiliSearch client."""
    global _client
    if _client is None:
        _client = Client(settings.meili_url, settings.meili_api_key)
    return _client


def search_index(index_name: str, query: str, limit: int = 5) -> list[dict]:
    client = get_meili_client()
    index = client.index(index_name)
    result = index.search(query, {"limit": limit})
    return result.get("hits", [])


class MetadataRetriever(BaseRetriever):
    """Simple retriever that performs a MeiliSearch text query."""

    def _get_relevant_documents(self, query: str, run_manager=None):
        hits = search_index(settings.files_index, query, limit=4)
        return [Document(page_content=h.get("content", ""), metadata=h) for h in hits]


def get_meta_retriever():
    """Return a retriever for lexical search of file metadata."""
    return MetadataRetriever()


def get_vector_retriever():
    """Return a retriever for semantic search of file chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model_name)
    store = MeiliVector(
        index_name=settings.file_chunks_index,
        url=settings.meili_url,
        api_key=settings.meili_api_key,
        embedding_function=embeddings.embed_query,
    )
    return store.as_retriever()
