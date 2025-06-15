from __future__ import annotations

from typing import Iterator, Optional, Sequence, Tuple

from meilisearch import Client
from langchain_community.vectorstores import Meilisearch as MeiliVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_core.stores import BaseStore

from .config import settings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from urllib.parse import urljoin
from datetime import datetime, UTC

_client = None


class MeiliDocStore(BaseStore[str, Document]):
    """Read-only DocStore backed by a Meilisearch index."""

    def __init__(self, client: Client, index_name: str) -> None:
        self.index = client.index(index_name)

    def mget(self, keys: Sequence[str]) -> list[Optional[Document]]:
        docs: list[Optional[Document]] = []
        for key in keys:
            try:
                data = self.index.get_document(key)
            except Exception:
                docs.append(None)
                continue
            docs.append(Document(page_content=data.get("content", ""), metadata=data))
        return docs

    def mset(self, key_value_pairs: Sequence[Tuple[str, Document]]) -> None:
        raise NotImplementedError("This docstore is read only")

    def mdelete(self, keys: Sequence[str]) -> None:
        raise NotImplementedError("This docstore is read only")

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        raise NotImplementedError("This docstore is read only")


def get_meili_client() -> Client:
    """Create or return a cached MeiliSearch client."""
    global _client
    if _client is None:
        _client = Client(settings.meili_url, settings.meili_api_key)
    return _client


def search_index(index_name: str, query: str, limit: int = 5, **params) -> list[dict]:
    """Run a Meilisearch query and return hits."""
    client = get_meili_client()
    index = client.index(index_name)
    search_params = {"limit": limit, **params}
    result = index.search(query, search_params)
    return result.get("hits", [])


class MetadataRetriever(BaseRetriever):
    """Simple retriever that performs a MeiliSearch text query."""

    def _get_relevant_documents(self, query: str, run_manager=None):
        hits = search_index(settings.files_index, query, limit=4)
        return [Document(page_content=h.get("content", ""), metadata=h) for h in hits]


class CanonicalURLRetriever(BaseRetriever):
    """Wrap a retriever and normalise file metadata for the LLM."""

    wrapped: BaseRetriever
    base_url: str

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, wrapped: BaseRetriever, base_url: str) -> None:
        super().__init__(wrapped=wrapped, base_url=base_url.rstrip("/"))

    def _get_relevant_documents(self, query: str, run_manager=None):
        docs = self.wrapped.invoke(query)
        for d in docs:
            paths_map = d.metadata.get("paths")
            if isinstance(paths_map, dict):
                canonical_url = None
                latest = None
                urls = []
                for path, mtime in paths_map.items():
                    abs_url = urljoin(self.base_url + "/", path.lstrip("/"))
                    urls.append(abs_url)
                    if latest is None or mtime > latest:
                        canonical_url = abs_url
                        latest = mtime
                d.metadata["paths"] = urls
                if canonical_url is not None:
                    d.metadata["url"] = canonical_url
                    if latest is not None:
                        d.metadata["mtime"] = (
                            datetime.fromtimestamp(latest, UTC)
                            .strftime("%a %Y-%m-%d %H:%M:%SZ")
                        )
            else:
                path = d.metadata.get("path")
                if path:
                    abs_url = urljoin(self.base_url + "/", str(path).lstrip("/"))
                    d.metadata["url"] = abs_url
        return docs


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


def get_parent_retriever():
    """Return a retriever that links chunks to their parent documents."""
    embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model_name)
    chunks_vs = MeiliVector(
        index_name=settings.file_chunks_index,
        url=settings.meili_url,
        api_key=settings.meili_api_key,
        embedding_function=embeddings.embed_query,
    )
    meta_store = MeiliDocStore(get_meili_client(), settings.files_index)
    parent = ParentDocumentRetriever(
        vectorstore=chunks_vs,
        docstore=meta_store,
        id_key="file_id",
    )
    return CanonicalURLRetriever(parent, base_url=settings.files_domain)
