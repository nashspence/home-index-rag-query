from __future__ import annotations

"""Utilities for building LangChain question answering pipelines."""

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.language_models import BaseChatModel

from .database import get_parent_retriever
from .llm import load_llm


def build_qa_chain(model: BaseChatModel | None = None) -> RetrievalQAWithSourcesChain:
    """Return a `RetrievalQAWithSourcesChain` configured for the app."""
    llm = model or load_llm()
    retriever = get_parent_retriever()
    return RetrievalQAWithSourcesChain.from_chain_type(
        llm, retriever=retriever, return_source_documents=True
    )

