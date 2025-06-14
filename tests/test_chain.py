from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.chains.router import MultiRetrievalQAChain
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_models import ChatOpenAI

from app.llm import load_llm
from app.database import get_meta_retriever


class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, run_manager=None):
        return []


def test_chain_uses_default_retriever(monkeypatch):
    dummy = DummyRetriever()
    monkeypatch.setattr('app.database.get_vector_retriever', lambda: dummy)

    llm = load_llm("sshleifer/tiny-gpt2")
    chain = MultiRetrievalQAChain.from_retrievers(
        llm=llm,
        retriever_infos=[
            {
                "name": "metadata",
                "description": "file metadata search",
                "retriever": get_meta_retriever(),
            },
            {
                "name": "semantic",
                "description": "content chunk semantic search",
                "retriever": dummy,
            },
        ],
        default_retriever=dummy,
    )

    default_llm = chain.default_chain.combine_documents_chain.llm_chain.llm
    assert default_llm is llm
    assert not isinstance(default_llm, ChatOpenAI)
