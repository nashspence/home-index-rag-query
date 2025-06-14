from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain.chains import RetrievalQAWithSourcesChain
from langchain_core.retrievers import BaseRetriever
from langchain_community.chat_models import ChatOpenAI

from app.llm import load_llm


class DummyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str, run_manager=None):
        return []


def test_chain_uses_llm():
    dummy = DummyRetriever()

    llm = load_llm("sshleifer/tiny-gpt2")
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=dummy)

    chain_llm = chain.combine_documents_chain.llm_chain.llm
    assert chain_llm is llm
    assert not isinstance(chain_llm, ChatOpenAI)
