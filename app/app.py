import streamlit as st

from langchain.chains import RetrievalQAWithSourcesChain

from .config import settings
from .database import (
    search_index,
    get_parent_retriever,
)
from .llm import load_llm


def format_sources(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        file_id = h.get("file_id") or h.get("id")
        path = h.get("path")
        lines.append(f"- {file_id}: {path}")
    return "\n".join(lines)


def main():
    st.title("Home Index RAG")
    query = st.text_input("Ask a question:")
    model_name = st.sidebar.text_input(
        "HuggingFace model", value=settings.llm_model_name
    )

    if st.sidebar.button("Load model"):
        llm = load_llm(model_name)
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm, retriever=get_parent_retriever()
        )
        st.session_state["chain"] = chain
        st.success(f"Loaded model {model_name}")

    chain = st.session_state.get("chain")
    if query and chain:
        answer = chain.run(query)
        st.write(answer)
        hits = search_index(settings.file_chunks_index, query, limit=5)
        st.markdown("## Sources")
        st.text(format_sources(hits))
    elif query:
        st.warning("Load the model first using the sidebar.")


if __name__ == "__main__":
    main()
