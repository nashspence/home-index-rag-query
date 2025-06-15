import streamlit as st
from urllib.parse import urljoin

from langchain_core.documents import Document

from .config import settings
from .llm import load_llm
from .chain import build_qa_chain


@st.cache_resource(show_spinner=False)
def get_chain(model_name: str):
    """Load model and return a QA chain."""
    llm = load_llm(model_name)
    return build_qa_chain(llm)

def render_source(doc: Document, *, base_download_dir: str = "/downloads/") -> None:
    """Render a document source in Streamlit."""
    mime = str(doc.metadata.get("mime", ""))
    url = doc.metadata.get("url")
    if not url:
        path = doc.metadata.get("path")
        if path:
            url = urljoin(settings.files_domain + "/", str(path).lstrip("/"))
    if not url:
        return

    if mime.startswith("video"):
        t0 = int(doc.metadata.get("start", 0))
        st.video(url, start_time=t0)
        st.markdown(f"[Open fullscreen]({url}#t={t0})")
    elif mime.startswith("audio"):
        st.audio(url)
        st.markdown(f"[Download clip]({url})")
    elif mime.startswith("image"):
        st.image(url)
        st.markdown(f"[View full-size]({url})")
    else:
        label = url.split("/")[-1]
        st.download_button(
            label=f"Download {label}",
            data=None,
            file_name=label,
            url=url,
        )


def main():
    st.title("Home Index RAG")
    query = st.text_input("Ask a question:")
    model_name = st.sidebar.text_input(
        "HuggingFace model", value=settings.llm_model_name
    )

    if st.sidebar.button("Load model"):
        st.session_state["chain"] = get_chain(model_name)
        st.success(f"Loaded model {model_name}")

    chain = st.session_state.get("chain")
    if query and chain:
        result = chain.invoke({"question": query})
        answer = result.get("answer", "") if isinstance(result, dict) else str(result)
        docs = result.get("source_documents", []) if isinstance(result, dict) else []
        with st.chat_message("assistant"):
            st.markdown(answer)
            for d in docs:
                render_source(d)
    elif query:
        st.warning("Load the model first using the sidebar.")


if __name__ == "__main__":
    main()
