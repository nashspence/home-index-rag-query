# Home Index RAG Query

This project implements a small retrieval–augmented generation (RAG)
interface for [home-index](https://github.com/nashspence/home-index).
It targets a default setup similar to a MacBook M4 Pro with 24 GB of
RAM but can be configured for other environments.

The UI is built with **Streamlit** and uses **LangChain** with a
`llama-cpp-python` powered Mistral model for question answering. Retrieval
links content chunks
back to their source documents via `ParentDocumentRetriever`. Data is stored in
**Meilisearch** using two indexes (configurable via environment
variables):

- `files` – metadata and plain text content for text search
- `file_chunks` – vectorised content chunks with a reference to the
  `files` index entry they came from

## Features

- Downloads the selected model on first run. When the model path ends with
  `.gguf` it will be loaded using `ChatLlamaCpp` from `llama-cpp-python`.
  ([test](tests/test_llm.py))
- Configurable model and Meilisearch connection via environment
  variables or the Streamlit sidebar. ([test](tests/test_config.py))
- Simple RAG pipeline that searches the `file_chunks` index and feeds the
  results to the model. ([test](tests/test_chain.py))
- Dockerised integration test runs in GitHub Actions.
  It uses `docker compose` to start the Streamlit app and Playwright
  to load the page in a headless browser.
  ([test](tests/test_streamlit_docker.py))

## Installation

```bash
pip install -r requirements.txt
CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install --upgrade \
  --force-reinstall llama-cpp-python
```

## Usage

Start the Streamlit app:

```bash
PYTHONPATH=. python -m streamlit run app.app
```

Use the sidebar to load the desired model (defaults to the
`mistralai/Mistral-7B-v0.1` checkpoint, which requires significant
resources). When a `.gguf` path is provided the model is loaded with
`ChatLlamaCpp`. Enter a question in the text box and the app will
search Meilisearch and generate an answer with sources.

### Configuration

Settings can be overridden with environment variables:

 - `LLM_MODEL_NAME` – model id or path to a `.gguf` file
- `EMBED_MODEL_NAME` – model for generating embeddings
- `MEILI_URL` – URL to the Meilisearch instance
- `MEILI_API_KEY` – optional API key
- `FILES_INDEX` – name of the files index (default: `files`)
- `FILE_CHUNKS_INDEX` – name of the chunk index (default:
  `file_chunks`)

## Tests

Run the automated tests with:

```bash
pytest
```

These tests verify configuration loading, model initialisation and basic
Meilisearch client setup.
