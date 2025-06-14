# Home Index RAG Query

This project implements a small retrieval–augmented generation (RAG)
interface for [home-index](https://github.com/nashspence/home-index).
It targets a default setup similar to a MacBook M4 Pro with 24 GB of
RAM but can be configured for other environments.

The UI is built with **Streamlit** and uses **LangChain** with a
HuggingFace model for question answering. Retrieval links content chunks
back to their source documents via `ParentDocumentRetriever`. Data is stored in
**Meilisearch** using two indexes (configurable via environment
variables):

- `files` – metadata and plain text content for text search
- `file_chunks` – vectorised content chunks with a reference to the
  `files` index entry they came from

## Features

- Downloads the selected language model from HuggingFace on first run.
- Configurable model and Meilisearch connection via environment
  variables or the Streamlit sidebar.
- Simple RAG pipeline that searches the `file_chunks` index and feeds the
  results to the model.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the Streamlit app:

```bash
streamlit run app/app.py
```

Use the sidebar to load the desired HuggingFace model (defaults to the
`mistralai/Mistral-7B-v0.1` checkpoint, which requires significant
resources). Enter a question in the text box and the app will
search Meilisearch and generate an answer with sources.

### Configuration

Settings can be overridden with environment variables:

- `LLM_MODEL_NAME` – HuggingFace model id
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
