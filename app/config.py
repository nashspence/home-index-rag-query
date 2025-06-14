from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model_name: str = "mistralai/Mistral-7B-v0.1"
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    meili_url: str = "http://localhost:7700"
    meili_api_key: str | None = None
    files_index: str = "files"
    file_chunks_index: str = "file_chunks"
    files_domain: str = "http://localhost"


settings = Settings()
