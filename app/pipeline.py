from __future__ import annotations

import json

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from .llm import load_llm
from .database import search_index
from .config import settings


class FileDocument(BaseModel):
    """Representation of a file stored in Meilisearch."""

    id: str = Field(description="xxh64 hexadecimal digest")
    type: str = Field(description="MIME type")
    size: int = Field(description="File size in bytes")
    paths: dict[str, float]
    copies: int
    mtime: float
    next: str = ""
    lat: Optional[float] = Field(default=None, description="Latitude")
    lon: Optional[float] = Field(default=None, description="Longitude")


SCHEMA = FileDocument.model_json_schema()


PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Return a JSON object matching this schema:\n" + json.dumps(SCHEMA),
        ),
        ("human", "{query}"),
    ]
)


def query_pipeline(query: str):
    """Generate a structured query from text and search Meilisearch."""
    llm = load_llm(settings.llm_model_name)
    parser = JsonOutputParser(pydantic_object=FileDocument)
    chain = PROMPT | llm | parser
    result = chain.invoke({"query": query})
    if isinstance(result, FileDocument):
        q = result.id
    else:
        q = getattr(result, "id", None) if isinstance(result, dict) else None
    return search_index(settings.files_index, q or query, limit=5)
