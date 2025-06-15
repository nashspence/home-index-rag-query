from __future__ import annotations

import json
from datetime import datetime

from geopy.geocoders import Nominatim

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from .llm import load_llm
from .database import search_index
from .config import settings


class FileDocument(BaseModel):
    """Representation of a file stored in Meilisearch."""

    file_type: Optional[str] = Field(
        default=None,
        description="Single word file class such as 'video', 'audio', 'text', or 'archive'",
    )
    path: Optional[str] = Field(
        default=None,
        description="Snippet of the canonical file path to match",
    )
    ctime: Optional[str] = Field(
        default=None,
        description="Creation time or interval in human readable form",
    )
    mtime: Optional[str] = Field(
        default=None,
        description="Modification time or interval in human readable form",
    )
    content: Optional[str] = Field(
        default=None, description="Content snippet used for vector search"
    )
    location: Optional[str] = Field(
        default=None,
        description="Named location associated with the document",
    )
    radius_km: Optional[float] = Field(
        default=None,
        description="Search radius in kilometres around the location",
    )


def _parse_date(value: str) -> float | tuple[float, float] | None:
    """Return a UNIX timestamp or interval for a human-readable date string."""
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        pass

    try:
        from timefhuman.main import timefhuman

        dts = timefhuman(value)
        if not dts:
            return None

        dt0 = dts[0]
        # Handle explicit range e.g. "3p-4p" which returns [(start, end)]
        if isinstance(dt0, tuple) and len(dt0) == 2:
            start, end = dt0
            if hasattr(start, "timestamp") and hasattr(end, "timestamp"):
                return start.timestamp(), end.timestamp()

        # Handle "between" expressions which return [start, end]
        if len(dts) == 2 and all(hasattr(x, "timestamp") for x in dts):
            return dts[0].timestamp(), dts[1].timestamp()

        # Default to first datetime
        if hasattr(dt0, "timestamp"):
            return dt0.timestamp()
    except Exception:
        pass

    return None


_geolocator: Nominatim | None = None


def _geocode(name: str) -> tuple[float | None, float | None]:
    """Look up a location name and return ``(lat, lon)`` coordinates."""
    global _geolocator
    if _geolocator is None:
        _geolocator = Nominatim(user_agent="home-index-rag-query")
    try:
        location = _geolocator.geocode(name)
    except Exception:
        return None, None
    if location is None:
        return None, None
    return location.latitude, location.longitude


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
    search_terms = None
    params: dict = {}
    if isinstance(result, FileDocument):
        parts = [
            result.path,
            result.file_type,
            result.location,
            result.ctime,
            result.mtime,
            result.content,
        ]
        search_terms = " ".join(str(p) for p in parts if p)

        filters = []
        if result.file_type:
            filters.append(f'CONTAINS(file_type, "{result.file_type}")')
        if result.path:
            filters.append(f'CONTAINS(path, "{result.path}")')
        if result.ctime:
            ts = _parse_date(result.ctime)
            if isinstance(ts, tuple):
                start, end = ts
                filters.append(f'ctime >= {int(start)} AND ctime <= {int(end)}')
            elif ts is not None:
                filters.append(f'ctime >= {int(ts)}')
        if result.mtime:
            ts = _parse_date(result.mtime)
            if isinstance(ts, tuple):
                start, end = ts
                filters.append(f'mtime >= {int(start)} AND mtime <= {int(end)}')
            elif ts is not None:
                filters.append(f'mtime >= {int(ts)}')
        if result.location and result.radius_km:
            lat, lon = _geocode(result.location)
            if lat is not None and lon is not None:
                m = int(result.radius_km * 1000)
                filters.append(f'_geoRadius({lat}, {lon}, {m})')
        if filters:
            params["filter"] = " AND ".join(filters)

    return search_index(
        settings.files_index,
        search_terms or query,
        limit=5,
        **params,
    )
