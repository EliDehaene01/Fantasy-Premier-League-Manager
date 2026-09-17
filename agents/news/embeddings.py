"""Embeddings for the News agent's RAG corpus: text-embedding-3-small on
Microsoft Foundry, via the same shared Foundry client
``shared/agent_service.py`` uses for chat completions (same
``MICROSOFT_FOUNDRY_OPENAI_ENDPOINT`` / ``MICROSOFT_FOUNDRY_KEY`` creds -
no new secret names introduced). A longer default timeout (15s vs the
chat-completion default's 12s) since embedding calls are batched (multiple
texts per request) and can legitimately take longer than a single
completion.
"""

from __future__ import annotations

import os

from shared.agent_service import default_foundry_client

EMBEDDING_MODEL = os.getenv("NEWS_AGENT_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536  # must match news_passages.embedding's VECTOR(1536) in ingestion/db.py


def _default_client():
    return default_foundry_client(timeout=15.0)


def embed_texts(texts: list[str], *, client_factory=_default_client) -> list[list[float]]:
    """Embed a batch of chunks. Returns one 1536-dim vector per input text,
    same order. Raises on failure - unlike the LLM reasoning calls, there is
    no sensible fallback for "couldn't embed this chunk": ingestion should
    stop and retry later, not silently store a passage with no vector (it
    would then never surface in a similarity search, a much quieter failure
    than an ingestion job that visibly stopped).
    """
    if not texts:
        return []
    client = client_factory()
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]
