"""Embeddings for the News agent's RAG corpus: text-embedding-3-small on
Microsoft Foundry, via the same OpenAI-compatible client pattern
``agents/stats/reasoning.py`` already uses for chat completions (same
``MICROSOFT_FOUNDRY_OPENAI_ENDPOINT`` / ``MICROSOFT_FOUNDRY_KEY`` creds -
no new secret names introduced).

Known duplication, left as-is: this module's ``_default_client()`` (and its
``_ENDPOINT_ENV``/``_KEY_ENV`` pair) is near-identical to the one in
``agents/stats/reasoning.py`` and ``agents/news/tier2.py`` - same 3-line
OpenAI-SDK client construction, repeated three times. Not extracted into a
shared helper: there's no shared-infra package in this repo yet (each agent
is meant to be its own independently containerized service per CLAUDE.md),
so picking a home for one would be a real architecture decision, not a
mechanical cleanup. Worth revisiting once a third or fourth agent needs the
same pattern.
"""

from __future__ import annotations

import os

EMBEDDING_MODEL = os.getenv("NEWS_AGENT_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = 1536  # must match news_passages.embedding's VECTOR(1536) in ingestion/db.py

_ENDPOINT_ENV = "MICROSOFT_FOUNDRY_OPENAI_ENDPOINT"
_KEY_ENV = "MICROSOFT_FOUNDRY_KEY"


def _default_client():
    from openai import OpenAI

    return OpenAI(api_key=os.environ[_KEY_ENV], base_url=os.environ[_ENDPOINT_ENV], timeout=15.0, max_retries=1)


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
