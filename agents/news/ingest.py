"""Ingestion for the News agent's RAG corpus: scrape both corpora, link
entities, chunk, embed, store - tagged by corpus (see ``ingestion/db.py``'s
``news_passages.corpus`` check constraint) so retrieval can ask for one
corpus specifically (see ``retrieval.py``).

Requires pgvector to actually be installed on the Postgres server (see
``ingestion.db.ensure_pgvector_schema`` - kept as its own step, not part of
every connection's schema bootstrap, exactly because it's an optional
native extension, not guaranteed present).

Run: ``python -m agents.news.ingest``
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from ingestion.db import ensure_pgvector_schema, get_connection

from . import embeddings, scraper_general_news, scraper_team_news
from .entity_linking import Player, link_players

logger = logging.getLogger("news_agent.ingest")


def _load_players(conn) -> list[Player]:
    rows = conn.execute(
        "SELECT p.player_id, p.web_name, t.name AS team_name "
        "FROM players p LEFT JOIN teams t ON t.team_id = p.team_id"
    ).fetchall()
    return [Player(player_id=r["player_id"], web_name=r["web_name"], team_name=r["team_name"] or "") for r in rows]


def _store_passages(conn, *, corpus: str, chunks: list[tuple[str, str, str | None]], players: list[Player]) -> int:
    """``chunks``: list of (source_url, headline, chunk_text). Links
    entities, embeds everything in one batch call (cheaper than one API
    call per chunk), then writes each passage + its player links.
    """
    linked: list[tuple[str, str, str, list[int]]] = []
    for source_url, headline, chunk_text in chunks:
        link = link_players(chunk_text, players)
        if link.ambiguous:
            logger.warning(
                "ambiguous surname(s) in a %s chunk (%r): %s", corpus, chunk_text[:80], link.ambiguous
            )
        if not link.player_ids:
            continue  # nothing to attach this passage to - skip rather than store an orphan chunk
        linked.append((source_url, headline, chunk_text, link.player_ids))

    if not linked:
        return 0

    vectors = embeddings.embed_texts([c[2] for c in linked])
    now = datetime.now(timezone.utc).isoformat()
    n = 0
    for (source_url, headline, chunk_text, player_ids), vector in zip(linked, vectors):
        row = conn.execute(
            """
            INSERT INTO news_passages (corpus, source_url, headline, published_at, chunk_text, embedding, scraped_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (corpus, source_url, headline, None, chunk_text, vector, now),
        ).fetchone()
        conn.executemany(
            "INSERT INTO news_passage_players (passage_id, player_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            [(row["id"], pid) for pid in player_ids],
        )
        n += 1
    conn.commit()
    return n


def ingest_team_news(conn, players: list[Player], *, headless: bool = True) -> int:
    rows = scraper_team_news.scrape_team_news(headless=headless)
    chunks = [
        (scraper_team_news.URL, r.name, f"{r.name} ({r.team}, {r.position}): {r.news_text}")
        for r in rows
    ]
    return _store_passages(conn, corpus="team_news", chunks=chunks, players=players)


def ingest_general_news(conn, players: list[Player], *, headless: bool = True) -> int:
    articles = scraper_general_news.scrape_general_news(headless=headless)
    chunks: list[tuple[str, str, str]] = []
    for art in articles:
        # Chunk per-article-section: the headline is always its own chunk -
        # it is frequently the whole signal on its own ("Maresca offers
        # latest on O'Reilly and Doku" - see scraper_general_news.py's
        # docstring for why the body isn't always fetchable). A fetched body
        # is split into paragraph-length sections rather than embedded as
        # one giant blob, so retrieval can surface the specific relevant
        # sentence rather than a whole article's worth of text.
        chunks.append((art.url, art.headline, art.headline))
        if art.body_text:
            for para in art.body_text.split("\n"):
                para = para.strip()
                if len(para) > 40:
                    chunks.append((art.url, art.headline, para))
    return _store_passages(conn, corpus="general_news", chunks=chunks, players=players)


def run_ingest(conn=None, *, headless: bool = True) -> dict:
    own_conn = conn is None
    conn = conn or get_connection()
    try:
        if not ensure_pgvector_schema(conn):
            raise RuntimeError(
                "pgvector extension is not installed on this Postgres instance - "
                "install it (see docs/news_agent.md's 'Vector store' section) before "
                "running News agent ingestion."
            )
        players = _load_players(conn)
        return {
            "team_news_passages": ingest_team_news(conn, players, headless=headless),
            "general_news_passages": ingest_general_news(conn, players, headless=headless),
        }
    finally:
        if own_conn:
            conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_ingest()))
