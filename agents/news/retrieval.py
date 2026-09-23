"""Retrieval: given a player_id, filter to passages already tagged for that
player (entity linking - see ``entity_linking.py``/``ingest.py``) FIRST,
then rank what's left by embedding similarity. Filtering first is what makes
this reliable with common surnames - by the time similarity ranking runs,
every remaining candidate is already confirmed to be about the right
player, not just textually similar to one.

The two corpora stay distinguishable here too: ``corpus`` narrows a call to
"availability-relevant passages only" (team_news) or "general coverage
only" (general_news), or leaves it unset for both - the caller decides,
rather than getting one undifferentiated pool back.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import embeddings

# Two fixed reference queries that stand in for "what is this tier actually
# looking for" - there's no free-text user query here (retrieval is keyed
# by player_id, not a question), so similarity ranks passages against a
# QUESTION appropriate to the job.
AVAILABILITY_QUERY = "Is this player injured, suspended, or otherwise unavailable to play?"
POSITIVE_COVERAGE_QUERY = "Is this player receiving notable positive attention, form, or praise?"

# tier2.py calls retrieve_passages once PER PLAYER with one of exactly these
# two fixed strings - without this cache, a full-pool scan (hundreds of
# players) makes that many identical, real embedding API calls for a query
# that never changes, which is what actually made a real 659-player
# gameweek take minutes instead of seconds (found by running one through
# the deployed cluster). The query text is fixed at import time, never
# user-controlled, so caching every distinct value seen is safe for the
# life of the process - there are only ever two.
_query_embedding_cache: dict[str, list[float]] = {}


def _embed_query(query_text: str) -> list[float]:
    if query_text not in _query_embedding_cache:
        _query_embedding_cache[query_text] = embeddings.embed_texts([query_text])[0]
    return _query_embedding_cache[query_text]


@dataclass
class RetrievedPassage:
    id: int
    corpus: str
    source_url: str
    headline: str | None
    chunk_text: str
    similarity: float


def retrieve_passages(
    conn, player_id: int, *, corpus: str | None = None, query_text: str | None = None, top_k: int = 5
) -> list[RetrievedPassage]:
    """Passages linked to ``player_id``, optionally narrowed to one corpus,
    ranked by cosine similarity to ``query_text`` (defaults to the
    availability query - the more common of the two callers).
    """
    query_text = query_text or AVAILABILITY_QUERY
    query_vector = _embed_query(query_text)

    # `%s::vector` casts, not bare `%s`: the `<=>` operator only has an
    # overload for (vector, vector). A bound parameter with no cast resolves
    # to a plain array/unknown type in this expression context (unlike a
    # straight `INSERT ... VALUES (%s)` into a `vector` column, where
    # Postgres can infer the type from the target column alone) and fails
    # with "operator does not exist: vector <=> numeric[]" - found by
    # actually running this against real data, not assumed.
    sql = """
        SELECT np.id, np.corpus, np.source_url, np.headline, np.chunk_text,
               1 - (np.embedding <=> %s::vector) AS similarity
        FROM news_passages np
        JOIN news_passage_players npp ON npp.passage_id = np.id
        WHERE npp.player_id = %s
    """
    params: list = [query_vector, player_id]
    if corpus:
        sql += " AND np.corpus = %s"
        params.append(corpus)
    sql += " ORDER BY np.embedding <=> %s::vector LIMIT %s"
    params += [query_vector, top_k]

    rows = conn.execute(sql, params).fetchall()
    return [
        RetrievedPassage(
            id=r["id"], corpus=r["corpus"], source_url=r["source_url"],
            headline=r["headline"], chunk_text=r["chunk_text"], similarity=float(r["similarity"]),
        )
        for r in rows
    ]
