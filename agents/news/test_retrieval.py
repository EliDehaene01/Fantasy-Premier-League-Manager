"""Unit test for the fix to agents/news/retrieval.py's real-world latency
bug: retrieve_passages() was calling embeddings.embed_texts fresh on every
invocation, even though tier2.py only ever passes one of two FIXED query
strings - a full-pool scan (hundreds of players) made that many identical,
redundant real embedding API calls. Running a real gameweek through the
deployed cluster showed this taking minutes; this test proves the fix
(_query_embedding_cache) actually eliminates the redundancy, not just that
retrieve_passages still returns the right shape.
"""

from __future__ import annotations

from agents.news import retrieval


class _FakeCursor:
    def fetchall(self):
        return []


class _FakeConn:
    def execute(self, sql, params):
        return _FakeCursor()


def test_repeated_calls_with_the_same_query_text_embed_only_once(monkeypatch):
    retrieval._query_embedding_cache.clear()
    calls = []

    def fake_embed_texts(texts, **kwargs):
        calls.append(texts)
        return [[0.1, 0.2, 0.3]]

    monkeypatch.setattr(retrieval.embeddings, "embed_texts", fake_embed_texts)

    conn = _FakeConn()
    for player_id in range(1, 660):  # the real-world shape: a full 659-player pool
        retrieval.retrieve_passages(conn, player_id, query_text=retrieval.POSITIVE_COVERAGE_QUERY)

    assert len(calls) == 1  # not 659 - the whole point of the cache


def test_different_query_texts_each_still_get_embedded(monkeypatch):
    retrieval._query_embedding_cache.clear()
    calls = []
    monkeypatch.setattr(retrieval.embeddings, "embed_texts", lambda texts, **kwargs: calls.append(texts) or [[0.0]])

    conn = _FakeConn()
    retrieval.retrieve_passages(conn, 1, query_text=retrieval.AVAILABILITY_QUERY)
    retrieval.retrieve_passages(conn, 2, query_text=retrieval.POSITIVE_COVERAGE_QUERY)

    assert len(calls) == 2  # two distinct queries - the cache dedupes by text, not blanket-skips embedding
