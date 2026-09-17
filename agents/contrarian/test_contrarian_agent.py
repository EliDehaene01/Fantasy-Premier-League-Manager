"""Tests for the Contrarian agent service.

Uses an isolated Postgres schema (CONTRARIAN_AGENT_TEST_SCHEMA - same
test-isolation mechanism ingestion's own test suite and the Fixtures agent's
tests use) seeded with synthetic predictions_log/player_gameweek_stats rows.

WHY test_reads_from_predictions_log_not_a_live_stats_call IS REAL EVIDENCE
----------------------------------------------------------------------------
No Stats service is running anywhere in this test process - there is no
mock, no stub, no HTTP server standing in for it. The only source
scoring.rank_players() could possibly get a predicted-points number from is
whatever is seeded directly into the `predictions_log` table below. The
test seeds a specific, distinctive value (13.37) and asserts that EXACT
number comes back on the winning pick: if Contrarian's code took a live-call
shortcut instead, this test would have nothing to talk to and would simply
fail (connection refused / timeout), not silently pass with a different
number. A passing test IS the proof of the architectural claim in
scoring.py's module docstring, not just a shape check.
"""

from __future__ import annotations

import os

os.environ["CONTRARIAN_AGENT_DISABLE_LLM"] = "1"
os.environ["CONTRARIAN_AGENT_TEST_SCHEMA"] = "test_contrarian_agent"

import pytest
from fastapi.testclient import TestClient

from agents.contrarian import scoring
from agents.contrarian.service import app
from ingestion.db import get_connection
from shared.contracts import PlayerEntry

SCHEMA = "test_contrarian_agent"


@pytest.fixture(scope="module")
def seeded_conn():
    conn = get_connection(schema=SCHEMA)
    conn.execute("TRUNCATE predictions_log, player_gameweek_stats")
    now = "2026-01-01T00:00:00Z"
    # player 201: high predicted quality, very low ownership - the genuine
    # differential this agent should surface.
    # player 202: high predicted quality too, but heavily owned - a great
    # player, but NOT a differential; the gap should rank it below 201.
    # player 203: low predicted quality, low ownership - low-owned but not
    # "genuinely differentiated" (no real upside behind it).
    conn.executemany(
        "INSERT INTO predictions_log (player_id, gw, predicted_points, model_version, logged_at) "
        "VALUES (%s,%s,%s,%s,%s)",
        [
            (201, 10, 13.37, "test-v1", now),
            (202, 10, 9.5, "test-v1", now),
            (203, 10, 1.0, "test-v1", now),
        ],
    )
    conn.executemany(
        "INSERT INTO player_gameweek_stats (player_id, gw, selected, source, ingested_at) "
        "VALUES (%s,%s,%s,%s,%s)",
        [
            (201, 10, 2.0, "test", now),
            (202, 10, 45.0, "test", now),
            (203, 10, 3.0, "test", now),
        ],
    )
    conn.commit()
    yield conn
    conn.close()


def _player(pid, name):
    return {"player_id": pid, "name": name, "features": {}}


def test_reads_from_predictions_log_not_a_live_stats_call(seeded_conn):
    """Direct unit test of scoring.rank_players against the seeded
    predictions_log - see this file's module docstring for why a passing
    assertion on the exact seeded number (13.37) is real evidence of the
    data path, not just a plausible-looking one.
    """
    # scoring.rank_players receives PlayerEntry objects (what FastAPI/Pydantic
    # hands the argue handler), not raw dicts - the _player() helper below
    # builds a plain dict for the JSON request body the TestClient-based
    # tests send instead.
    players = [
        PlayerEntry(player_id=201, name="Differential"),
        PlayerEntry(player_id=202, name="Owned Gem"),
        PlayerEntry(player_id=203, name="Nobody Special"),
    ]
    picks = scoring.rank_players(seeded_conn, gameweek=10, players=players, top_k=5)

    by_id = {p.player_id: p for p in picks}
    assert by_id[201].predicted_points == pytest.approx(13.37)
    assert by_id[202].predicted_points == pytest.approx(9.5)
    # the low-ownership, high-quality gap sits above the high-ownership, high-quality one
    assert picks[0].player_id == 201


@pytest.fixture(scope="module")
def client(seeded_conn):
    with TestClient(app) as c:
        yield c


def test_argue_returns_contract_shape(client):
    resp = client.post(
        "/argue",
        json={
            "gameweek": 10,
            "players": [_player(201, "Differential"), _player(202, "Owned Gem"), _player(203, "Nobody Special")],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"agent", "recommendations", "vetoes", "reasoning", "chip_recommendation"}
    assert body["agent"] == "contrarian"
    # Contrarian never populates vetoes - that field belongs to the News agent alone.
    assert body["vetoes"] == []
    # Nor the Chips agent's timing field.
    assert body["chip_recommendation"] is None
    assert isinstance(body["reasoning"], str) and body["reasoning"]

    assert len(body["recommendations"]) == 3
    for rec in body["recommendations"]:
        assert set(rec) == {"player_id", "conviction", "predicted_points"}
        # Contrarian DOES have a real number, unlike Fixtures/Template.
        assert isinstance(rec["predicted_points"], (int, float))
        assert 0.0 <= rec["conviction"] <= 1.0


def test_recommendations_sorted_by_conviction(client):
    resp = client.post(
        "/argue",
        json={
            "gameweek": 10,
            "players": [_player(201, "Differential"), _player(202, "Owned Gem"), _player(203, "Nobody Special")],
            "top_k": 5,
        },
    )
    convictions = [r["conviction"] for r in resp.json()["recommendations"]]
    assert convictions == sorted(convictions, reverse=True)
    assert convictions[0] == pytest.approx(1.0)


def test_empty_pool_does_not_crash(client):
    resp = client.post("/argue", json={"gameweek": 10, "players": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "contrarian"
    assert body["recommendations"] == []
    assert body["vetoes"] == []
    assert isinstance(body["reasoning"], str) and body["reasoning"]


def test_players_with_no_logged_prediction_are_skipped(client):
    """A player Stats never scored this gameweek gets no fabricated
    prediction and no rank - see scoring.py's module docstring.
    """
    resp = client.post(
        "/argue",
        json={"gameweek": 10, "players": [_player(9999, "Unscored Player")], "top_k": 5},
    )
    assert resp.status_code == 200
    assert resp.json()["recommendations"] == []
