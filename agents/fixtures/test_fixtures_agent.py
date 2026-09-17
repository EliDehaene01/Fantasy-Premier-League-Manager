"""Tests for the Fixtures agent service.

Uses an isolated Postgres schema (FIXTURES_AGENT_TEST_SCHEMA - same
test-isolation mechanism ingestion's own test suite uses) seeded with
synthetic teams/players/fixtures, so the FDR/blank/double-gameweek scoring
can be verified against known inputs rather than whatever real fixtures
happen to be in the database when tests run.

The LLM is disabled (FIXTURES_AGENT_DISABLE_LLM=1), same discipline as
Stats/News's own test suites - offline, deterministic, exercises the
templated-fallback reasoning path.
"""

from __future__ import annotations

import os

os.environ["FIXTURES_AGENT_DISABLE_LLM"] = "1"
os.environ["FIXTURES_AGENT_TEST_SCHEMA"] = "test_fixtures_agent"

import pytest
from fastapi.testclient import TestClient

from agents.fixtures.service import app
from ingestion.db import get_connection

SCHEMA = "test_fixtures_agent"


@pytest.fixture(scope="module")
def seeded_conn():
    conn = get_connection(schema=SCHEMA)
    conn.execute("TRUNCATE players, teams, fixtures")
    now = "2026-01-01T00:00:00Z"
    conn.executemany(
        "INSERT INTO teams (team_id, name, short_name, strength, updated_at) VALUES (%s,%s,%s,%s,%s)",
        [(1, "Easytown", "EZT", 3, now), (2, "Hardrock", "HRK", 3, now), (3, "Nobody FC", "NBD", 3, now)],
    )
    conn.executemany(
        "INSERT INTO players (player_id, web_name, team_id, position, updated_at) VALUES (%s,%s,%s,%s,%s)",
        [
            (101, "Easy Player", 1, "MID", now),
            (102, "Hard Player", 2, "MID", now),
            (103, "No Fixture Player", 3, "MID", now),
        ],
    )
    conn.executemany(
        """INSERT INTO fixtures
           (fixture_id, gw, team_h, team_a, team_h_difficulty, team_a_difficulty, finished, updated_at)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)""",
        [
            # team 1 (Easytown): a double gameweek 10, both fixtures easy (FDR 2).
            (9001, 10, 1, 9, 2, 4, 0, now),
            (9002, 10, 1, 8, 2, 4, 0, now),
            # team 2 (Hardrock): blank at gameweek 10, one hard (FDR 5) fixture at gameweek 11.
            (9003, 11, 2, 9, 5, 1, 0, now),
            # team 3 (Nobody FC): no fixture rows at all in this window - a genuine blank window.
        ],
    )
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def client(seeded_conn):
    with TestClient(app) as c:
        yield c


def _player(pid, name):
    return {"player_id": pid, "name": name, "features": {}}


def test_argue_returns_contract_shape(client):
    resp = client.post(
        "/argue",
        json={
            "gameweek": 10,
            "players": [_player(101, "Easy Player"), _player(102, "Hard Player"), _player(103, "No Fixture Player")],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"agent", "recommendations", "vetoes", "reasoning", "chip_recommendation"}
    assert body["agent"] == "fixtures"
    # Fixtures never populates vetoes - that field belongs to the News agent alone.
    assert body["vetoes"] == []
    # Nor the Chips agent's timing field.
    assert body["chip_recommendation"] is None
    assert isinstance(body["reasoning"], str) and body["reasoning"]

    assert len(body["recommendations"]) == 3
    for rec in body["recommendations"]:
        assert set(rec) == {"player_id", "conviction", "predicted_points"}
        # Fixture favorability isn't a points prediction - never invented.
        assert rec["predicted_points"] is None
        assert 0.0 <= rec["conviction"] <= 1.0


def test_double_gameweek_beats_blank_gameweek(client):
    """A team with an easy double gameweek should rank above a team with a
    blank gameweek followed by one hard fixture - the core signal this
    agent exists to surface.
    """
    resp = client.post(
        "/argue",
        json={
            "gameweek": 10,
            "players": [_player(102, "Hard Player"), _player(101, "Easy Player"), _player(103, "No Fixture Player")],
            "top_k": 5,
        },
    )
    recs = resp.json()["recommendations"]
    ranking = [r["player_id"] for r in recs]
    assert ranking == [101, 102, 103]
    # the strongest pick anchors conviction at 1.0, same convention as Stats
    assert recs[0]["conviction"] == pytest.approx(1.0)
    # a genuinely fixtureless team scores at the floor
    assert recs[-1]["conviction"] == pytest.approx(0.0)


def test_recommendations_sorted_by_conviction(client):
    resp = client.post(
        "/argue",
        json={
            "gameweek": 10,
            "players": [_player(101, "Easy Player"), _player(102, "Hard Player"), _player(103, "No Fixture Player")],
            "top_k": 5,
        },
    )
    convictions = [r["conviction"] for r in resp.json()["recommendations"]]
    assert convictions == sorted(convictions, reverse=True)


def test_empty_pool_does_not_crash(client):
    resp = client.post("/argue", json={"gameweek": 10, "players": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "fixtures"
    assert body["recommendations"] == []
    assert body["vetoes"] == []
    assert isinstance(body["reasoning"], str) and body["reasoning"]


def test_health_reports_db_connected(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["db_connected"] is True
