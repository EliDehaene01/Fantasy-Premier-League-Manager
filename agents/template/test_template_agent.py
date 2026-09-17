"""Tests for the Template agent service.

Uses an isolated Postgres schema (TEMPLATE_AGENT_TEST_SCHEMA - same
test-isolation mechanism ingestion's own test suite and the Fixtures/
Contrarian agents' tests use) seeded with synthetic player_gameweek_stats
rows, so the ownership/transfer-momentum scoring can be verified against
known inputs.
"""

from __future__ import annotations

import os

os.environ["TEMPLATE_AGENT_DISABLE_LLM"] = "1"
os.environ["TEMPLATE_AGENT_TEST_SCHEMA"] = "test_template_agent"

import pytest
from fastapi.testclient import TestClient

from agents.template.service import app
from ingestion.db import get_connection

SCHEMA = "test_template_agent"


@pytest.fixture(scope="module")
def seeded_conn():
    conn = get_connection(schema=SCHEMA)
    conn.execute("TRUNCATE player_gameweek_stats")
    now = "2026-01-01T00:00:00Z"
    # player 301: the true template pick - very high ownership, rising.
    # player 302: high-ish ownership but fading (net transfers out).
    # player 303: barely owned at all - not a template pick.
    conn.executemany(
        "INSERT INTO player_gameweek_stats (player_id, gw, selected, transfers_balance, source, ingested_at) "
        "VALUES (%s,%s,%s,%s,%s,%s)",
        [
            (301, 10, 55.0, 20000, "test", now),
            (302, 10, 40.0, -15000, "test", now),
            (303, 10, 1.5, 500, "test", now),
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
            "players": [_player(301, "Nailed On"), _player(302, "Fading Template"), _player(303, "Obscure")],
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"agent", "recommendations", "vetoes", "reasoning", "chip_recommendation"}
    assert body["agent"] == "template"
    # Template never populates vetoes - that field belongs to the News agent alone.
    assert body["vetoes"] == []
    # Nor the Chips agent's timing field.
    assert body["chip_recommendation"] is None
    assert isinstance(body["reasoning"], str) and body["reasoning"]

    assert len(body["recommendations"]) == 3
    for rec in body["recommendations"]:
        assert set(rec) == {"player_id", "conviction", "predicted_points"}
        # Ownership safety isn't a points prediction - never invented.
        assert rec["predicted_points"] is None
        assert 0.0 <= rec["conviction"] <= 1.0


def test_ranks_by_ownership_with_transfer_momentum_as_a_nudge(client):
    resp = client.post(
        "/argue",
        json={
            "gameweek": 10,
            "players": [_player(303, "Obscure"), _player(301, "Nailed On"), _player(302, "Fading Template")],
            "top_k": 5,
        },
    )
    ranking = [r["player_id"] for r in resp.json()["recommendations"]]
    # highest-owned and rising beats highest-owned-but-fading beats barely-owned
    assert ranking == [301, 302, 303]


def test_recommendations_sorted_by_conviction(client):
    resp = client.post(
        "/argue",
        json={
            "gameweek": 10,
            "players": [_player(301, "Nailed On"), _player(302, "Fading Template"), _player(303, "Obscure")],
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
    assert body["agent"] == "template"
    assert body["recommendations"] == []
    assert body["vetoes"] == []
    assert isinstance(body["reasoning"], str) and body["reasoning"]


def test_health_reports_db_connected(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["db_connected"] is True
