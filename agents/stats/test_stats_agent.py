"""Tests for the Stats agent service.

They cover the three things the task asks for:
  1. ``POST /argue`` returns the shared specialist-agent contract shape.
  2. Recommendations come back sorted by conviction (strongest first).
  3. An empty player pool is handled without crashing.

The LLM call is switched off (``STATS_AGENT_DISABLE_LLM=1``) so the tests are
offline and deterministic - they exercise the templated-fallback path for
``reasoning``, which is enough to check the contract. Prediction logging is
switched off too (``STATS_AGENT_DISABLE_PREDICTION_LOG=1``), so test runs
never write synthetic player ids into the real Postgres predictions_log.
"""

from __future__ import annotations

import os

# Must be set before the service module is imported, so generate_reasoning
# never tries to reach Foundry, and the lifespan handler never opens a real
# Postgres connection, during tests.
os.environ["STATS_AGENT_DISABLE_LLM"] = "1"
os.environ["STATS_AGENT_DISABLE_PREDICTION_LOG"] = "1"

import pytest
from fastapi.testclient import TestClient

from agents.stats.features import build_feature_frame
from agents.stats.service import app


@pytest.fixture(scope="module")
def client():
    # ``with`` triggers the lifespan handler, so the model loads once here.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def real_players():
    """A realistic pool: actual engineered feature rows from a late gameweek.

    Using real rows means the model produces a real spread of predictions, so
    the "sorted by conviction" check is meaningful rather than testing on
    identical dummy inputs.
    """
    frame, feature_cols, _ = build_feature_frame()
    sample = frame[frame["GW"] == 34].head(12).reset_index(drop=True)
    players = []
    for i, row in sample.iterrows():
        players.append(
            {
                "player_id": 1000 + int(i),  # frame has no FPL id; synthesise one
                "name": row["name"],
                "position": row["position"],
                "features": {f: float(row[f]) for f in feature_cols},
            }
        )
    return players


def test_argue_returns_contract_shape(client, real_players):
    resp = client.post("/argue", json={"gameweek": 34, "players": real_players, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()

    # exact top-level shape
    assert set(body) == {"agent", "recommendations", "reasoning"}
    assert body["agent"] == "stats"
    assert isinstance(body["reasoning"], str) and len(body["reasoning"]) > 0

    assert isinstance(body["recommendations"], list)
    assert 1 <= len(body["recommendations"]) <= 5
    for rec in body["recommendations"]:
        assert set(rec) == {"player_id", "conviction", "predicted_points"}
        assert isinstance(rec["player_id"], int)
        assert 0.0 <= rec["conviction"] <= 1.0
        assert isinstance(rec["predicted_points"], (int, float))

    # every recommended id was actually in the request pool
    pool_ids = {p["player_id"] for p in real_players}
    assert {r["player_id"] for r in body["recommendations"]} <= pool_ids


def test_recommendations_sorted_by_conviction(client, real_players):
    resp = client.post("/argue", json={"gameweek": 34, "players": real_players, "top_k": 6})
    recs = resp.json()["recommendations"]

    convictions = [r["conviction"] for r in recs]
    assert convictions == sorted(convictions, reverse=True), convictions
    # the strongest pick is the anchor, so its conviction is 1.0
    assert convictions[0] == pytest.approx(1.0)
    # conviction should track predicted points (it's derived from them)
    points = [r["predicted_points"] for r in recs]
    assert points == sorted(points, reverse=True), points


def test_empty_pool_does_not_crash(client):
    resp = client.post("/argue", json={"gameweek": 20, "players": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["agent"] == "stats"
    assert body["recommendations"] == []
    assert isinstance(body["reasoning"], str) and len(body["reasoning"]) > 0


def test_top_k_is_respected_and_clamped(client, real_players):
    # ask for more than the pool holds -> get the whole pool back, no error
    resp = client.post(
        "/argue", json={"gameweek": 34, "players": real_players[:3], "top_k": 25}
    )
    assert resp.status_code == 200
    assert len(resp.json()["recommendations"]) == 3


def test_health_reports_loaded_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["n_features"] > 0
