"""Contract-shape tests for the News agent service.

``NEWS_AGENT_DISABLE_LLM=1`` (set below, before ``service`` is imported -
same discipline as ``test_stats_agent.py``'s ``STATS_AGENT_DISABLE_LLM``)
keeps these deterministic and offline regardless of whether pgvector/real
ingested data happen to be present: both tier2.py functions check that flag
BEFORE touching retrieval or the DB, so with it set, /argue never makes a
real Foundry call or depends on what's actually been scraped. Without this
flag, these tests silently made real LLM calls once pgvector went live and
real data existed - found by actually running against real data, not
assumed to be a one-off assertion issue.
"""

from __future__ import annotations

import os

os.environ["NEWS_AGENT_DISABLE_LLM"] = "1"

from fastapi.testclient import TestClient

from agents.news.service import app


def _player(player_id, chance=None, news=None, name=None):
    return {
        "player_id": player_id, "name": name or f"Player {player_id}",
        "chance_of_playing_this_round": chance, "news": news, "features": {},
    }


def test_argue_returns_extended_contract_shape():
    with TestClient(app) as client:
        resp = client.post(
            "/argue",
            json={
                "gameweek": 10,
                "players": [
                    _player(1, chance=0, news="Hamstring injury"),  # clear OUT, Tier 1 only
                    _player(2, chance=100),                          # clear FIT, Tier 1 only
                ],
            },
        )
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"agent", "recommendations", "vetoes", "reasoning", "chip_recommendation"}
    assert body["agent"] == "news"
    # News never populates the Chips agent's timing field either.
    assert body["chip_recommendation"] is None
    assert isinstance(body["reasoning"], str) and body["reasoning"]
    assert isinstance(body["recommendations"], list)

    assert isinstance(body["vetoes"], list)
    assert len(body["vetoes"]) == 2
    for veto in body["vetoes"]:
        assert set(veto) == {"player_id", "status", "confidence", "grounding_snippet"}
        assert veto["status"] in ("OUT", "DOUBT", "FIT")
        assert 0.0 <= veto["confidence"] <= 1.0

    by_id = {v["player_id"]: v for v in body["vetoes"]}
    assert by_id[1]["status"] == "OUT"
    assert by_id[2]["status"] == "FIT"


def test_recommendation_predicted_points_is_optional_on_the_wire():
    """News's own recommendations (when Tier 2 produces any) never carry a
    real predicted_points number - confirms the shared schema's Optional
    change actually round-trips through JSON as null, not a required field
    that would reject the response."""
    with TestClient(app) as client:
        resp = client.post("/argue", json={"gameweek": 10, "players": [_player(1, chance=100)]})
    assert resp.status_code == 200
    # No recommendations expected here (Tier 2 has no real corpus in this
    # test run), which is itself the conservative, correct behaviour - see
    # tier2.py's "return none rather than guessing" fallback.
    assert resp.json()["recommendations"] == []


def test_empty_pool_does_not_crash():
    with TestClient(app) as client:
        resp = client.post("/argue", json={"gameweek": 10, "players": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["vetoes"] == []
    assert body["recommendations"] == []
