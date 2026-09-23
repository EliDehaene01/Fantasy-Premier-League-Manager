"""Unit tests for ingestion/trigger.py - the previously-missing wiring from
"ingestion finished" to the orchestrator's real POST /run. No live network
call: requests.post is monkeypatched, same convention as the rest of the
ingestion test suite (fpl_client is monkeypatched in test_ingestion.py).
"""

from __future__ import annotations

from ingestion import trigger


def _bootstrap():
    return {
        "elements": [
            {"id": 1, "team": 1, "element_type": 1, "now_cost": 45, "chance_of_playing_this_round": None, "news": ""},
            {"id": 2, "team": 2, "element_type": 3, "now_cost": 75, "chance_of_playing_this_round": 50, "news": "Ankle injury - 50% chance of playing"},
        ]
    }


def test_build_player_pool_carries_availability_fields_and_current_squad_flag():
    pool = trigger.build_player_pool(_bootstrap(), squad_ids={1})
    by_id = {p["player_id"]: p for p in pool}

    assert by_id[1]["in_current_squad"] is True
    assert by_id[1]["price"] == 4.5
    assert by_id[1]["position"] == "GK"
    assert by_id[2]["in_current_squad"] is False
    assert by_id[2]["chance_of_playing_this_round"] == 50
    assert by_id[2]["news"] == "Ankle injury - 50% chance of playing"


def test_trigger_live_run_posts_to_the_orchestrators_run_endpoint(monkeypatch):
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"approval_status": "auto_accepted"}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(trigger.requests, "post", fake_post)
    monkeypatch.setenv(trigger.ORCHESTRATOR_URL_ENV, "http://orchestrator-test:8000")

    result = trigger.trigger_live_run(_bootstrap(), gameweek=5, squad_ids={1}, bank=2.1, team_value=99.5)

    assert result == {"approval_status": "auto_accepted"}
    assert captured["url"] == "http://orchestrator-test:8000/run"
    assert captured["json"]["gameweek"] == 5
    assert captured["json"]["initial_state"]["budget"] == 101.6
    assert len(captured["json"]["initial_state"]["player_pool"]) == 2
