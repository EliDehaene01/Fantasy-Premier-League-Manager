"""Tests for frontend-export/export_gameweek.py - pure transform, no DB/HTTP.
"""

from __future__ import annotations

import json

from export_gameweek import build_gameweek_record, write_gameweek_json


def _state(final=False):
    manage_result = {
        "feasible": True,
        "squad": [
            {"player_id": 1, "starting": True},
            {"player_id": 2, "starting": True},
            {"player_id": 3, "starting": False},
        ],
        "captain_id": 1,
        "vice_captain_id": 2,
        "transfers_made": 1,
        "hits_taken": 0,
        "hit_points_cost": 0.0,
        "narration": "Squad finalized.",
    }
    return {
        "gameweek": 5,
        "mode": "live",
        "approval_status": "approved" if final else None,
        "manage_result": manage_result,
        "chip": None,
        "first_round": {
            "stats": {"recommendations": [{"player_id": 1, "conviction": 1.0, "predicted_points": 8.0}], "vetoes": [], "chip_recommendation": None, "reasoning": "form is strong"},
        },
        "reactions": {"stats": "sticking with my pick"},
    }


def test_transcript_recommendations_are_truncated_to_a_short_display_list():
    state = _state()
    state["first_round"]["stats"]["recommendations"] = [
        {"player_id": pid, "conviction": 1.0, "predicted_points": 5.0} for pid in range(200)
    ]
    record = build_gameweek_record(state)
    assert len(record["transcript"][0]["recommendations"]) == 5


def test_player_facts_enrich_squad_and_recommendations():
    facts = {1: {"name": "Test Player", "position": "MID", "team": "Testville"}}
    record = build_gameweek_record(_state(), player_facts=facts)

    squad_entry = next(p for p in record["squad"] if p["player_id"] == 1)
    assert squad_entry["name"] == "Test Player"
    assert squad_entry["position"] == "MID"

    rec = record["transcript"][0]["recommendations"][0]
    assert rec["name"] == "Test Player"


def test_pending_record_has_no_points():
    record = build_gameweek_record(_state())
    assert record["gameweek"] == 5
    assert record["state"] == "pending"
    assert record["points"] is None
    assert record["squad"][0]["actual_points"] is None


def test_final_record_computes_captain_doubled_points():
    record = build_gameweek_record(_state(final=True), actual_points={1: 10.0, 2: 4.0, 3: 100.0})
    assert record["state"] == "final"
    # player 1 (captain, starting): 10*2=20; player 2 (starting): 4; player 3 (bench): excluded despite huge score
    assert record["points"]["gross"] == 24.0
    assert record["points"]["net"] == 24.0


def test_transcript_includes_reaction_text():
    record = build_gameweek_record(_state())
    assert record["transcript"][0]["agent"] == "stats"
    assert record["transcript"][0]["reaction"] == "sticking with my pick"


def test_write_gameweek_json_updates_index(tmp_path):
    record = build_gameweek_record(_state())
    path = write_gameweek_json(record, output_dir=tmp_path)

    assert path.exists()
    assert json.loads(path.read_text())["gameweek"] == 5

    index = json.loads((tmp_path / "index.json").read_text())
    assert index["gameweeks"] == [{"gameweek": 5, "state": "pending"}]

    # re-writing the same gameweek updates in place, not duplicates
    final_record = build_gameweek_record(_state(final=True), actual_points={1: 1.0, 2: 1.0, 3: 1.0})
    write_gameweek_json(final_record, output_dir=tmp_path)
    index = json.loads((tmp_path / "index.json").read_text())
    assert index["gameweeks"] == [{"gameweek": 5, "state": "final"}]
