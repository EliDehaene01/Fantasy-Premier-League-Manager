"""One-off backfill: real historical gameweeks the agent system wasn't
running for yet, exported in the same frontend contract as an agent-driven
gameweek but with source="manual" and an empty transcript - the frontend
(App.jsx/Transcript.jsx) checks that field to show a plain note instead of
an agent debate that never happened. Real account data only (FPL's public
API via FPL_TEAM_ID), no fabricated reasoning.

Usage: python scripts/backfill_history.py 1 5   # backfill gw1..gw5 inclusive
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend-export"))

from dotenv import load_dotenv

from export_gameweek import write_gameweek_json
from ingestion.fpl_client import get_bootstrap_static, get_entry_history, get_entry_picks, get_event_live

load_dotenv()

CHIP_NAME_MAP = {"wildcard": "wildcard", "3xc": "triple_captain", "bboost": "bench_boost", "freehit": "free_hit"}
POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def build_manual_record(team_id: int, gw: int, history_by_gw: dict, bootstrap_elements: dict) -> dict:
    picks_resp = get_entry_picks(team_id, gw)
    live = get_event_live(gw)
    live_points = {row["id"]: row["stats"]["total_points"] for row in live["elements"]}
    hist = history_by_gw[gw]

    players = []
    gross = 0.0
    captain_id = vice_id = None
    for pick in picks_resp["picks"]:
        pid = pick["element"]
        el = bootstrap_elements.get(pid, {})
        pts = live_points.get(pid, 0)
        gross += pts * pick["multiplier"]
        if pick["is_captain"]:
            captain_id = pid
        if pick["is_vice_captain"]:
            vice_id = pid
        players.append(
            {
                "player_id": pid,
                "starting": pick["multiplier"] > 0 or pick["position"] <= 11,
                "actual_points": pts,
                "name": el.get("web_name"),
                "position": POSITION_BY_ELEMENT_TYPE.get(el.get("element_type")),
                "team": el.get("team_name"),
            }
        )

    raw_chip = picks_resp.get("active_chip")
    hit_cost = hist["event_transfers_cost"]
    return {
        "gameweek": gw,
        "mode": "live",
        "state": "final",
        "source": "manual",
        "approval_status": None,
        "squad": players,
        "captain_id": captain_id,
        "vice_captain_id": vice_id,
        "chip": CHIP_NAME_MAP.get(raw_chip),
        "transfers_made": hist["event_transfers"],
        "hits_taken": hit_cost // 4 if hit_cost else 0,
        "hit_points_cost": hit_cost,
        "narration": None,
        "transcript": [],
        "points": {"gross": gross, "net": gross - hit_cost},
    }


def main(from_gw: int, to_gw: int) -> None:
    team_id = int(os.environ["FPL_TEAM_ID"])
    bootstrap = get_bootstrap_static()
    elements = {e["id"]: {**e, "team_name": next((t["name"] for t in bootstrap["teams"] if t["id"] == e["team"]), None)} for e in bootstrap["elements"]}
    history_by_gw = {ev["event"]: ev for ev in get_entry_history(team_id)["current"]}

    for gw in range(from_gw, to_gw + 1):
        record = build_manual_record(team_id, gw, history_by_gw, elements)
        path = write_gameweek_json(record)
        print(f"wrote gw{gw}: {path} (net {record['points']['net']} pts, chip={record['chip']})")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
