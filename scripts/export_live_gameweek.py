"""Exports a live gameweek's pending proposal (the orchestrator graph's state
held at the human-approval interrupt) as static JSON for the frontend.
Reads the state straight from the Postgres checkpointer - no new run, no
resume - and enriches player ids with names from FPL's public bootstrap.

Usage: python scripts/export_live_gameweek.py [gw]   # default: next deadline's gameweek
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO_ROOT), str(REPO_ROOT / "frontend-export")]

from dotenv import load_dotenv

from export_gameweek import build_gameweek_record, write_gameweek_json
from ingestion.deadline import next_gameweek_id
from ingestion.fpl_client import get_bootstrap_static
from orchestrator.graph import build_graph
from orchestrator.run import _thread_id, postgres_checkpointer

load_dotenv()

POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def main(gw: int | None = None) -> None:
    bootstrap = get_bootstrap_static()
    gw = gw or next_gameweek_id(bootstrap)
    if gw is None:
        print("no upcoming gameweek - nothing to export")
        return
    graph = build_graph(checkpointer=postgres_checkpointer())
    state = graph.get_state({"configurable": {"thread_id": _thread_id("live", gw)}}).values
    if not state.get("manage_result"):
        # Normal on days the deadline check skipped - not an error.
        print(f"no live run with a proposal for gw{gw} yet - nothing to export")
        return

    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    facts = {
        e["id"]: {"name": e["web_name"], "position": POSITION_BY_ELEMENT_TYPE.get(e["element_type"]), "team": teams.get(e["team"])}
        for e in bootstrap["elements"]
    }
    path = write_gameweek_json(build_gameweek_record(state, player_facts=facts))
    print(f"wrote {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
