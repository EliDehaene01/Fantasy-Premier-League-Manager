"""Exports a live gameweek's pending proposal (the orchestrator graph's state
held at the human-approval interrupt) as static JSON for the frontend.
Reads the state straight from the Postgres checkpointer - no new run, no
resume - and enriches player ids with names from FPL's public bootstrap.

Usage: python scripts/export_live_gameweek.py 6
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend-export"))

from dotenv import load_dotenv

from export_gameweek import build_gameweek_record, write_gameweek_json
from ingestion.fpl_client import get_bootstrap_static
from orchestrator.graph import build_graph
from orchestrator.run import _thread_id, postgres_checkpointer

load_dotenv()

POSITION_BY_ELEMENT_TYPE = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def main(gw: int) -> None:
    graph = build_graph(checkpointer=postgres_checkpointer())
    state = graph.get_state({"configurable": {"thread_id": _thread_id("live", gw)}}).values
    if not state.get("manage_result"):
        sys.exit(f"no live run with a proposal found for gw{gw}")

    bootstrap = get_bootstrap_static()
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    facts = {
        e["id"]: {"name": e["web_name"], "position": POSITION_BY_ELEMENT_TYPE.get(e["element_type"]), "team": teams.get(e["team"])}
        for e in bootstrap["elements"]
    }
    path = write_gameweek_json(build_gameweek_record(state, player_facts=facts))
    print(f"wrote {path}")


if __name__ == "__main__":
    main(int(sys.argv[1]))
