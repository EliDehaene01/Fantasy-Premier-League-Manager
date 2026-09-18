"""ARCHITECTURE.md 8a: exports one gameweek's record as static JSON for the
frontend - the only thing the frontend ever reads (CLAUDE.md hard
constraint: "The frontend never talks to Postgres or any backend, directly
or indirectly"). Two data drops per gameweek, same underlying shape:

  * pending: right after the Manager's proposal is ready - the orchestrator
    graph's state at (or just past) the human-approval interrupt.
  * final: once real results land - the same record plus each player's
    actual points and the season/backtest benchmark to show it against.

Input is a GraphState-shaped dict (orchestrator/state.py) - whatever
orchestrator/run.py's functions return, live or backtest, so this module
has no opinion about where the state came from.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "frontend" / "public" / "data"


def build_gameweek_record(
    state: dict[str, Any],
    actual_points: dict[int, float] | None = None,
    player_facts: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Pure transform: GraphState dict -> the frontend's JSON contract.
    ``actual_points`` (player_id -> points scored) is None for the pending
    state, populated once the gameweek has been played for the final state.
    ``player_facts`` (player_id -> {"name", "position", "team"}) is optional
    display enrichment - showing a bare numeric id in the UI isn't usable,
    but this module has no DB access of its own (frontend-export stays a
    pure transform), so the caller looks these up and passes them in.
    """
    manage_result = state.get("manage_result") or {}
    squad = manage_result.get("squad", [])
    is_final = actual_points is not None
    player_facts = player_facts or {}

    players = []
    gross_points = 0.0
    for pick in squad:
        pid = pick["player_id"]
        pts = actual_points.get(pid) if actual_points else None
        if pts is not None:
            multiplier = 1 if pick["starting"] else 0
            if pid == manage_result.get("captain_id"):
                multiplier *= 2
            gross_points += pts * multiplier
        players.append({"player_id": pid, "starting": pick["starting"], "actual_points": pts, **player_facts.get(pid, {})})

    def _with_name(entry: dict) -> dict:
        return {**entry, "name": player_facts.get(entry["player_id"], {}).get("name")}

    # The graph calls each agent with a large top_k so the Manager/solver see
    # the full affordable candidate pool (orchestrator/callers.py's own
    # docstring on this), but the DEBATE TRANSCRIPT was always meant to be a
    # short list, not the whole player universe (see PlayerEntry/ArgueRequest
    # in shared/contracts.py: "The debate wants a short list"). Truncating
    # here, at the display boundary, rather than changing how any agent is
    # actually called - the Manager's own aggregation still sees everything.
    TRANSCRIPT_DISPLAY_LIMIT = 5

    transcript = [
        {
            "agent": name,
            "recommendations": [_with_name(r) for r in arg.get("recommendations", [])[:TRANSCRIPT_DISPLAY_LIMIT]],
            "vetoes": [_with_name(v) for v in arg.get("vetoes", [])[:TRANSCRIPT_DISPLAY_LIMIT]],
            "chip_recommendation": arg.get("chip_recommendation"),
            "reasoning": arg.get("reasoning", ""),
            "reaction": (state.get("reactions") or {}).get(name),
        }
        for name, arg in (state.get("first_round") or {}).items()
    ]

    return {
        "gameweek": state.get("gameweek"),
        "mode": state.get("mode"),
        "state": "final" if is_final else "pending",
        "approval_status": state.get("approval_status"),
        "squad": players,
        "captain_id": manage_result.get("captain_id"),
        "vice_captain_id": manage_result.get("vice_captain_id"),
        "chip": state.get("chip"),
        "transfers_made": manage_result.get("transfers_made"),
        "hits_taken": manage_result.get("hits_taken"),
        "hit_points_cost": manage_result.get("hit_points_cost"),
        "narration": manage_result.get("narration"),
        "transcript": transcript,
        "points": {"gross": gross_points, "net": gross_points - (manage_result.get("hit_points_cost") or 0)} if is_final else None,
    }


def write_gameweek_json(record: dict[str, Any], output_dir: Path = OUTPUT_DIR) -> Path:
    """Writes ``gw<N>.json`` and refreshes ``index.json`` (the list of
    gameweeks the frontend fetches to know what's available - the frontend
    never lists a directory, it only ever fetches files it already knows
    the name of).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    gw = record["gameweek"]
    path = output_dir / f"gw{gw}.json"
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    index_path = output_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"gameweeks": []}
    entry = {"gameweek": gw, "state": record["state"]}
    index["gameweeks"] = [e for e in index["gameweeks"] if e["gameweek"] != gw] + [entry]
    index["gameweeks"].sort(key=lambda e: e["gameweek"])
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    return path
