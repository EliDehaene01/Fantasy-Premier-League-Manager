"""The missing wiring docs/human_in_the_loop.md step 2 assumed existed:
"something calls POST /run" on the orchestrator once ingestion lands fresh
data. The endpoint itself is real (orchestrator/service.py) - what was
missing is anything in the deployed pipeline actually calling it. This
module is that caller, invoked from ingestion/__main__.py's `auto` mode
right after a real weekly ingestion run, using the same bootstrap-static
response run_weekly() already fetched (no second live API round trip).
"""

from __future__ import annotations

import os

import requests

from .silver import POSITION_BY_ELEMENT_TYPE

ORCHESTRATOR_URL_ENV = "ORCHESTRATOR_URL"
DEFAULT_ORCHESTRATOR_URL = "http://orchestrator:8000"


def build_player_pool(bootstrap: dict, squad_ids: set[int]) -> list[dict]:
    """bootstrap-static's ``elements`` already carries everything a live
    ManagerPlayerFact/PlayerEntry needs (price, position, club) plus the
    live-only availability fields (chance_of_playing_this_round, news)
    ingestion/silver.py never persists - no Postgres join required.
    """
    return [
        {
            "player_id": e["id"],
            "club_id": e["team"],
            "position": POSITION_BY_ELEMENT_TYPE.get(e["element_type"]),
            "price": e["now_cost"] / 10.0,
            "in_current_squad": e["id"] in squad_ids,
            "chance_of_playing_this_round": e.get("chance_of_playing_this_round"),
            "news": e.get("news") or None,
        }
        for e in bootstrap["elements"]
    ]


def trigger_live_run(
    bootstrap: dict,
    gameweek: int,
    squad_ids: set[int],
    bank: float,
    team_value: float,
    *,
    base_url: str | None = None,
    timeout: float = 600.0,
) -> dict:
    """POSTs to the orchestrator's real /run (orchestrator/service.py) -
    the six-specialist debate, Manager, and solver run for real, pausing at
    the human-approval interrupt. ``timeout`` defaults high: News's Tier 2
    does a real per-player RAG round trip and can take minutes for a full
    live pool (see orchestrator/callers.py's own SPECIALIST_TIMEOUT_SECONDS
    for why 30s isn't enough there either).
    """
    base_url = base_url or os.environ.get(ORCHESTRATOR_URL_ENV, DEFAULT_ORCHESTRATOR_URL)
    initial_state = {
        "player_pool": build_player_pool(bootstrap, squad_ids),
        "budget": round(team_value + bank, 1),
        "free_transfers": 1,
        "hit_cost": 4.0,
    }
    resp = requests.post(f"{base_url}/run", json={"gameweek": gameweek, "initial_state": initial_state}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()
