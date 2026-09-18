"""One-off script: runs a single real backtest gameweek and exports it as
sample static JSON for the frontend (both pending and final states), so
the frontend has genuine data to render against during development -
never fabricated, always a real solver/Manager decision over real
historical data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frontend-export"))

from agents.stats.features import build_feature_frame
from agents.stats.model_runtime import StatsModel
from data.backtest_seed import seed_backtest_schema
from export_gameweek import build_gameweek_record, write_gameweek_json
from ingestion.db import get_connection
from manager.reaction import generate_reactions
from manager.service import manage as manage_impl
from orchestrator.graph import build_graph
from orchestrator.run import run_backtest_gameweek

from backtest.agents_bridge import SpecialistBridge, disable_all_llm_calls, log_predictions, write_team_state
from backtest.engine import actual_points, build_player_pool, _latest_prices
from backtest.state import SeasonState


def _run_one_gameweek(conn, graph, bridge, stats_model, frame, feature_cols, gw: int) -> dict:
    state = SeasonState()
    write_team_state(conn, gw, state.squad, state.bank)
    pool = build_player_pool(conn, frame, feature_cols, gw, state.squad)

    stats_arg = bridge("stats", gw, pool)
    log_predictions(conn, gw, stats_arg, stats_model.model_version)
    conn.commit()

    initial_state = {"player_pool": pool, "budget": 100.0, "free_transfers": 1, "hit_cost": 4.0}
    result_state = run_backtest_gameweek(gw, initial_state, graph=graph)
    result_state["mode"] = "live"  # sample export pretends to be a live gameweek for demo purposes
    return result_state


def _player_facts(conn, player_ids: list[int]) -> dict[int, dict]:
    rows = conn.execute(
        "SELECT p.player_id, p.web_name, p.position, t.name AS team "
        "FROM players p LEFT JOIN teams t ON t.team_id = p.team_id "
        "WHERE p.player_id = ANY(%s)",
        (player_ids,),
    ).fetchall()
    return {r["player_id"]: {"name": r["web_name"], "position": r["position"], "team": r["team"]} for r in rows}


def main(pending_gw: int = 10, final_gw: int = 9) -> None:
    """Exports two DIFFERENT real gameweeks - one left pending, one taken to
    final - so the frontend has a genuine example of each state to render
    (ARCHITECTURE.md 8a: "two views of the same underlying gameweek record").
    """
    disable_all_llm_calls()
    schema = seed_backtest_schema("2025-26")
    conn = get_connection(schema=schema)
    frame, feature_cols, _ = build_feature_frame()
    stats_model = StatsModel()
    bridge = SpecialistBridge(conn, stats_model)
    graph = build_graph(agent_caller=bridge, manage_caller=manage_impl, react_caller=generate_reactions)

    pending_state = _run_one_gameweek(conn, graph, bridge, stats_model, frame, feature_cols, pending_gw)
    pending_ids = [s["player_id"] for s in pending_state["manage_result"]["squad"]]
    pending_record = build_gameweek_record(pending_state, player_facts=_player_facts(conn, pending_ids))
    print(f"wrote pending: {write_gameweek_json(pending_record)}")

    final_state = _run_one_gameweek(conn, graph, bridge, stats_model, frame, feature_cols, final_gw)
    squad_ids = [s["player_id"] for s in final_state["manage_result"]["squad"]]
    actual = actual_points(conn, final_gw, squad_ids)
    final_record = build_gameweek_record(final_state, actual_points=actual, player_facts=_player_facts(conn, squad_ids))
    print(f"wrote final: {write_gameweek_json(final_record)}")


if __name__ == "__main__":
    main()
