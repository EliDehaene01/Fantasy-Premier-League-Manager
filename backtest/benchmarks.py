"""ARCHITECTURE.md 9 step 3: "Total season points vs. the published FPL
average-manager score and a 'never transfer' baseline."

**Average-manager score: not available for this benchmark, documented
rather than faked.** Checked the actual vaastav archive file listing for
2025-26 (``cleaned_players.csv``, ``fixtures.csv``, ``gws/``,
``player_idlist.csv``, ``players/``, ``players_raw.csv``, ``teams.csv``) -
none of it is a per-gameweek average-entry-score file. The live FPL API's
``bootstrap-static`` has this field, but only for whichever season is
CURRENTLY active - by the time a season is old enough to be "the last
completed season" this project backtests, the live API has moved on to a
newer season and no longer serves it. ``average_manager_total`` below
returns ``None`` with this reason attached rather than a guessed number.

**"Never transfer" baseline**: build a squad once, at ``start_gw``, via the
exact same Manager/solver pipeline as a real gameweek (empty current squad,
full budget) - then never call the solver again. The starting XI and
captain chosen that one time are used, unchanged, for every subsequent
gameweek through ``end_gw``. This is the strict reading of "never
transfer" - no weekly captain reshuffle either, since that's a decision a
manager makes, and "never transfer" is meant to isolate the VALUE of the
transfer/chip machinery specifically, not compare against a strawman that
still gets weekly captaincy optimization for free.
"""

from __future__ import annotations

from dataclasses import dataclass

from agents.stats.features import build_feature_frame
from agents.stats.model_runtime import StatsModel
from data.backtest_seed import seed_backtest_schema
from ingestion.db import get_connection
from manager.reaction import generate_reactions
from manager.schemas import ManageResult
from manager.service import manage as manage_impl
from orchestrator.graph import build_graph
from orchestrator.run import run_backtest_gameweek

from .agents_bridge import SpecialistBridge, disable_all_llm_calls
from .engine import actual_points, build_player_pool


@dataclass
class BenchmarkReport:
    engine_total: float
    never_transfer_total: float
    average_manager_total: float | None
    average_manager_unavailable_reason: str | None


def average_manager_total(start_gw: int, end_gw: int) -> tuple[float | None, str | None]:
    return None, (
        "The vaastav archive has no per-gameweek average-entry-score file for this "
        "season, and the live FPL API only serves this for the currently active season "
        "- by the time a season qualifies as 'the last completed season' this project "
        "backtests, the live number has already rolled over. Not fabricated."
    )


def never_transfer_total(season: str, start_gw: int, end_gw: int) -> float:
    disable_all_llm_calls()
    schema = seed_backtest_schema(season)
    conn = get_connection(schema=schema)
    frame, feature_cols, _ = build_feature_frame()
    bridge = SpecialistBridge(conn, StatsModel())
    graph = build_graph(agent_caller=bridge, manage_caller=manage_impl, react_caller=generate_reactions)

    pool = build_player_pool(conn, frame, feature_cols, start_gw, squad=frozenset())
    initial_state = {"player_pool": pool, "budget": 100.0, "free_transfers": 1, "hit_cost": 4.0}
    result_state = run_backtest_gameweek(start_gw, initial_state, graph=graph)
    manage_result_dict = result_state["manage_result"]
    if not manage_result_dict["feasible"]:
        return 0.0

    manage_result = ManageResult.model_validate(manage_result_dict)
    starting_ids = [s.player_id for s in manage_result.squad if s.starting]
    captain_id = manage_result.captain_id

    total = 0.0
    for gw in range(start_gw, end_gw + 1):
        actual = actual_points(conn, gw, starting_ids)
        gross = sum(actual.values())
        if captain_id is not None:
            gross += actual.get(captain_id, 0.0)  # captain's extra share (doubled)
        total += gross
    return total


def compare(season: str, engine_total: float, start_gw: int, end_gw: int) -> BenchmarkReport:
    avg, reason = average_manager_total(start_gw, end_gw)
    return BenchmarkReport(
        engine_total=engine_total,
        never_transfer_total=never_transfer_total(season, start_gw, end_gw),
        average_manager_total=avg,
        average_manager_unavailable_reason=reason,
    )
