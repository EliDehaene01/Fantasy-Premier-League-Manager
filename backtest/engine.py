"""ARCHITECTURE.md 9's walk-forward loop: seeds one season's archive, then
steps gameweek by gameweek through the SAME orchestrator graph live mode
uses (backtest branch: no human-approval pause - ARCHITECTURE.md 6c),
scoring each proposal against actual results.

**No-lookahead, concretely**: each gameweek's player_pool carries only that
gameweek's own already-lagged feature row (features/engineering.py's
``.shift(1)`` discipline - a row for GW N structurally cannot see GW N's
own outcome) and only price/ownership known "at or before" that gameweek
(``player_gameweek_stats.gw <= this gameweek``, the same bound every
DB-backed agent's own scoring.py already enforces). Nothing in this module
reaches forward past the gameweek being played.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from agents.stats.features import build_feature_frame
from agents.stats.model_runtime import StatsModel
from data.backtest_seed import seed_backtest_schema
from ingestion.db import get_connection
from manager.reaction import generate_reactions
from manager.schemas import ManageResult
from manager.service import manage as manage_impl
from orchestrator.graph import build_graph
from orchestrator.run import run_backtest_gameweek
from shared.contracts import AgentArgument

from .agents_bridge import SpecialistBridge, disable_all_llm_calls, log_predictions, write_team_state
from .scoring import score_gameweek
from .state import SeasonState

DEFAULT_START_GW = 2  # GW1 has no prior history to matter for transfers; still solved from scratch if used
MIN_CANDIDATES = 15  # can't hand the solver a pool smaller than one full squad


@dataclass
class SeasonReport:
    season: str
    ablated_agent: str | None = None
    gameweeks: list[int] = field(default_factory=list)
    net_points_by_gw: dict[int, float] = field(default_factory=dict)
    final_state: SeasonState = field(default_factory=SeasonState)

    @property
    def total_points(self) -> float:
        return self.final_state.total_points


def _latest_prices(conn, gameweek: int, player_ids: list[int]) -> dict[int, float]:
    if not player_ids:
        return {}
    rows = conn.execute(
        """SELECT DISTINCT ON (player_id) player_id, price FROM player_gameweek_stats
           WHERE player_id = ANY(%s) AND gw <= %s AND price IS NOT NULL
           ORDER BY player_id, gw DESC""",
        (player_ids, gameweek),
    ).fetchall()
    return {r["player_id"]: float(r["price"]) for r in rows}


def actual_points(conn, gameweek: int, player_ids: list[int]) -> dict[int, float]:
    if not player_ids:
        return {}
    rows = conn.execute(
        "SELECT player_id, total_points FROM player_gameweek_stats WHERE gw = %s AND player_id = ANY(%s)",
        (gameweek, player_ids),
    ).fetchall()
    return {r["player_id"]: float(r["total_points"] or 0) for r in rows}


def build_player_pool(conn, frame: pd.DataFrame, feature_cols: list[str], gameweek: int, squad: frozenset[int]) -> list[dict]:
    gw_frame = frame[frame["GW"] == gameweek]
    player_ids = gw_frame["player_id"].astype(int).tolist()
    prices = _latest_prices(conn, gameweek, player_ids)

    fact_rows = conn.execute("SELECT player_id, team_id, position FROM players WHERE player_id = ANY(%s)", (player_ids,)).fetchall()
    facts = {r["player_id"]: r for r in fact_rows}

    pool = []
    for row in gw_frame.itertuples():
        pid = int(row.player_id)
        fact = facts.get(pid)
        price = prices.get(pid)
        if fact is None or price is None:
            continue  # no resolvable club/position/price this gameweek - not a usable candidate
        pool.append(
            {
                "player_id": pid,
                "club_id": fact["team_id"],
                "position": fact["position"],
                "price": price,
                "name": row.name,
                "features": {c: float(getattr(row, c)) for c in feature_cols},
                "in_current_squad": pid in squad,
            }
        )
    return pool


def run_season(season: str = "2025-26", start_gw: int = DEFAULT_START_GW, end_gw: int = 38, ablate_agent: str | None = None) -> SeasonReport:
    """Walks forward from ``start_gw`` to ``end_gw``. ``ablate_agent``, if
    given, removes one specialist entirely from the debate for the whole
    run (ARCHITECTURE.md 9's per-agent ablation runs) - that agent's
    AgentArgument is replaced with an empty one, so the Manager sees
    exactly what "missing opinions default to zero adjustment" already
    handles, not a special case.
    """
    disable_all_llm_calls()
    schema = seed_backtest_schema(season)
    conn = get_connection(schema=schema)
    frame, feature_cols, _ = build_feature_frame()
    stats_model = StatsModel()
    bridge = SpecialistBridge(conn, stats_model)

    def agent_caller(agent_name, gameweek, player_pool):
        if agent_name == ablate_agent:
            return AgentArgument(agent=agent_name, reasoning=f"{agent_name} ablated for this run")
        return bridge(agent_name, gameweek, player_pool)

    graph = build_graph(agent_caller=agent_caller, manage_caller=manage_impl, react_caller=generate_reactions)

    state = SeasonState()
    report = SeasonReport(season=season, ablated_agent=ablate_agent)

    for gw in range(start_gw, end_gw + 1):
        write_team_state(conn, gw, state.squad, state.bank)
        pool = build_player_pool(conn, frame, feature_cols, gw, state.squad)
        if len(pool) < MIN_CANDIDATES:
            continue  # e.g. a gameweek past the archive's coverage - skip rather than crash

        squad_value = sum(_latest_prices(conn, gw, list(state.squad)).values())
        initial_state = {
            "player_pool": pool,
            "budget": state.bank + squad_value,
            "free_transfers": state.free_transfers,
            "hit_cost": 4.0,
        }
        result_state = run_backtest_gameweek(gw, initial_state, graph=graph)

        stats_raw = result_state["first_round"].get("stats")
        if stats_raw is not None:
            log_predictions(conn, gw, AgentArgument.model_validate(stats_raw), stats_model.model_version)
        conn.commit()

        manage_result_dict = result_state["manage_result"]
        report.gameweeks.append(gw)
        if not manage_result_dict["feasible"]:
            report.net_points_by_gw[gw] = 0.0
            continue

        manage_result = ManageResult.model_validate(manage_result_dict)
        squad_ids = [s.player_id for s in manage_result.squad]
        actual = actual_points(conn, gw, squad_ids)
        # Union with the OLD squad, not just the new one - score_gameweek's
        # bank_delta needs a dropped player's sale price too, and a player
        # who left the squad this gameweek isn't in squad_ids at all.
        prices = _latest_prices(conn, gw, list(set(squad_ids) | state.squad))
        chip = result_state.get("chip")

        gw_result = score_gameweek(manage_result, actual, prices, state, chip=chip)
        state = gw_result.new_state
        report.net_points_by_gw[gw] = gw_result.net_points

    report.final_state = state
    return report


# The four adjustment agents plus News (Stats is the base currency, never
# ablated - removing it would leave the solver nothing to optimize at all,
# not a meaningful "what does this agent contribute" comparison).
ABLATABLE_AGENTS = ("fixtures", "contrarian", "template", "chips", "news")


def run_ablations(season: str = "2025-26", start_gw: int = DEFAULT_START_GW, end_gw: int = 38) -> dict[str, SeasonReport]:
    """ARCHITECTURE.md 9: "rerun the season with one specialist agent
    removed at a time, to show which agent's input actually moved the
    final score." Returns one SeasonReport per ablated agent - compare each
    against a baseline ``run_season(season, start_gw, end_gw)`` run with no
    ablation.
    """
    return {agent: run_season(season, start_gw, end_gw, ablate_agent=agent) for agent in ABLATABLE_AGENTS}
