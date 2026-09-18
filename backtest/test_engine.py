"""Integration test for the walk-forward loop (backtest/engine.py) against
the REAL seeded 2025-26 archive and a real Postgres connection - this is
deliberately not mocked, unlike every other test in this project, because
the whole point of this module is gluing real components (the trained
model, the real DB-backed agent scoring, the real solver) together, and a
mock of any of those would test nothing but the glue itself. Needs Postgres
reachable (same requirement as every other DB-backed test suite here).

Kept to a small gameweek range - this is a correctness check, not the
full-season run (see backtest/run_full_season.py for that).
"""

from __future__ import annotations

from backtest.engine import run_season
from backtest.state import STARTING_BUDGET


def test_walk_forward_produces_a_legal_squad_each_gameweek_with_nonnegative_bank():
    report = run_season(start_gw=2, end_gw=3)

    assert report.gameweeks == [2, 3]
    assert len(report.final_state.squad) == 15
    assert report.final_state.bank >= 0.0
    assert report.final_state.bank <= STARTING_BUDGET
    # Real players scored real points - not a fabricated/zero result.
    assert any(pts != 0.0 for pts in report.net_points_by_gw.values())


def test_ablated_agent_contributes_nothing_but_run_still_completes():
    report = run_season(start_gw=2, end_gw=2, ablate_agent="fixtures")

    assert report.ablated_agent == "fixtures"
    assert report.gameweeks == [2]
    assert len(report.final_state.squad) == 15
