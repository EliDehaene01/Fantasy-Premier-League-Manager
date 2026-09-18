"""Tests for backtest/scoring.py - pure functions, no DB, fast. Covers the
captain doubling/tripling, chip effects (Bench Boost counting the bench,
Free Hit reverting), hit-cost deduction, and free-transfer rollover math
ARCHITECTURE.md 9 calls for.
"""

from __future__ import annotations

import pytest

from backtest.scoring import score_gameweek
from backtest.state import MAX_FREE_TRANSFERS, SeasonState
from manager.schemas import ManageResult
from solver.schemas import SquadSelection


def _squad(starting_ids, bench_ids):
    return [SquadSelection(player_id=pid, starting=True) for pid in starting_ids] + [
        SquadSelection(player_id=pid, starting=False) for pid in bench_ids
    ]


def test_normal_gameweek_sums_starting_xi_with_captain_doubled():
    squad = _squad(starting_ids=[1, 2, 3], bench_ids=[4])
    result = ManageResult(feasible=True, squad=squad, captain_id=1, vice_captain_id=2, transfers_made=0, hits_taken=0, hit_points_cost=0.0)
    actual = {1: 10.0, 2: 5.0, 3: 2.0, 4: 100.0}  # bench player's huge score must NOT count

    r = score_gameweek(result, actual, player_prices={}, state=SeasonState(squad=frozenset({1, 2, 3, 4})), chip=None)

    assert r.gross_points == pytest.approx(10.0 + 5.0 + 2.0 + 10.0)  # captain doubled: +10 extra
    assert r.net_points == pytest.approx(r.gross_points)


def test_triple_captain_triples_instead_of_doubles():
    squad = _squad(starting_ids=[1], bench_ids=[])
    result = ManageResult(feasible=True, squad=squad, captain_id=1, vice_captain_id=None, transfers_made=0, hits_taken=0, hit_points_cost=0.0)

    r = score_gameweek(result, {1: 10.0}, {}, SeasonState(squad=frozenset({1})), chip="triple_captain")

    assert r.gross_points == pytest.approx(30.0)


def test_bench_boost_counts_the_bench():
    squad = _squad(starting_ids=[1], bench_ids=[2])
    result = ManageResult(feasible=True, squad=squad, captain_id=1, vice_captain_id=None, transfers_made=0, hits_taken=0, hit_points_cost=0.0)

    normal = score_gameweek(result, {1: 10.0, 2: 6.0}, {}, SeasonState(squad=frozenset({1, 2})), chip=None)
    boosted = score_gameweek(result, {1: 10.0, 2: 6.0}, {}, SeasonState(squad=frozenset({1, 2})), chip="bench_boost")

    assert normal.gross_points == pytest.approx(20.0)  # captain doubled, bench (6) excluded
    assert boosted.gross_points == pytest.approx(26.0)  # bench's 6 now included


def test_hit_cost_is_deducted_from_net_but_not_gross():
    squad = _squad(starting_ids=[1], bench_ids=[])
    result = ManageResult(feasible=True, squad=squad, captain_id=1, vice_captain_id=None, transfers_made=2, hits_taken=1, hit_points_cost=4.0)

    r = score_gameweek(result, {1: 10.0}, {}, SeasonState(squad=frozenset({1})), chip=None)

    assert r.gross_points == pytest.approx(20.0)
    assert r.net_points == pytest.approx(16.0)


def test_free_hit_squad_reverts_next_gameweek():
    old_squad = frozenset({1, 2, 3})
    free_hit_squad = _squad(starting_ids=[4, 5, 6], bench_ids=[])
    result = ManageResult(feasible=True, squad=free_hit_squad, captain_id=4, vice_captain_id=None, transfers_made=3, hits_taken=0, hit_points_cost=0.0)

    r = score_gameweek(result, {4: 8.0}, {}, SeasonState(squad=old_squad, bank=50.0), chip="free_hit")

    assert r.new_state.squad == old_squad  # reverted, not the free-hit XI
    assert r.new_state.bank == pytest.approx(50.0)  # nothing was actually spent


def test_wildcard_squad_is_permanent_and_updates_bank():
    old_squad = frozenset({1, 2})
    new_squad = _squad(starting_ids=[3, 4], bench_ids=[])
    result = ManageResult(feasible=True, squad=new_squad, captain_id=3, vice_captain_id=None, transfers_made=2, hits_taken=0, hit_points_cost=0.0)
    prices = {1: 5.0, 2: 5.0, 3: 6.0, 4: 7.0}

    r = score_gameweek(result, {3: 8.0}, prices, SeasonState(squad=old_squad, bank=10.0), chip="wildcard")

    assert r.new_state.squad == frozenset({3, 4})
    assert r.new_state.bank == pytest.approx(10.0 + (5.0 + 5.0) - (6.0 + 7.0))


def test_free_transfers_roll_over_and_cap():
    squad = _squad(starting_ids=[1], bench_ids=[])
    result = ManageResult(feasible=True, squad=squad, captain_id=1, vice_captain_id=None, transfers_made=0, hits_taken=0, hit_points_cost=0.0)

    r = score_gameweek(result, {1: 1.0}, {}, SeasonState(squad=frozenset({1}), free_transfers=MAX_FREE_TRANSFERS), chip=None)

    assert r.new_state.free_transfers == MAX_FREE_TRANSFERS  # already at cap, stays capped


def test_using_a_free_transfer_reduces_next_gameweeks_allowance_before_the_weekly_increment():
    squad = _squad(starting_ids=[1], bench_ids=[])
    result = ManageResult(feasible=True, squad=squad, captain_id=1, vice_captain_id=None, transfers_made=1, hits_taken=0, hit_points_cost=0.0)

    r = score_gameweek(result, {1: 1.0}, {}, SeasonState(squad=frozenset({1}), free_transfers=1), chip=None)

    assert r.new_state.free_transfers == 1  # used the 1 available, then +1 for next week


def test_chips_used_is_recorded():
    squad = _squad(starting_ids=[1], bench_ids=[])
    result = ManageResult(feasible=True, squad=squad, captain_id=1, vice_captain_id=None, transfers_made=0, hits_taken=0, hit_points_cost=0.0)

    r = score_gameweek(result, {1: 1.0}, {}, SeasonState(squad=frozenset({1})), chip="wildcard")

    assert "wildcard" in r.new_state.chips_used
