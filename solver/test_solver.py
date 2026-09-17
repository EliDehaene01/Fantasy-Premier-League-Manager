"""Tests for the solver's LP (solver/optimizer.py), called directly - no
DB, no HTTP, no LLM involved, so these run fully offline and fast.

Player pools are hand-built per test rather than one shared generic pool:
each test wants total control over which constraint is actually binding, and
a shared "realistic" pool makes it hard to tell whether a given constraint
was truly exercised or just happened to be satisfied.
"""

from __future__ import annotations

import pytest

from shared.contracts import ChipType
from solver.optimizer import solve_squad
from solver.schemas import SolveRequest, SolverPlayer


def _p(pid, position, club, price, score, **kwargs):
    return SolverPlayer(player_id=pid, position=position, club_id=club, price=price, adjusted_score=score, **kwargs)


def _base_squad():
    """A legal 15-man squad (2 GK, 5 DEF, 5 MID, 3 FWD), each player at a
    distinct club so the club cap never binds unless a test deliberately
    stacks one. Prices/scores are distinct so ties never leave ordering
    ambiguous.
    """
    players = []
    pid = 1
    club = 1
    for position, count, price, score in [("GK", 2, 4.5, 4.0), ("DEF", 5, 4.5, 4.0), ("MID", 5, 5.0, 5.0), ("FWD", 3, 5.5, 6.0)]:
        for i in range(count):
            players.append(_p(pid, position, club, price + i * 0.1, score + i, in_current_squad=True))
            pid += 1
            club += 1
    return players


def test_builds_valid_squad_from_scratch():
    players = [SolverPlayer(**{**p.model_dump(), "in_current_squad": False}) for p in _base_squad()]
    result = solve_squad(SolveRequest(gameweek=1, players=players))

    assert result.feasible
    assert len(result.squad) == 15
    starters = [s for s in result.squad if s.starting]
    assert len(starters) == 11
    assert result.transfers_made == 0
    assert result.hits_taken == 0


def test_hard_excludes_vetoed_out_players():
    players = [SolverPlayer(**{**p.model_dump(), "in_current_squad": False}) for p in _base_squad()]
    # A 4th, unambiguously-best FWD candidate that is vetoed OUT - must never
    # appear in the result even though its score dwarfs everything else.
    players.append(_p(999, "FWD", 900, price=5.0, score=1000.0, vetoed_out=True))

    result = solve_squad(SolveRequest(gameweek=1, players=players))

    assert result.feasible
    squad_ids = {s.player_id for s in result.squad}
    assert 999 not in squad_ids


def test_club_cap_enforced():
    players = [SolverPlayer(**{**p.model_dump(), "in_current_squad": False}) for p in _base_squad()]
    # Stack 4 excellent MIDs on one club (id 500) - only 3 may ever be
    # squadded. Add two extra ordinary-club MIDs so a legal 5-MID squad
    # still exists once the cap bites (3 from club 500 + 2 fillers = 5).
    players = [p for p in players if p.position != "MID"]
    for i in range(4):
        players.append(_p(600 + i, "MID", 500, price=5.0, score=100.0))
    players.append(_p(610, "MID", 501, price=5.0, score=1.0))
    players.append(_p(611, "MID", 502, price=5.0, score=1.0))

    result = solve_squad(SolveRequest(gameweek=1, players=players))

    assert result.feasible
    squad_ids = {s.player_id for s in result.squad}
    from_stacked_club = squad_ids & {600, 601, 602, 603}
    assert len(from_stacked_club) <= 3
    # both fillers are needed to reach 5 MIDs once the cap caps club 500 at 3
    assert {610, 611} <= squad_ids


def test_transfer_within_free_allowance_costs_nothing():
    """A dominant replacement gets bought in for free. Note: which specific
    original MID ends up dropped vs merely benched is not asserted - the
    objective only scores the starting XI, so the solver is legitimately
    indifferent between two current squad members once one of them is
    benched either way (see solver/optimizer.py's module docstring for the
    general shape of this trade-off). What must hold is that the swap
    happened, cost nothing, and the new player actually starts. Scoped to
    the non-Bench-Boost case specifically - this tie only exists because a
    bench player's score doesn't reach the objective; under Bench Boost it
    would (see test_bench_boost_counts_the_full_squad_in_the_objective),
    and this indifference wouldn't hold.
    """
    players = _base_squad()
    weakest_mid = min((p for p in players if p.position == "MID"), key=lambda p: p.adjusted_score)
    replacement = _p(9001, "MID", 9001, price=weakest_mid.price, score=weakest_mid.adjusted_score + 50.0)
    players.append(replacement)

    result = solve_squad(SolveRequest(gameweek=1, players=players, free_transfers=1))

    assert result.feasible
    assert result.transfers_made == 1
    assert result.hits_taken == 0
    assert result.hit_points_cost == 0.0
    starters = {s.player_id for s in result.squad if s.starting}
    assert replacement.player_id in starters


def test_veto_forces_a_paid_hit_when_no_free_transfer_remains():
    players = _base_squad()
    vetoed_fwd = next(p for p in players if p.position == "FWD")
    players = [SolverPlayer(**{**p.model_dump(), "vetoed_out": True}) if p.player_id == vetoed_fwd.player_id else p for p in players]
    replacement = _p(9002, "FWD", 9002, price=5.5, score=1.0)
    players.append(replacement)

    result = solve_squad(SolveRequest(gameweek=1, players=players, free_transfers=0, hit_cost=4.0))

    assert result.feasible
    assert result.transfers_made == 1
    assert result.hits_taken == 1
    assert result.hit_points_cost == pytest.approx(4.0)
    squad_ids = {s.player_id for s in result.squad}
    assert vetoed_fwd.player_id not in squad_ids
    assert replacement.player_id in squad_ids


def test_takes_a_beneficial_hit_even_when_the_free_only_squad_is_feasible():
    """The core fix: a plan using only the free transfer is already
    feasible (just keep the squad, or make the one free swap) - proving the
    solver takes a SECOND, paid transfer anyway requires that its point
    gain clear the -4 hit cost, not that it was forced by infeasibility.
    """
    players = _base_squad()
    mids = sorted((p for p in players if p.position == "MID"), key=lambda p: p.adjusted_score)
    weakest, second_weakest = mids[0], mids[1]
    # Both replacements dwarf what they'd replace by far more than hit_cost
    # (4.0) - taking both nets a large gain even after paying for the 2nd.
    replacement_a = _p(9101, "MID", 9101, price=weakest.price, score=weakest.adjusted_score + 100.0)
    replacement_b = _p(9102, "MID", 9102, price=second_weakest.price, score=second_weakest.adjusted_score + 90.0)
    players += [replacement_a, replacement_b]

    result = solve_squad(SolveRequest(gameweek=1, players=players, free_transfers=1, hit_cost=4.0))

    assert result.feasible
    assert result.transfers_made == 2
    assert result.hits_taken == 1
    assert result.hit_points_cost == pytest.approx(4.0)
    starters = {s.player_id for s in result.squad if s.starting}
    assert {replacement_a.player_id, replacement_b.player_id} <= starters


def test_wildcard_allows_unlimited_free_transfers():
    """The new squad's scores dominate the old one's, so every STARTING
    slot should be filled from the new pool - not necessarily every bench
    slot too, since (as in the transfer test above) an unused bench slot
    contributes nothing to the objective either way and the solver may
    leave a cheap old player parked there. What proves unlimited transfers
    actually worked is that the whole starting XI turned over, well beyond
    the nominal 1 free transfer. Scoped to the non-Bench-Boost case, same
    caveat as the transfer test above - chip is WILDCARD here, not
    BENCH_BOOST, so bench-slot identity is genuinely a don't-care.
    """
    old_squad = _base_squad()
    new_squad = []
    pid = 2000
    club = 2000
    for position, count, price, score in [("GK", 2, 4.5, 40.0), ("DEF", 5, 4.5, 40.0), ("MID", 5, 5.0, 50.0), ("FWD", 3, 5.5, 60.0)]:
        for i in range(count):
            new_squad.append(_p(pid, position, club, price, score))
            pid += 1
            club += 1

    result = solve_squad(SolveRequest(gameweek=1, players=old_squad + new_squad, free_transfers=1, chip=ChipType.WILDCARD))

    assert result.feasible
    assert result.hits_taken == 0
    assert result.hit_points_cost == 0.0
    starters = {s.player_id for s in result.squad if s.starting}
    new_ids = {p.player_id for p in new_squad}
    assert starters <= new_ids
    assert len(starters) == 11
    assert result.transfers_made > 1  # well beyond the 1 free transfer available


def test_bench_boost_counts_the_full_squad_in_the_objective():
    players = _base_squad()
    total_score = sum(p.adjusted_score for p in players)

    normal = solve_squad(SolveRequest(gameweek=1, players=players))
    boosted = solve_squad(SolveRequest(gameweek=1, players=players, chip=ChipType.BENCH_BOOST))

    assert normal.feasible and boosted.feasible
    assert boosted.objective_value == pytest.approx(total_score)
    assert normal.objective_value < total_score


def test_infeasible_budget_returns_no_valid_plan_message():
    players = [SolverPlayer(**{**p.model_dump(), "in_current_squad": False}) for p in _base_squad()]

    result = solve_squad(SolveRequest(gameweek=1, players=players, budget=1.0))

    assert result.feasible is False
    assert result.squad == []
    assert result.message == "no valid plan found"
