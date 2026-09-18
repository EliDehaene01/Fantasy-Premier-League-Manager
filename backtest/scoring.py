"""ARCHITECTURE.md 9 step 2's scoring: "Score the resulting squad against
the actual results for that gameweek; apply transfer-hit penalties and chip
effects." Pure functions - given actual points per player and a
ManageResult, compute what was scored and the next SeasonState. No DB
access here; ``engine.py`` fetches the actual-points dict separately so
this module stays trivially testable.

**Known simplification, documented rather than silently assumed**: FPL's
real auto-substitution (a non-playing captain's armband passing to the
vice-captain, a 0-minute starter being swapped for a bench player in a
valid formation) is NOT modelled. The Manager's captain/vice are always
drawn from the finalized starting 11 by construction
(manager/aggregation.py::select_captain), so this only under-counts points
in the specific case where a starting XI pick ends up unavailable/subbed
off for 0 minutes after the deadline - a real but second-order effect, left
as a documented gap rather than reimplementing FPL's full auto-sub rules.
"""

from __future__ import annotations

from dataclasses import dataclass

from manager.schemas import ManageResult
from shared.contracts import ChipType

from .state import MAX_FREE_TRANSFERS, SeasonState


@dataclass
class GameweekResult:
    gross_points: float
    hit_points_cost: float
    net_points: float
    new_state: SeasonState


def score_gameweek(
    manage_result: ManageResult,
    actual_points: dict[int, float],
    player_prices: dict[int, float],
    state: SeasonState,
    chip: str | None,
) -> GameweekResult:
    squad_ids = {s.player_id for s in manage_result.squad}
    starting_ids = {s.player_id for s in manage_result.squad if s.starting}

    # Bench Boost counts the full 15; otherwise only the starting XI scores
    # (mirrors the solver's own bench_boost handling of the objective, now
    # applied to the ACTUAL result).
    scoring_ids = squad_ids if chip == ChipType.BENCH_BOOST.value else starting_ids
    base_points = sum(actual_points.get(pid, 0.0) for pid in scoring_ids)

    captain_bonus = 0.0
    if manage_result.captain_id is not None:
        # Captain is always drawn from the starting XI (see module
        # docstring), so their points are already counted once above -
        # the bonus is the EXTRA multiplier-minus-one share.
        cap_actual = actual_points.get(manage_result.captain_id, 0.0)
        multiplier = 3 if chip == ChipType.TRIPLE_CAPTAIN.value else 2
        captain_bonus = cap_actual * (multiplier - 1)

    gross_points = base_points + captain_bonus
    net_points = gross_points - manage_result.hit_points_cost

    if chip == ChipType.FREE_HIT.value:
        # Reverts after one gameweek (ARCHITECTURE.md 6a): the Free Hit
        # squad this gameweek was never actually "bought" - next gameweek's
        # squad and bank are exactly what they were before it.
        new_squad = state.squad
        new_bank = state.bank
    else:
        dropped = state.squad - squad_ids
        bought = squad_ids - state.squad
        bank_delta = sum(player_prices.get(pid, 0.0) for pid in dropped) - sum(player_prices.get(pid, 0.0) for pid in bought)
        new_squad = frozenset(squad_ids)
        new_bank = state.bank + bank_delta

    unlimited_transfers = chip in (ChipType.WILDCARD.value, ChipType.FREE_HIT.value)
    if unlimited_transfers:
        new_free_transfers = min(MAX_FREE_TRANSFERS, state.free_transfers + 1)
    else:
        used_free = max(0, manage_result.transfers_made - manage_result.hits_taken)
        new_free_transfers = min(MAX_FREE_TRANSFERS, max(0, state.free_transfers - used_free) + 1)

    new_chips_used = state.chips_used | ({chip} if chip else set())

    new_state = SeasonState(
        squad=new_squad,
        bank=new_bank,
        free_transfers=new_free_transfers,
        chips_used=new_chips_used,
        total_points=state.total_points + net_points,
    )
    return GameweekResult(gross_points=gross_points, hit_points_cost=manage_result.hit_points_cost, net_points=net_points, new_state=new_state)
