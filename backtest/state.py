"""ARCHITECTURE.md 9 step 1: "Initialize state: starting squad, £100m bank,
1 free transfer, all chips available." Carried forward across gameweeks by
``engine.py``'s walk-forward loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

STARTING_BUDGET = 100.0
STARTING_FREE_TRANSFERS = 1
# FPL's real rollover cap - unused free transfers accumulate but never
# exceed this, so a long run of "do nothing" gameweeks doesn't let transfers
# pile up without bound.
MAX_FREE_TRANSFERS = 5
ALL_CHIPS = frozenset({"wildcard", "bench_boost", "triple_captain", "free_hit"})


@dataclass
class SeasonState:
    squad: frozenset[int] = field(default_factory=frozenset)
    bank: float = STARTING_BUDGET
    free_transfers: int = STARTING_FREE_TRANSFERS
    chips_used: frozenset[str] = field(default_factory=frozenset)
    total_points: float = 0.0
