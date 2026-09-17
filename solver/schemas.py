"""Request/response models for the solver service. Separate from
shared/contracts.py deliberately - that file is the specialist-agent debate
contract (``ArgueRequest``/``AgentArgument``); the solver isn't a specialist
and speaks a different shape (a scored candidate pool in, a squad out). Only
``ChipType`` is reused from there, since "which chip is active" is the same
concept in both places.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.contracts import ChipType

Position = Literal["GK", "DEF", "MID", "FWD"]


class SolverPlayer(BaseModel):
    """One candidate the solver may pick. ``adjusted_score`` is the
    Manager's already-computed number (ARCHITECTURE.md 6b) - the solver
    does no scoring of its own, only selection.
    """

    model_config = ConfigDict(extra="ignore")

    player_id: int
    club_id: int
    position: Position
    price: float = Field(ge=0)
    adjusted_score: float
    # True for a player already in the squad being transferred FROM. Ignored
    # entirely when building a squad from scratch (no player has this set).
    in_current_squad: bool = False
    # The News agent's hard OUT veto (CLAUDE.md's hard constraints). The
    # Manager is expected to pass this through unfiltered - the solver does
    # the actual exclusion, so the constraint lives in one tested place
    # rather than being re-implemented by every caller.
    vetoed_out: bool = False


class SolveRequest(BaseModel):
    gameweek: int = Field(ge=1, le=38)
    players: list[SolverPlayer] = Field(default_factory=list)
    budget: float = Field(default=100.0, gt=0)
    free_transfers: int = Field(default=1, ge=0)
    hit_cost: float = Field(default=4.0, ge=0)
    # None = ordinary gameweek. WILDCARD/FREE_HIT lift the transfer cap
    # entirely; BENCH_BOOST changes the objective to count all 15 squad
    # members, not just the starting 11 (see optimizer.py).
    chip: ChipType | None = None


class SquadSelection(BaseModel):
    player_id: int
    starting: bool


class SolveResult(BaseModel):
    feasible: bool
    squad: list[SquadSelection] = Field(default_factory=list)
    transfers_made: int = 0
    hits_taken: int = 0
    hit_points_cost: float = 0.0
    objective_value: float | None = None
    # Set only when feasible is False - "no valid plan found", never a
    # partial/degenerate squad (ARCHITECTURE.md 6a).
    message: str | None = None
