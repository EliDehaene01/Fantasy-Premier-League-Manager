"""Request/response models for the Manager service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shared.contracts import AgentArgument, ChipType
from solver.schemas import SquadSelection

Position = Literal["GK", "DEF", "MID", "FWD"]


class ManagerPlayerFact(BaseModel):
    """Static facts the agent contract doesn't carry (price/position/club).
    Normally read from silver's `players` (position/team_id) and
    `player_gameweek_stats` (price) tables by whoever assembles this
    request - see manager/__init__.py's module docstring for why that isn't
    this service's own job.
    """

    model_config = ConfigDict(extra="ignore")

    player_id: int
    club_id: int
    position: Position
    price: float = Field(ge=0)
    in_current_squad: bool = False


class ManageRequest(BaseModel):
    gameweek: int = Field(ge=1, le=38)
    budget: float = Field(default=100.0, gt=0)
    free_transfers: int = Field(default=1, ge=0)
    hit_cost: float = Field(default=4.0, ge=0)
    chip: ChipType | None = None
    player_pool: list[ManagerPlayerFact] = Field(default_factory=list)
    stats: AgentArgument
    # Keyed by "fixtures"/"contrarian"/"template"/"news" - any subset; a
    # missing agent contributes zero adjustment (ARCHITECTURE.md 6b).
    adjustments: dict[str, AgentArgument] = Field(default_factory=dict)


class ManageResult(BaseModel):
    feasible: bool
    squad: list[SquadSelection] = Field(default_factory=list)
    captain_id: int | None = None
    vice_captain_id: int | None = None
    transfers_made: int = 0
    hits_taken: int = 0
    hit_points_cost: float = 0.0
    narration: str = ""
    message: str | None = None


class ReactionRequest(BaseModel):
    gameweek: int = Field(ge=1, le=38)
    first_round: dict[str, AgentArgument] = Field(default_factory=dict)
