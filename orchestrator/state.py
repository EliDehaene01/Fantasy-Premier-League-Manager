"""ARCHITECTURE.md 6c's typed state: squad, bank, free transfers, chips
used, current gameweek, each agent's raw output, the reaction-round text,
solver result, approval status.

Every agent-contract field is stored as a plain dict (``.model_dump(mode="json")``),
not a raw Pydantic object, so the whole state stays JSON-serializable for
the Postgres checkpointer without a custom serializer - graph.py converts
to/from the real Pydantic models (AgentArgument, ManageResult) right at the
node boundaries where they're needed.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict


def merge_dicts(left: dict | None, right: dict | None) -> dict:
    """Reducer for state fields that six parallel specialist nodes each
    contribute one key to. LangGraph's default channel update is overwrite,
    not merge - without this, whichever specialist's write lands last in a
    given superstep would clobber the other five instead of the state
    accumulating all six (verified empirically before writing this module -
    see the session's LangGraph fan-out/fan-in check).
    """
    return {**(left or {}), **(right or {})}


class GraphState(TypedDict, total=False):
    mode: Literal["backtest", "live"]
    gameweek: int
    budget: float
    free_transfers: int
    hit_cost: float
    chip: str | None

    # Squad/bank/chips-used tracking across gameweeks (Phase 2's backtest
    # state loop reads/writes these; this graph itself only reads them to
    # build ManageRequest.player_pool's `in_current_squad` flags and to
    # pass free_transfers/chip through - it doesn't mutate season state).
    squad: list[int]
    bank: float
    chips_used: list[str]

    # Superset player facts (player_id, club_id, position, price, features,
    # chance_of_playing_this_round, news, name, in_current_squad) - each
    # downstream call (specialist/Manager) projects out what it needs via
    # Pydantic's `extra="ignore"` on PlayerEntry/ManagerPlayerFact.
    player_pool: list[dict]

    first_round: Annotated[dict[str, dict], merge_dicts]  # agent_name -> AgentArgument dict
    reactions: dict[str, str]
    manage_result: dict | None
    approval_status: Literal["auto_accepted", "approved", "rejected", "declined"] | None
