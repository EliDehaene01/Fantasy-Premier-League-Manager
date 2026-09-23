"""Manager - FastAPI service (ARCHITECTURE.md 6b). Aggregates the four
adjustment agents' conviction onto Stats' predicted_points, applies News's
vetoes, calls the solver, picks captain/vice-captain deterministically, and
narrates the result. The LLM touches only the reaction round and the final
narration - every number that decides WHICH players are picked is fixed
math, computed before any LLM call happens.

Calls ``solver.optimizer.solve_squad`` as a direct Python import rather than
over HTTP - both are pure, DB-free functions within the same monorepo, and
the "separate containerized service" boundary (CLAUDE.md's repo layout) is
a Phase 4 deployment concern, not a reason to add network flakiness to this
component's own tests. Swapping this for an HTTP call to the solver's
deployed service is a small, isolated change if that boundary needs to be
real before containerization.
"""

from __future__ import annotations

from fastapi import FastAPI

from shared.contracts import VetoStatus
from solver.optimizer import solve_squad
from solver.schemas import SolveRequest, SolverPlayer

from .aggregation import PlayerScore, compute_adjusted_scores, select_captain
from .narration import narrate
from .reaction import generate_reactions
from .schemas import ManageRequest, ManageResult, ManagerPlayerFact, ReactionRequest

app = FastAPI(title="FPL Manager", version="1.0")


def _build_candidates(scores: dict[int, PlayerScore], player_pool: list[ManagerPlayerFact]) -> list[SolverPlayer]:
    facts = {f.player_id: f for f in player_pool}
    candidates = []
    for player_id, score in scores.items():
        fact = facts.get(player_id)
        if fact is None:
            continue  # no price/position/club known - can't hand to the solver
        candidates.append(
            SolverPlayer(
                player_id=player_id,
                club_id=fact.club_id,
                position=fact.position,
                price=fact.price,
                adjusted_score=score.adjusted_score,
                in_current_squad=fact.in_current_squad,
                vetoed_out=(score.veto == VetoStatus.OUT),
            )
        )
    return candidates


def _summarize(gameweek: int, transfers_made: int, hits_taken: int, hit_points_cost: float, captain_id: int | None, vice_id: int | None) -> str:
    return (
        f"Gameweek {gameweek}: squad finalized with {transfers_made} transfer(s) and "
        f"{hits_taken} hit(s) ({hit_points_cost:.0f}pt cost). Captain: player #{captain_id}. "
        f"Vice-captain: player #{vice_id}."
    )


def manage(request: ManageRequest) -> ManageResult:
    scores = compute_adjusted_scores(request.stats, request.adjustments, request.player_pool)
    candidates = _build_candidates(scores, request.player_pool)

    solve_result = solve_squad(
        SolveRequest(
            gameweek=request.gameweek,
            players=candidates,
            budget=request.budget,
            free_transfers=request.free_transfers,
            hit_cost=request.hit_cost,
            chip=request.chip,
        )
    )
    if not solve_result.feasible:
        return ManageResult(feasible=False, message=solve_result.message, narration="No valid squad could be found this gameweek.")

    starting_ids = {s.player_id for s in solve_result.squad if s.starting}
    captain_id, vice_id = select_captain(scores, starting_ids)

    summary = _summarize(request.gameweek, solve_result.transfers_made, solve_result.hits_taken, solve_result.hit_points_cost, captain_id, vice_id)
    narration = narrate(request.gameweek, summary)

    return ManageResult(
        feasible=True,
        squad=solve_result.squad,
        captain_id=captain_id,
        vice_captain_id=vice_id,
        transfers_made=solve_result.transfers_made,
        hits_taken=solve_result.hits_taken,
        hit_points_cost=solve_result.hit_points_cost,
        narration=narration,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/manage", response_model=ManageResult)
def manage_route(request: ManageRequest) -> ManageResult:
    return manage(request)


@app.post("/react")
def react_route(request: ReactionRequest) -> dict[str, str]:
    return generate_reactions(request.gameweek, request.first_round)
