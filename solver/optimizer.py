"""The actual PuLP linear program (ARCHITECTURE.md 6a). Pure function, no
FastAPI/DB dependency - the Manager (or a test) hands it a fully-scored
candidate pool and gets a squad back.

**Transfer-hit design**: a single solve prices every transfer beyond the
free allowance at ``-hit_cost`` directly in the objective, via an unbounded
integer ``hits`` variable (``hits >= transfers_out - free_transfers``,
``hits >= 0``). This is a real decision variable, not a hard cap - the
solver is free to spend a hit whenever the point gain from a swap outweighs
its cost, exactly as ARCHITECTURE.md 6a describes ("price transfer-hit cost
... directly into the objective"). An earlier version hard-capped transfers
at ``free_transfers`` in the primary attempt and only ever paid for a hit as
an infeasibility fallback, which meant it could never proactively recommend
a beneficial hit - fixed here.

The infeasibility retry (ARCHITECTURE.md 6a's "retry once ... then surface
no valid plan found") is kept as a genuinely separate second attempt, run
only if the first is infeasible. Worth being honest about its scope now:
since ``hits`` has no upper bound, the hits/transfers bookkeeping can never
itself be the reason a solve is infeasible (any transfer count is payable,
in principle, for the right price) - real infeasibility only comes from
budget, formation, or the club cap, which the retry doesn't relax. It's kept
here as defensive depth (protects against a solver-level hiccup on the first
attempt) rather than as a mechanism expected to change the outcome; if that
turns out to never matter in practice, simplifying back to one attempt is a
reasonable follow-up once there's backtest evidence either way.
"""

from __future__ import annotations

import pulp

from shared.contracts import ChipType

from .schemas import SolveRequest, SolveResult, SolverPlayer, SquadSelection

SQUAD_SHAPE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
SQUAD_SIZE = sum(SQUAD_SHAPE.values())  # 15
STARTING_SIZE = 11
STARTING_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
MAX_PER_CLUB = 3


def solve_squad(request: SolveRequest) -> SolveResult:
    candidates = [p for p in request.players if not p.vetoed_out]
    current_squad_ids = {p.player_id for p in request.players if p.in_current_squad}
    unlimited = request.chip in (ChipType.WILDCARD, ChipType.FREE_HIT)
    bench_boost = request.chip == ChipType.BENCH_BOOST
    price_hits = bool(current_squad_ids) and not unlimited

    result = _solve(candidates, current_squad_ids, request.budget, request.free_transfers, price_hits, request.hit_cost, bench_boost)
    if not result.feasible:
        # Defensive retry only - see module docstring for why this is no
        # longer expected to change the outcome now that hits are unbounded.
        result = _solve(candidates, current_squad_ids, request.budget, request.free_transfers, price_hits, request.hit_cost, bench_boost)

    if not result.feasible:
        return SolveResult(feasible=False, message="no valid plan found")
    return result


def _solve(
    candidates: list[SolverPlayer],
    current_squad_ids: set[int],
    budget: float,
    free_transfers: int,
    price_hits: bool,
    hit_cost: float,
    bench_boost: bool,
) -> SolveResult:
    if len(candidates) < SQUAD_SIZE:
        return SolveResult(feasible=False)

    by_id = {p.player_id: p for p in candidates}
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    x = {pid: pulp.LpVariable(f"x_{pid}", cat="Binary") for pid in by_id}
    s = {pid: pulp.LpVariable(f"s_{pid}", cat="Binary") for pid in by_id}

    for pid in by_id:
        prob += s[pid] <= x[pid]

    prob += pulp.lpSum(x.values()) == SQUAD_SIZE
    prob += pulp.lpSum(s.values()) == STARTING_SIZE
    prob += pulp.lpSum(by_id[pid].price * x[pid] for pid in by_id) <= budget

    for pos, count in SQUAD_SHAPE.items():
        prob += pulp.lpSum(x[pid] for pid in by_id if by_id[pid].position == pos) == count
    for pos, (lo, hi) in STARTING_BOUNDS.items():
        pos_sum = pulp.lpSum(s[pid] for pid in by_id if by_id[pid].position == pos)
        prob += pos_sum >= lo
        prob += pos_sum <= hi

    for club in {p.club_id for p in candidates}:
        prob += pulp.lpSum(x[pid] for pid in by_id if by_id[pid].club_id == club) <= MAX_PER_CLUB

    hits_var = None
    if price_hits:
        dropped = [1 - x[pid] for pid in current_squad_ids if pid in x]
        forced_out = sum(1 for pid in current_squad_ids if pid not in x)
        transfers_out_expr = pulp.lpSum(dropped) + forced_out
        # No upper bound on hits - the objective term below is what makes an
        # unjustified hit unattractive, not a hard cap on how many are legal.
        hits_var = pulp.LpVariable("hits", lowBound=0, cat="Integer")
        prob += hits_var >= transfers_out_expr - free_transfers

    objective = pulp.lpSum(
        (x[pid] if bench_boost else s[pid]) * by_id[pid].adjusted_score for pid in by_id
    )
    if hits_var is not None:
        objective = objective - hit_cost * hits_var
    prob += objective

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return SolveResult(feasible=False)

    final_ids = {pid for pid in by_id if round(x[pid].value()) == 1}
    transfers_made = len(current_squad_ids - final_ids) if current_squad_ids else 0
    hits_taken = int(round(hits_var.value())) if hits_var is not None else 0

    squad = [SquadSelection(player_id=pid, starting=round(s[pid].value()) == 1) for pid in final_ids]
    return SolveResult(
        feasible=True,
        squad=squad,
        transfers_made=transfers_made,
        hits_taken=hits_taken,
        hit_points_cost=hits_taken * hit_cost,
        objective_value=pulp.value(prob.objective),
    )
