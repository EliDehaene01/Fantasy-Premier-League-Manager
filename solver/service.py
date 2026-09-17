"""Solver - FastAPI service. Thin on purpose: the LP itself lives in
optimizer.py and is unit-tested directly without going through HTTP at all;
this file only wires it up as ``POST /solve``. No DB, no LLM - the solver
gets everything it needs (already-scored candidates) in the request body.
"""

from __future__ import annotations

from fastapi import FastAPI

from .optimizer import solve_squad
from .schemas import SolveRequest, SolveResult

app = FastAPI(title="FPL Solver", version="1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/solve", response_model=SolveResult)
def solve(request: SolveRequest) -> SolveResult:
    return solve_squad(request)
