"""Orchestrator - FastAPI wrapper exposing the live-mode graph as an HTTP
trigger. ARCHITECTURE.md 8 lists the orchestrator as a Deployment (a
persistent service), not a one-shot Job - this is what makes it one: the
ingestion Job calls ``POST /run`` once fresh data is ready for a gameweek,
and the human-approval step (Phase 7) calls ``POST /resume`` with the
reviewed decision.

Backtest mode is deliberately NOT exposed here - it's a script/library
entrypoint (``backtest/engine.py``), run standalone against a seeded
archive, never through this live service (ARCHITECTURE.md 1: backtest is
"manual/scripted", live is the CronJob-triggered path).
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass

from fastapi import FastAPI
from pydantic import BaseModel

from .graph import build_graph
from .run import postgres_checkpointer, resume_live_gameweek, run_live_gameweek

app = FastAPI(title="FPL Orchestrator", version="1.0")
_graph = None


def _get_graph():
    # Built once per process (the Postgres checkpointer connection is
    # reused across requests), not per request - matches run.py's own
    # docstring note that build_graph is a once-per-run(s) call.
    global _graph
    if _graph is None:
        _graph = build_graph(checkpointer=postgres_checkpointer())
    return _graph


def _json_safe(value):
    """LangGraph's ``__interrupt__`` entries are ``Interrupt`` dataclasses,
    not plain dicts - FastAPI's default JSON encoder doesn't know them.
    Recursively convert dataclasses/tuples so the raw graph-state dict
    returned by run.py's functions serializes without a custom response
    model duplicating GraphState's shape.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class RunRequest(BaseModel):
    gameweek: int
    initial_state: dict


class ResumeRequest(BaseModel):
    gameweek: int
    decision: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/run")
def run(request: RunRequest) -> dict:
    return _json_safe(run_live_gameweek(request.gameweek, request.initial_state, graph=_get_graph()))


@app.post("/resume")
def resume(request: ResumeRequest) -> dict:
    return _json_safe(resume_live_gameweek(request.gameweek, request.decision, graph=_get_graph()))
