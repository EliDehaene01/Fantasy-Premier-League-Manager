"""Contrarian agent - FastAPI service.

Same two-step shape as Stats/Fixtures: scoring.py (deterministic
predicted-quality-vs-ownership rule, reading the Stats agent's predictions
log rather than calling it live - see scoring.py's module docstring) then
reasoning.py (Foundry prose, best-effort, never load-bearing).

Never populates `vetoes` - that field belongs to the News agent alone (see
CLAUDE.md's hard constraints). DOES set `predicted_points` on its
recommendations, unlike Fixtures/Template - it has a real number, read
straight from the predictions log (see scoring.py).
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.agent_service import create_service
from shared.contracts import AgentArgument, ArgueRequest, Recommendation

from . import scoring
from .reasoning import generate_reasoning

log = logging.getLogger("contrarian_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ingestion.db import get_connection

    app.state.db_conn = None
    try:
        # CONTRARIAN_AGENT_TEST_SCHEMA lets tests point this at an isolated,
        # seeded schema instead of the real public schema - same mechanism
        # ingestion's own test suite and the Fixtures agent's tests use.
        schema = os.environ.get("CONTRARIAN_AGENT_TEST_SCHEMA")
        app.state.db_conn = get_connection(schema=schema) if schema else get_connection()
    except Exception:  # pragma: no cover - no DB configured/reachable
        log.warning(
            "Contrarian agent: Postgres unavailable at startup - /argue will return no picks", exc_info=True
        )
    yield
    if app.state.db_conn is not None:
        app.state.db_conn.close()


def _health(app: FastAPI) -> dict:
    return {"status": "ok", "db_connected": app.state.db_conn is not None}


def _argue(app: FastAPI, request: ArgueRequest) -> AgentArgument:
    conn = app.state.db_conn
    picks = scoring.rank_players(conn, request.gameweek, request.players, request.top_k) if conn is not None else []
    reasoning = generate_reasoning(request.gameweek, picks)
    return AgentArgument(
        agent="contrarian",
        recommendations=[
            Recommendation(player_id=p.player_id, conviction=p.conviction, predicted_points=p.predicted_points)
            for p in picks
        ],
        reasoning=reasoning,
    )


app = create_service(
    title="FPL Contrarian agent", argue_handler=_argue, health_handler=_health, lifespan=lifespan
)
