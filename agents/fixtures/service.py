"""Fixtures agent - FastAPI service.

Same two-step shape as Stats (see agents/stats/service.py's module
docstring): scoring.py (deterministic FDR/blank/double rule) then
reasoning.py (Foundry prose, best-effort, never load-bearing). No ML model,
no dependency on any other agent's output - this agent reads only the
`players`/`fixtures` silver tables.

Never populates `vetoes` - that field belongs to the News agent alone (see
CLAUDE.md's hard constraints). Never sets `predicted_points` on its
recommendations either - fixture favorability isn't a points prediction,
and inventing a number would misrepresent what this agent actually knows.
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

log = logging.getLogger("fixtures_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ingestion.db import get_connection

    app.state.db_conn = None
    try:
        # FIXTURES_AGENT_TEST_SCHEMA lets tests point this at an isolated,
        # seeded schema instead of the real public schema - same
        # test-isolation mechanism ingestion's own test suite already uses
        # (ingestion/db.py::get_connection's `schema` parameter).
        schema = os.environ.get("FIXTURES_AGENT_TEST_SCHEMA")
        app.state.db_conn = get_connection(schema=schema) if schema else get_connection()
    except Exception:  # pragma: no cover - no DB configured/reachable
        log.warning("Fixtures agent: Postgres unavailable at startup - /argue will return no picks", exc_info=True)
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
        agent="fixtures",
        recommendations=[
            Recommendation(player_id=p.player_id, conviction=p.conviction, predicted_points=None)
            for p in picks
        ],
        reasoning=reasoning,
    )


app = create_service(
    title="FPL Fixtures agent", argue_handler=_argue, health_handler=_health, lifespan=lifespan
)
