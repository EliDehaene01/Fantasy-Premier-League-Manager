"""Chips agent - FastAPI service.

Reasons about the fixture CALENDAR SHAPE for our own tracked squad
(my_team_state), not the player pool the caller passes in -
``request.players``/``request.top_k`` are ignored entirely (see scoring.py's
module docstring); only ``request.gameweek`` is used. The pool the debate
passes around is whatever candidate players the OTHER specialists are
arguing about, which isn't necessarily our actual 15-man squad - Chips has
to source its own squad independently regardless (see
``scoring._our_squad_player_ids``).

Never populates ``recommendations`` or ``vetoes`` - this agent argues for
WHEN to play a chip, not WHICH players to pick. Populates
``chip_recommendation`` instead (see shared/contracts.py).

THE CHIP DECISION IS NEVER MADE BY THE LLM: ``scoring.evaluate_chip_timing``
(deterministic, DB-only) decides the verdict BEFORE ``reasoning.py`` is ever
called - the LLM only writes prose describing an already-fixed decision, the
same "LLM never decides, only narrates" split every other agent in this
project uses (see agents/stats/service.py's module docstring). So "the
fallback case should default to chip: None rather than guessing" is true
even more strongly than a simple fallback: the LLM is never in a position to
invent a chip at all, on success OR failure.
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.agent_service import create_service
from shared.contracts import AgentArgument, ArgueRequest, ChipRecommendation

from . import scoring
from .reasoning import generate_reasoning

log = logging.getLogger("chips_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from ingestion.db import get_connection

    app.state.db_conn = None
    try:
        # CHIPS_AGENT_TEST_SCHEMA lets tests point this at an isolated,
        # seeded schema - same mechanism ingestion's own test suite and the
        # Fixtures/Contrarian/Template agents' tests use.
        schema = os.environ.get("CHIPS_AGENT_TEST_SCHEMA")
        app.state.db_conn = get_connection(schema=schema) if schema else get_connection()
    except Exception:  # pragma: no cover - no DB configured/reachable
        log.warning("Chips agent: Postgres unavailable at startup - /argue will return chip=None", exc_info=True)
    yield
    if app.state.db_conn is not None:
        app.state.db_conn.close()


def _health(app: FastAPI) -> dict:
    return {"status": "ok", "db_connected": app.state.db_conn is not None}


def _argue(app: FastAPI, request: ArgueRequest) -> AgentArgument:
    conn = app.state.db_conn
    if conn is None:
        verdict = scoring.ChipVerdict(chip=None, confidence=0.0, reasoning_factors=["no database connection available"])
    else:
        verdict = scoring.evaluate_chip_timing(conn, request.gameweek)

    reasoning = generate_reasoning(request.gameweek, verdict)
    return AgentArgument(
        agent="chips",
        recommendations=[],
        vetoes=[],
        chip_recommendation=ChipRecommendation(
            chip=verdict.chip,
            confidence=verdict.confidence,
            reasoning="; ".join(verdict.reasoning_factors) or reasoning,
        ),
        reasoning=reasoning,
    )


app = create_service(
    title="FPL Chips agent", argue_handler=_argue, health_handler=_health, lifespan=lifespan
)
