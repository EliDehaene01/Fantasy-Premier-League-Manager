"""News agent - FastAPI service.

One endpoint, ``POST /argue``, same shared contract every specialist uses
(``shared/contracts.py``). Unlike Stats, this agent's response actually
uses BOTH output fields: ``vetoes`` (availability, a hard constraint on the
Manager) and ``recommendations`` (notable positive coverage, a normal vote)
- see agents/news/__init__.py for why those are different kinds of output
from the same agent, and tier1.py/tier2.py for why the work is split into
two tiers.

Per player, in order:
  1. Tier 1 (tier1.py) - free, deterministic. Resolves most players outright.
  2. Tier 2 availability (tier2.py) - only for players Tier 1 left
     unresolved; RAG + LLM, with Tier 1's (non-)answer as the fallback.
Then, once per request (not once per player):
  3. Tier 2 positive-coverage scan across the whole pool - a separate job,
     feeding `recommendations` instead of `vetoes`.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from shared.agent_service import create_service
from shared.contracts import AgentArgument, ArgueRequest, Veto

from . import tier2
from .tier1 import check_tier1

log = logging.getLogger("news_agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_conn = None
    try:
        from ingestion.db import get_connection, register_pgvector_adapter

        conn = get_connection()
        register_pgvector_adapter(conn)
        app.state.db_conn = conn
    except Exception:
        # No DB, or pgvector not installed yet - Tier 1 still works fully
        # (it never touches the database); only Tier 2 degrades, and it
        # already has its own per-call fallback (see tier2.py) for exactly
        # this situation, so a missing vector store doesn't break /argue.
        log.warning("News agent: Postgres/pgvector unavailable at startup - Tier 2 will fall back", exc_info=True)
    yield
    if app.state.db_conn is not None:
        app.state.db_conn.close()


def _health(app: FastAPI) -> dict:
    return {"status": "ok", "db_connected": app.state.db_conn is not None}


def _argue(app: FastAPI, request: ArgueRequest) -> AgentArgument:
    conn = app.state.db_conn
    vetoes: list[Veto] = []
    unresolved: list[tuple[int, str]] = []

    for player in request.players:
        veto = check_tier1(player.player_id, player.chance_of_playing_this_round, player.news)
        if veto is not None:
            vetoes.append(veto)
        else:
            unresolved.append((player.player_id, player.name or f"player {player.player_id}"))

    if unresolved and conn is not None:
        for player_id, name in unresolved:
            veto = tier2.resolve_availability(conn, player_id, name, fallback=None)
            if veto is not None:
                vetoes.append(veto)

    recommendations, coverage_reasoning = [], None
    if conn is not None:
        pool = [(p.player_id, p.name or f"player {p.player_id}") for p in request.players]
        recommendations, coverage_reasoning = tier2.find_notable_positive_coverage(conn, pool)

    if coverage_reasoning:
        reasoning = coverage_reasoning
    elif vetoes:
        out = [v.player_id for v in vetoes if v.status.value == "OUT"]
        reasoning = (
            f"No notable positive coverage found this gameweek. Availability flags: "
            f"{len(vetoes)} player(s) checked, {len(out)} ruled OUT."
        )
    else:
        reasoning = "No availability concerns or notable positive coverage found this gameweek."

    return AgentArgument(agent="news", recommendations=recommendations, vetoes=vetoes, reasoning=reasoning)


app = create_service(
    title="FPL News agent", argue_handler=_argue, health_handler=_health, lifespan=lifespan
)
