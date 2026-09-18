"""Thin entrypoints wrapping the compiled graph (ARCHITECTURE.md 6c) - what
the backtest engine (Phase 2) and the live weekly trigger (Phase 5's
CronJob) actually call. Owns the Postgres checkpointer wiring for live/
production use; the orchestrator's own test suite builds its graph with an
InMemorySaver instead (no live Postgres needed to test the graph's wiring).
"""

from __future__ import annotations

import os

from langgraph.types import Command

from .graph import build_graph

DATABASE_URL_ENV = "DATABASE_URL"


def _thread_id(mode: str, gameweek: int) -> str:
    return f"{mode}-gw{gameweek}"


def postgres_checkpointer():
    """A PostgresSaver connected to $DATABASE_URL, with its tables ensured.
    Import kept local to this function - langgraph.checkpoint.postgres pulls
    in psycopg, which callers that only need the in-memory test path
    shouldn't have to have installed.

    ``PostgresSaver.from_conn_string`` is a ``@contextmanager`` wrapping
    ``with Connection.connect(...) as conn: yield cls(conn)`` - the
    connection is only kept open for as long as that generator's frame is
    alive. Calling ``.__enter__()`` manually (instead of a ``with`` block)
    advances it to the yield and returns the checkpointer, but if the
    generator object itself (``checkpointer_cm`` below) isn't kept
    referenced somewhere, it gets garbage-collected once this function
    returns - which closes the connection via ``GeneratorExit``, so the
    very next real query fails with "the connection is closed" (caught by
    actually calling this against the live k8s deployment; every test
    before that used InMemorySaver and never exercised this path at all).
    Stashing it as an attribute keeps it alive exactly as long as the
    checkpointer object itself is.
    """
    from langgraph.checkpoint.postgres import PostgresSaver

    checkpointer_cm = PostgresSaver.from_conn_string(os.environ[DATABASE_URL_ENV])
    checkpointer = checkpointer_cm.__enter__()
    checkpointer.setup()
    checkpointer._conn_string_cm = checkpointer_cm  # keep the generator (and its connection) alive
    return checkpointer


def run_backtest_gameweek(gameweek: int, initial_state: dict, *, graph=None) -> dict:
    """Backtest mode never pauses (ARCHITECTURE.md 6c) - runs straight
    through to an auto-accepted or declined result. ``graph`` is injectable
    for tests; a production caller builds one via
    ``build_graph(checkpointer=postgres_checkpointer())`` once per season
    run, not once per gameweek.
    """
    app = graph or build_graph()
    state = {**initial_state, "mode": "backtest", "gameweek": gameweek}
    config = {"configurable": {"thread_id": _thread_id("backtest", gameweek)}}
    return app.invoke(state, config=config)


def run_live_gameweek(gameweek: int, initial_state: dict, *, graph=None) -> dict:
    """Starts a live gameweek's run - returns with ``__interrupt__`` set,
    holding at the approval node. Call ``resume_live_gameweek`` with the
    human's decision to continue from exactly that point.
    """
    app = graph or build_graph()
    state = {**initial_state, "mode": "live", "gameweek": gameweek}
    config = {"configurable": {"thread_id": _thread_id("live", gameweek)}}
    return app.invoke(state, config=config)


def resume_live_gameweek(gameweek: int, decision: str, *, graph=None) -> dict:
    """``decision`` is "approve" or anything else (rejected). No retry loop
    - ARCHITECTURE.md 6c: rejection/timeout just logs as declined.
    """
    app = graph or build_graph()
    config = {"configurable": {"thread_id": _thread_id("live", gameweek)}}
    return app.invoke(Command(resume=decision), config=config)
