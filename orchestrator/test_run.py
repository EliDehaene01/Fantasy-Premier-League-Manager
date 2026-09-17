"""Tests for the entrypoint wrappers (orchestrator/run.py) - the thread-id
scheme and the backtest/live/resume wiring, using an injected graph built
with fakes (same fakes as test_graph.py) so no live services or Postgres
are needed.
"""

from __future__ import annotations

import os

os.environ["MANAGER_NARRATE_DISABLE_LLM"] = "1"
os.environ["MANAGER_REACT_DISABLE_LLM"] = "1"

from langgraph.checkpoint.memory import InMemorySaver

from orchestrator.graph import build_graph
from orchestrator.run import resume_live_gameweek, run_backtest_gameweek, run_live_gameweek
from orchestrator.test_graph import _fake_manage, _fake_react, _fake_specialist_factory, _pool


def _app():
    return build_graph(agent_caller=_fake_specialist_factory(), manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver())


def test_run_backtest_gameweek_auto_accepts():
    result = run_backtest_gameweek(7, {"player_pool": _pool(), "budget": 100.0, "free_transfers": 1, "hit_cost": 4.0}, graph=_app())
    assert result["approval_status"] == "auto_accepted"
    assert result["gameweek"] == 7


def test_run_live_gameweek_then_resume_approve():
    app = _app()
    paused = run_live_gameweek(8, {"player_pool": _pool(), "budget": 100.0, "free_transfers": 1, "hit_cost": 4.0}, graph=app)
    assert "__interrupt__" in paused

    resumed = resume_live_gameweek(8, "approve", graph=app)
    assert resumed["approval_status"] == "approved"


def test_different_gameweeks_get_independent_thread_ids():
    """Two live gameweeks run through the same graph instance must not
    collide - each needs its own checkpointed thread so gameweek 9's
    pending approval doesn't leak into gameweek 10's.
    """
    app = _app()
    run_live_gameweek(9, {"player_pool": _pool(), "budget": 100.0, "free_transfers": 1, "hit_cost": 4.0}, graph=app)
    run_live_gameweek(10, {"player_pool": _pool(), "budget": 100.0, "free_transfers": 1, "hit_cost": 4.0}, graph=app)

    resumed_9 = resume_live_gameweek(9, "approve", graph=app)
    resumed_10 = resume_live_gameweek(10, "reject", graph=app)

    assert resumed_9["approval_status"] == "approved"
    assert resumed_10["approval_status"] == "rejected"
