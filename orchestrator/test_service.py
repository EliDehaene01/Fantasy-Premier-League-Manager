"""Tests for the orchestrator's HTTP wrapper (orchestrator/service.py) -
the live-mode POST /run + POST /resume trigger surface. Injects a fake
graph (same fakes as test_graph.py, InMemorySaver-backed) via the module's
_graph cache rather than a real Postgres checkpointer - this file tests the
HTTP plumbing, not the graph logic itself (already covered by
test_graph.py).
"""

from __future__ import annotations

import os

os.environ["MANAGER_NARRATE_DISABLE_LLM"] = "1"
os.environ["MANAGER_REACT_DISABLE_LLM"] = "1"

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

import orchestrator.service as service_module
from orchestrator.graph import build_graph
from orchestrator.test_graph import _fake_manage, _fake_react, _fake_specialist_factory, _pool


@pytest.fixture(autouse=True)
def _fake_graph():
    service_module._graph = build_graph(
        agent_caller=_fake_specialist_factory(), manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver()
    )
    yield
    service_module._graph = None


@pytest.fixture
def client():
    with TestClient(service_module.app) as c:
        yield c


def test_health():
    with TestClient(service_module.app) as c:
        assert c.get("/health").json() == {"status": "ok"}


def test_run_then_resume_round_trip(client):
    body = {"gameweek": 1, "initial_state": {"player_pool": _pool(), "budget": 100.0, "free_transfers": 1, "hit_cost": 4.0}}
    paused = client.post("/run", json=body).json()

    assert "__interrupt__" in paused
    assert isinstance(paused["__interrupt__"], list)
    assert "proposal" in paused["__interrupt__"][0]["value"]

    resumed = client.post("/resume", json={"gameweek": 1, "decision": "approve"}).json()
    assert resumed["approval_status"] == "approved"
