"""Default HTTP callers the graph nodes use to reach each service
(ARCHITECTURE.md 6c: "a graph node calls out to that service over HTTP
rather than running the agent's logic in-process"). graph.py's
``build_graph`` takes these as injectable parameters - tests substitute
fakes so the graph's own wiring (fan-out/fan-in, reaction round, mode
branching, checkpointing) is verifiable without six live services and a
database.
"""

from __future__ import annotations

import os

import httpx

from manager.schemas import ManageRequest, ManageResult, ReactionRequest
from shared.contracts import AgentArgument, ArgueRequest, PlayerEntry

SPECIALIST_URL_ENV = {
    "stats": "STATS_AGENT_URL",
    "fixtures": "FIXTURES_AGENT_URL",
    "news": "NEWS_AGENT_URL",
    "contrarian": "CONTRARIAN_AGENT_URL",
    "template": "TEMPLATE_AGENT_URL",
    "chips": "CHIPS_AGENT_URL",
}
MANAGER_URL_ENV = "MANAGER_URL"
# The debate transcript only wants a short list (see each agent's own
# top_k), but the Manager needs the full affordable candidate pool to build
# a legal squad - the caller (this module) is responsible for asking for
# enough coverage, per ArgueRequest's own top_k cap of 50.
FULL_POOL_TOP_K = 50
# News's Tier 2 (agents/news/tier2.py) does a real per-player RAG round trip
# (embeddings + pgvector retrieval, and a classifier or LLM call) for every
# unresolved availability case AND every player in the positive-coverage
# scan - unlike the other five specialists, which score the whole pool in
# one pass. 30s (fine for those) isn't enough once Tier 2 is actually doing
# real network calls instead of failing fast on a config error - found by
# running a real gameweek through the deployed cluster.
SPECIALIST_TIMEOUT_SECONDS = {"news": 300.0}
DEFAULT_TIMEOUT_SECONDS = 30.0


def call_specialist(agent_name: str, gameweek: int, player_pool: list[dict]) -> AgentArgument:
    base_url = os.environ[SPECIALIST_URL_ENV[agent_name]]
    request = ArgueRequest(
        gameweek=gameweek,
        players=[PlayerEntry.model_validate(p) for p in player_pool],
        top_k=min(FULL_POOL_TOP_K, max(len(player_pool), 1)),
    )
    timeout = SPECIALIST_TIMEOUT_SECONDS.get(agent_name, DEFAULT_TIMEOUT_SECONDS)
    resp = httpx.post(f"{base_url}/argue", json=request.model_dump(mode="json"), timeout=timeout)
    resp.raise_for_status()
    return AgentArgument.model_validate(resp.json())


def call_manage(request: ManageRequest) -> ManageResult:
    base_url = os.environ[MANAGER_URL_ENV]
    resp = httpx.post(f"{base_url}/manage", json=request.model_dump(mode="json"), timeout=30.0)
    resp.raise_for_status()
    return ManageResult.model_validate(resp.json())


def call_react(gameweek: int, first_round: dict[str, AgentArgument]) -> dict[str, str]:
    base_url = os.environ[MANAGER_URL_ENV]
    request = ReactionRequest(gameweek=gameweek, first_round=first_round)
    resp = httpx.post(f"{base_url}/react", json=request.model_dump(mode="json"), timeout=30.0)
    resp.raise_for_status()
    return resp.json()
