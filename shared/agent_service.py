"""Shared FastAPI scaffolding + Foundry LLM-call wrapper for every
specialist agent service (Stats, News, Fixtures, Contrarian, Template, ...).

WHY THIS EXISTS
----------------
Before this module, three near-identical copies of the same two patterns
had accumulated:
  1. A tiny OpenAI-SDK client pointed at the Foundry endpoint, used for a
     chat-completion call wrapped in a try/except that falls back to
     something deterministic if the call fails (agents/stats/reasoning.py,
     agents/news/tier2.py, and agents/news/embeddings.py each had their own
     copy of the client construction).
  2. The same FastAPI service skeleton - load .env, build the app, expose
     GET /health and POST /argue returning the shared contract
     (agents/stats/service.py and agents/news/service.py each wrote this
     out in full).
Adding three more agents (Fixtures, Contrarian, Template) would have made
that a FOURTH+ copy of pattern 1 and a THIRD+ of pattern 2. This module is
that shared home instead - Stats and News were migrated onto it too (see
their reasoning.py/service.py), not just the three new agents, so the
duplication is actually eliminated, not just stopped from growing further.

Each agent still owns its own scoring logic, system prompt, and fallback
sentence - only the mechanical wiring (the client, the try/except-and-
fallback shape, the route definitions) lives here.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

from fastapi import FastAPI

from .contracts import AgentArgument, ArgueRequest

logger = logging.getLogger("shared.agent_service")

FOUNDRY_ENDPOINT_ENV = "MICROSOFT_FOUNDRY_OPENAI_ENDPOINT"
FOUNDRY_KEY_ENV = "MICROSOFT_FOUNDRY_KEY"


def default_foundry_client(*, timeout: float = 12.0, max_retries: int = 1):
    """An OpenAI-SDK client pointed at the Foundry deployment. Foundry
    exposes an OpenAI-compatible ``/openai/v1`` surface, so the plain
    ``OpenAI`` client works with ``base_url`` set to the Foundry endpoint.
    Raises if the env vars are missing - callers turn that into a fallback.
    """
    from openai import OpenAI

    return OpenAI(
        api_key=os.environ[FOUNDRY_KEY_ENV],
        base_url=os.environ[FOUNDRY_ENDPOINT_ENV],
        timeout=timeout,
        max_retries=max_retries,
    )


def llm_disabled(env_var: str) -> bool:
    """An explicit off switch for tests and offline/backtest runs, so a
    call to Foundry is never made where one isn't wanted. Each agent passes
    its own env var name (e.g. ``STATS_AGENT_DISABLE_LLM``,
    ``FIXTURES_AGENT_DISABLE_LLM``) so agents can be disabled independently.
    """
    return os.getenv(env_var) == "1"


def call_foundry_llm(
    *,
    model: str,
    system_prompt: str,
    user_content: str,
    max_completion_tokens: int,
    json_response: bool = False,
    client_factory: Callable[[], object] = default_foundry_client,
) -> str:
    """One Foundry chat-completion call. Returns the raw text content
    (``json.loads`` it yourself if ``json_response=True`` - this function
    stays agnostic to what shape each agent expects back, since some agents
    want plain prose and some want structured JSON).

    Raises on ANY failure - missing credentials, network, rate limit, or an
    empty completion. Deliberately does not catch anything itself: what
    "safe" means after a failure differs per agent (a templated sentence, a
    Tier 1 answer, an empty list), so the fallback decision belongs to the
    caller, not this shared plumbing.
    """
    client = client_factory()
    kwargs = {"response_format": {"type": "json_object"}} if json_response else {}
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_completion_tokens=max_completion_tokens,
        **kwargs,
    )
    text = (completion.choices[0].message.content or "").strip()
    if not text:
        raise ValueError("empty completion from Foundry")
    return text


def create_service(
    *,
    title: str,
    argue_handler: Callable[[FastAPI, ArgueRequest], AgentArgument],
    health_handler: Callable[[FastAPI], dict] | None = None,
    lifespan=None,
) -> FastAPI:
    """Build the standard specialist-agent FastAPI app: load .env, expose
    GET /health and POST /argue.

    Each agent still owns its OWN lifespan (what to load/connect at
    startup - a model, a DB connection, both, neither) and its own
    ``argue_handler`` (the actual scoring/reasoning logic, given the app so
    it can reach ``app.state``) - this just wires the two into the same
    shape every agent's service.py used to write out by hand.
    """
    try:  # load .env locally; harmless if python-dotenv isn't installed in prod
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass

    app = FastAPI(title=title, version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return health_handler(app) if health_handler is not None else {"status": "ok"}

    @app.post("/argue", response_model=AgentArgument)
    def argue(request: ArgueRequest) -> AgentArgument:
        return argue_handler(app, request)

    return app
