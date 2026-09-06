"""Stats agent - FastAPI service.

One endpoint, ``POST /argue``. It answers the question the debate asks every
specialist: "given this player pool and this gameweek, who are you pushing
for and why?"

HOW A REQUEST FLOWS - and why it's two steps, not one
----------------------------------------------------
1. ``StatsModel.score()`` (model_runtime.py) runs the trained model over the
   pool, ranks it, sets ``conviction``, and pulls each pick's SHAP factors.
   This is the part that must be correct and reproducible: the same input
   always gives the same players, points and ordering. No LLM anywhere near it.

2. ``generate_reasoning()`` (reasoning.py) takes *only the output of step 1* -
   the ranked picks and their SHAP factors - and asks the Foundry LLM to
   write it up as a couple of sentences for the transcript.

They are separate because they have different failure modes and different
guarantees. Step 1 is deterministic and offline. Step 2 is a network call
that can time out or rate-limit; when it does, we return the step-1 result
with a templated sentence instead. A flaky LLM must never cost us a
recommendation, and the LLM must never be able to change who we recommend.

The model is loaded once, in the lifespan handler, and reused for every
request - loading it (and building the SHAP explainer) per request would add
hundreds of milliseconds for no reason.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .model_runtime import StatsModel
from .reasoning import generate_reasoning
from .schemas import AgentArgument, ArgueRequest, Recommendation

try:  # load .env locally; harmless if python-dotenv isn't installed in prod
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup. Everything expensive and reusable goes here.
    app.state.model = StatsModel()
    yield
    # nothing to tear down


app = FastAPI(title="FPL Stats agent", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    """Liveness probe for Kubernetes; also confirms which model is loaded."""
    model: StatsModel = app.state.model
    return {"status": "ok", "model": model.model_name, "n_features": len(model.features)}


@app.post("/argue", response_model=AgentArgument)
def argue(request: ArgueRequest) -> AgentArgument:
    """Score the pool, rank it, and return this agent's argument.

    Returns the shared specialist-agent contract (see schemas.AgentArgument).
    An empty pool is a valid request - it just yields an empty recommendation
    list and a one-line explanation, never an error.
    """
    model: StatsModel = app.state.model

    # --- step 1: the numbers (trained model, deterministic) ---------------
    picks = model.score(request.players, top_k=request.top_k)

    # --- step 2: the prose (LLM, best-effort, never load-bearing) ---------
    reasoning = generate_reasoning(request.gameweek, picks)

    return AgentArgument(
        agent="stats",
        recommendations=[
            Recommendation(
                player_id=pick.player_id,
                conviction=pick.conviction,
                predicted_points=pick.predicted_points,
            )
            for pick in picks
        ],
        reasoning=reasoning,
    )
