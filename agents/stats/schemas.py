"""Request / response models for the Stats agent service.

The response shape defined here (`AgentArgument`) is the **standard contract
for every specialist agent** in this project (stats, fixtures, injuries,
contrarian, template, chips). If you are writing another specialist, import
these same field names - the Manager agent aggregates all six on the
assumption that they look identical.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PlayerEntry(BaseModel):
    """One player in the pool the caller wants scored.

    ``features`` is the dict of engineered model features for this player
    *as of the upcoming gameweek's deadline* - e.g. ``pts_roll5``,
    ``minutes_roll3``, ``fixture_adj_xp`` ... (the full list is
    ``bundle["features"]`` inside ``models/stats_model.pkl``, and the code
    that builds them from raw FPL data lives in ``agents/stats/features.py``).

    Producing those features is the ingestion service's job, not this
    service's - the specialist stays thin and just scores what it's handed.
    Any feature the model expects but the caller omits is treated as 0.0,
    exactly as during training; unknown extra keys are ignored.
    """

    model_config = ConfigDict(extra="ignore")

    player_id: int
    features: dict[str, float] = Field(default_factory=dict)
    # Optional, purely so the reasoning text can name the player instead of
    # printing an id. Not used in any calculation.
    name: str | None = None
    position: str | None = None


class ArgueRequest(BaseModel):
    gameweek: int = Field(ge=1, le=38)
    players: list[PlayerEntry] = Field(default_factory=list)
    # How many of the top-scoring players to actually argue for. Clamped to
    # the pool size. The debate wants a short list, not the whole player
    # universe.
    top_k: int = Field(default=5, ge=1, le=50)


class Recommendation(BaseModel):
    player_id: int
    # conviction is a RELATIVE push, NOT a probability and NOT a confidence
    # score. It answers "of the players I'm recommending, how hard am I
    # arguing for this one compared with the others?" - computed as this
    # player's predicted points over the top pick's predicted points, so the
    # strongest pick is 1.0 and the rest scale down. Two different agents both
    # returning 1.0 for their own top pick is expected and fine; the Manager
    # uses conviction only to read a single agent's internal ordering.
    conviction: float = Field(ge=0.0, le=1.0)
    predicted_points: float


class AgentArgument(BaseModel):
    """THE shared specialist-agent contract. Do not reshape casually."""

    agent: str = "stats"
    recommendations: list[Recommendation] = Field(default_factory=list)
    # 2-4 sentences of natural language for the debate transcript. Grounded in
    # the model's SHAP factors, written by the LLM (or a templated fallback if
    # the LLM call fails).
    reasoning: str
