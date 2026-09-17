"""Request / response models for every specialist agent service.

The response shape defined here (`AgentArgument`) is the **standard contract
for every specialist agent** in this project (stats, fixtures, news,
contrarian, template, chips). If you are writing another specialist, import
these same field names - the Manager agent aggregates all of them on the
assumption that they look identical.

This used to live in `agents/stats/schemas.py` (written there first, when
Stats was the only agent) and was relocated here once a second, third,
fourth and fifth agent all needed to import it too - a specialist agent
importing from a package literally named "stats" for its own contract was
misleading coupling that would only have gotten worse as more agents were
added. `agents/stats/` and `agents/news/` were both migrated to import from
here; neither's behavior changed as part of that move (see their own test
suites, unchanged and still passing).
"""

from __future__ import annotations

from enum import Enum

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
    # Structured availability fields from silver's `bootstrap-static`-sourced
    # data, needed by the News agent's Tier 1 deterministic check
    # (agents/news/tier1.py) - not features, so not folded into the dict
    # above. Optional and unused by Stats, which never sets them.
    chance_of_playing_this_round: float | None = None
    news: str | None = None
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
    # Optional because not every specialist's recommendation comes with a
    # real predicted-points number. Stats always sets this (it's the model's
    # actual output); Contrarian sets it too (read from the predictions log -
    # see agents/contrarian/scoring.py). The News agent's "notable positive
    # coverage" pick and the Fixtures/Template agents' picks are qualitative
    # judgments from a different signal entirely - there is no quantitative
    # prediction behind them, and making one up (e.g. defaulting to 0.0)
    # would let the Manager's objective function silently treat "no data" as
    # "predicted zero points", which is wrong in the opposite direction from
    # the truth. None means exactly that: no quantitative prediction, weigh
    # this on conviction/reasoning alone.
    predicted_points: float | None = None


class VetoStatus(str, Enum):
    OUT = "OUT"
    DOUBT = "DOUBT"
    FIT = "FIT"


class ChipType(str, Enum):
    WILDCARD = "wildcard"
    BENCH_BOOST = "bench_boost"
    TRIPLE_CAPTAIN = "triple_captain"
    FREE_HIT = "free_hit"


class ChipRecommendation(BaseModel):
    """The Chips agent's timing verdict - the News agent's `Veto` has an
    exact parallel here: a self-contained judgment with its own confidence
    and grounding text, not folded into `recommendations` (which argues for
    PLAYERS) or `vetoes` (which is News-only). Lives here only because this
    is the shared contract file - never produced by any other specialist.
    """

    # None is the expected, common answer - most gameweeks have no fixture
    # shape unusual enough to justify burning a chip. See
    # agents/chips/scoring.py's module docstring for why defaulting to
    # "don't play anything" is correct behavior, not a fallback to apologize
    # for.
    chip: ChipType | None = None
    # How confident the agent is in THIS verdict - including the verdict
    # "don't play a chip yet". A quiet, ordinary gameweek should produce
    # high confidence in chip=None, not low confidence just because nothing
    # was recommended.
    confidence: float = Field(ge=0.0, le=1.0)
    # Why - which fixture-calendar shape (or absence of one) drove this
    # verdict. Always populated, even for chip=None: "no chip is right" is
    # a real, explainable conclusion, not a state with nothing to say.
    reasoning: str


class Veto(BaseModel):
    """One player's availability verdict - the News agent's hard-constraint
    output (see `AgentArgument.vetoes` below). Never produced by any other
    specialist; lives here only because this is the shared contract file.
    """

    player_id: int
    status: VetoStatus
    # How sure the agent is in THIS verdict, not a probability of playing -
    # a Tier 1 rule match (0% chance, explicit "ruled out" text) is high
    # confidence; an LLM's read of an ambiguous team-news snippet is lower.
    confidence: float = Field(ge=0.0, le=1.0)
    # The actual source text the verdict is grounded in - a Tier 1 rule cites
    # the raw `news`/`chance_of_playing_this_round` value; a Tier 2 verdict
    # cites the retrieved passage it read. Never fabricated: if there is no
    # supporting text, this is None, not a made-up sentence.
    grounding_snippet: str | None = None


class AgentArgument(BaseModel):
    """THE shared specialist-agent contract. Do not reshape casually."""

    agent: str = "stats"
    recommendations: list[Recommendation] = Field(default_factory=list)
    # Hard-constraint availability verdicts (see ARCHITECTURE.md's "Hard
    # constraints" section) - empty for every specialist except News. The
    # Manager must never select a player who appears here with status OUT or
    # DOUBT, regardless of any agent's predicted points or conviction. This
    # lives on the shared contract (not a News-only schema) so the Manager
    # can read it identically off any agent's response without a special
    # case, even though only News ever populates it.
    vetoes: list[Veto] = Field(default_factory=list)
    # The Chips agent's timing verdict (see `ChipRecommendation` above) -
    # None for every specialist except Chips, which argues for WHEN to play
    # a chip rather than WHICH players to pick, so it leaves
    # `recommendations` and `vetoes` both empty and uses this field instead.
    chip_recommendation: ChipRecommendation | None = None
    # 2-4 sentences of natural language for the debate transcript. Grounded in
    # each agent's own deterministic factors, written by the LLM (or a
    # templated fallback if the LLM call fails).
    reasoning: str
