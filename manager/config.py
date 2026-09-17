"""Fixed, config-driven Manager weights (ARCHITECTURE.md 6b; CLAUDE.md hard
constraint: "Manager aggregation weights are fixed and config-driven, never
decided by an LLM per run" - the same inputs must always produce the same
squad). Change these values to retune; nothing in the Manager's code path
may set them dynamically.
"""

from __future__ import annotations

# Fixtures/Contrarian/Template weighted equally; News weighted higher since
# its RAG-grounded qualitative signal is closer to genuine inside
# information than the other three's more mechanical signals
# (ARCHITECTURE.md 6b). Documented as an untuned baseline to revisit once
# backtest evidence exists.
AGENT_WEIGHTS: dict[str, float] = {
    "fixtures": 1.0,
    "contrarian": 1.0,
    "template": 1.0,
    "news": 1.3,
}


def doubt_multiplier(confidence: float) -> float:
    """News's DOUBT veto applies a steep multiplicative penalty scaled by
    confidence (ARCHITECTURE.md 6b), not a binary exclusion - that's OUT's
    job, handled by the solver's hard candidate-pool filter, not here.
    Linear in confidence: a low-confidence doubt barely dents adjusted_score,
    a near-certain one nearly zeroes it - "steep" relative to the small
    multiplicative nudges the four adjustment agents' conviction scores
    otherwise apply.
    """
    return max(0.0, 1.0 - confidence)
