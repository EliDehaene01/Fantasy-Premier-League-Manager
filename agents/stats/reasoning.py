"""The 'prose' half of the Stats agent.

This module's only job is to turn the model's ranked picks + SHAP factors
into 2-4 readable sentences for the debate transcript. It does NOT decide
who to recommend, in what order, or with what predicted points - all of that
is already fixed by ``model_runtime.py`` before this module is called.

Why a separate step at all?
  * The ranking and numbers must be reproducible and auditable. An LLM is
    neither, so it is kept away from anything numeric.
  * The LLM is a *rendering* layer. If the Foundry call is slow, rate-limited
    or misconfigured, we still want to return a correct recommendation - so
    every failure path here falls back to a deterministic templated sentence
    built from the same SHAP factors. The service degrades; it never breaks.
"""

from __future__ import annotations

import logging
import os

from shared.agent_service import call_foundry_llm, default_foundry_client, llm_disabled

from .model_runtime import Pick

logger = logging.getLogger("stats_agent.reasoning")

# The Foundry deployment we call. Overridable by env for staging/testing, but
# the credentials themselves reuse the existing MICROSOFT_FOUNDRY_* variables
# from .env - we do not introduce new names for the same secret.
FOUNDRY_DEPLOYMENT = os.getenv("STATS_AGENT_LLM_MODEL", "gpt-5.4-nano")

SYSTEM_PROMPT = (
    "You are the Stats specialist on a Fantasy Premier League selection committee. "
    "A trained points-prediction model has already chosen and ranked the players "
    "below and computed the statistical factors behind each pick. "
    "Write a SHORT argument - 2 to 4 sentences, under 90 words total - for backing "
    "these players this gameweek. "
    "Ground every claim in the provided factors: name the specific signals "
    "(e.g. rising minutes, season points-per-game, a soft fixture). "
    "Do NOT invent statistics, predict exact scorelines, or mention players not listed. "
    "Lead with the strongest pick. Plain prose, no bold, no bullet points."
)


def _factor_clause(pick: Pick) -> str:
    """A plain-English 'because ...' clause from a pick's SHAP factors."""
    ups = [f.phrase for f in pick.factors if f.direction == "raises"]
    downs = [f.phrase for f in pick.factors if f.direction == "lowers"]
    parts: list[str] = []
    if ups:
        verb = "are" if len(ups[:2]) > 1 else "is"
        parts.append(f"{', '.join(ups[:2])} {verb} pushing the model toward him")
    if downs:
        parts.append(f"though {downs[0]} is a mild drag")
    return "; ".join(parts) if parts else "the model rates him on recent form"


def _fallback_reasoning(gameweek: int, picks: list[Pick]) -> str:
    """Deterministic sentence used whenever the LLM call can't be made.

    Still SHAP-grounded and still specific - it just isn't as fluent as the
    model-written version.
    """
    if not picks:
        return "No players in the pool scored highly enough to argue for this gameweek."

    lead = picks[0]
    who = lead.name or f"player {lead.player_id}"
    text = (
        f"For GW{gameweek} the model's top pick is {who} "
        f"({lead.predicted_points:.1f} pts projected) - {_factor_clause(lead)}."
    )
    if len(picks) > 1:
        others = ", ".join(p.name or f"player {p.player_id}" for p in picks[1:3])
        text += f" It also backs {others} on similar form and fixture signals."
    return text


def _build_user_payload(gameweek: int, picks: list[Pick]) -> str:
    lines = [f"Gameweek: {gameweek}", "Ranked picks (strongest first):"]
    for rank, p in enumerate(picks, start=1):
        who = p.name or f"player {p.player_id}"
        pos = f", {p.position}" if p.position else ""
        factors = "; ".join(
            f"{f.phrase} {f.direction} the prediction" for f in p.factors
        ) or "recent form"
        lines.append(
            f"{rank}. {who}{pos} - projected {p.predicted_points:.1f} pts, "
            f"conviction {p.conviction:.2f}. Factors: {factors}."
        )
    return "\n".join(lines)


def generate_reasoning(gameweek: int, picks: list[Pick], *, client_factory=default_foundry_client) -> str:
    """Return the debate argument for these picks.

    Tries the Foundry LLM first (via shared.agent_service.call_foundry_llm);
    on ANY problem (missing creds, network, rate limit, empty completion)
    logs it and returns the templated fallback. ``client_factory`` is
    injectable so tests can stub the LLM.
    """
    if not picks:
        return _fallback_reasoning(gameweek, picks)

    # An explicit off switch for tests and offline/backtest runs, so we never
    # make a real API call where one isn't wanted.
    if llm_disabled("STATS_AGENT_DISABLE_LLM"):
        return _fallback_reasoning(gameweek, picks)

    try:
        return call_foundry_llm(
            model=FOUNDRY_DEPLOYMENT,
            system_prompt=SYSTEM_PROMPT,
            user_content=_build_user_payload(gameweek, picks),
            # generous headroom: gpt-5-class models spend some of the budget on
            # internal reasoning tokens before the visible answer, so too small
            # a cap gets truncated mid-sentence.
            max_completion_tokens=600,
            client_factory=client_factory,
        )
    except Exception as exc:  # noqa: BLE001 - we genuinely want to catch everything here
        logger.warning("Foundry reasoning call failed (%s); using templated fallback", exc)
        return _fallback_reasoning(gameweek, picks)
