"""The 'prose' half of the Contrarian agent - same two-step split as
Stats'/Fixtures' reasoning.py. Uses the shared Foundry wrapper
(shared/agent_service.py) instead of its own client/try-except copy.
"""

from __future__ import annotations

import logging
import os

from shared.agent_service import call_foundry_llm, default_foundry_client, llm_disabled

from .scoring import Pick

logger = logging.getLogger("contrarian_agent.reasoning")

FOUNDRY_DEPLOYMENT = os.getenv("CONTRARIAN_AGENT_LLM_MODEL", "gpt-5.4-nano")

SYSTEM_PROMPT = (
    "You are the Contrarian specialist on a Fantasy Premier League selection committee, "
    "arguing for low-ownership, high-upside differential picks that could climb rank. "
    "A deterministic rule has already ranked the players below by the gap between their "
    "predicted points (from the Stats agent's model) and how low their ownership is. "
    "Write a SHORT argument - 2 to 4 sentences, under 90 words total - for backing these "
    "differentials. Ground every claim in the listed factors: name specific predicted "
    "points and ownership percentages. Do NOT invent numbers or mention players not "
    "listed. Lead with the strongest pick. Plain prose, no bullet points."
)


def _fallback_reasoning(gameweek: int, picks: list[Pick]) -> str:
    if not picks:
        return "No genuinely differentiated low-ownership picks stood out this gameweek."
    lead = picks[0]
    who = lead.name or f"player {lead.player_id}"
    factor = "; ".join(lead.factors) if lead.factors else "a strong quality-vs-ownership gap"
    text = f"For GW{gameweek} the top differential is {who} - {factor}."
    if len(picks) > 1:
        others = ", ".join(p.name or f"player {p.player_id}" for p in picks[1:3])
        text += f" Also worth a punt: {others}."
    return text


def _build_user_payload(gameweek: int, picks: list[Pick]) -> str:
    lines = [f"Gameweek: {gameweek}", "Ranked differential picks (strongest gap first):"]
    for rank, p in enumerate(picks, start=1):
        who = p.name or f"player {p.player_id}"
        factors = "; ".join(p.factors) or "no detail"
        lines.append(f"{rank}. {who} - conviction {p.conviction:.2f}. {factors}.")
    return "\n".join(lines)


def generate_reasoning(gameweek: int, picks: list[Pick], *, client_factory=default_foundry_client) -> str:
    if not picks:
        return _fallback_reasoning(gameweek, picks)
    if llm_disabled("CONTRARIAN_AGENT_DISABLE_LLM"):
        return _fallback_reasoning(gameweek, picks)
    try:
        return call_foundry_llm(
            model=FOUNDRY_DEPLOYMENT,
            system_prompt=SYSTEM_PROMPT,
            user_content=_build_user_payload(gameweek, picks),
            max_completion_tokens=400,
            client_factory=client_factory,
        )
    except Exception as exc:  # noqa: BLE001 - genuinely want to catch everything here
        logger.warning("Foundry reasoning call failed (%s); using templated fallback", exc)
        return _fallback_reasoning(gameweek, picks)
