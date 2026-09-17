"""The 'prose' half of the Fixtures agent - same two-step split as Stats'
reasoning.py (see agents/stats/service.py's module docstring for why):
scoring.py is deterministic and reproducible, this module turns its output
into 2-4 sentences via Foundry, with a templated fallback if the call fails
or is disabled. Uses the shared Foundry wrapper (shared/agent_service.py)
instead of its own client/try-except copy.
"""

from __future__ import annotations

import logging
import os

from shared.agent_service import call_foundry_llm, default_foundry_client, llm_disabled

from .scoring import Pick

logger = logging.getLogger("fixtures_agent.reasoning")

FOUNDRY_DEPLOYMENT = os.getenv("FIXTURES_AGENT_LLM_MODEL", "gpt-5.4-nano")

SYSTEM_PROMPT = (
    "You are the Fixtures specialist on a Fantasy Premier League selection committee. "
    "A deterministic rule has already ranked the players below by how favorable their "
    "upcoming run of fixtures is (opponent difficulty, and any blank or double gameweeks), "
    "weighted most heavily toward the very next gameweek. "
    "Write a SHORT argument - 2 to 4 sentences, under 90 words total - for backing these "
    "players on fixtures alone. Ground every claim in the listed factors: name specific "
    "gameweeks, difficulty ratings, or blank/double gameweeks. Do NOT invent fixtures or "
    "mention players not listed. Lead with the strongest pick. Plain prose, no bullet points."
)


def _fallback_reasoning(gameweek: int, picks: list[Pick]) -> str:
    if not picks:
        return "No players in the pool have a fixture run favorable enough to argue for this gameweek."
    lead = picks[0]
    who = lead.name or f"player {lead.player_id}"
    factor = lead.factors[0] if lead.factors else "a favorable run of fixtures"
    text = f"For GW{gameweek} the fixture rule's top pick is {who} - {factor}."
    if len(picks) > 1:
        others = ", ".join(p.name or f"player {p.player_id}" for p in picks[1:3])
        text += f" It also backs {others} on similar fixture strength."
    return text


def _build_user_payload(gameweek: int, picks: list[Pick]) -> str:
    lines = [f"Gameweek: {gameweek}", "Ranked picks (strongest fixture run first):"]
    for rank, p in enumerate(picks, start=1):
        who = p.name or f"player {p.player_id}"
        factors = "; ".join(p.factors) or "no fixture detail"
        lines.append(f"{rank}. {who} - conviction {p.conviction:.2f}. Fixtures: {factors}.")
    return "\n".join(lines)


def generate_reasoning(gameweek: int, picks: list[Pick], *, client_factory=default_foundry_client) -> str:
    if not picks:
        return _fallback_reasoning(gameweek, picks)
    if llm_disabled("FIXTURES_AGENT_DISABLE_LLM"):
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
