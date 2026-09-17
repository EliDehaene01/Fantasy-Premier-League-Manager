"""The 'prose' half of the Chips agent - same two-step split as every other
specialist (deterministic verdict first, LLM prose second, templated
fallback if the call fails or is disabled). Uses the shared Foundry wrapper
(shared/agent_service.py) instead of its own client/try-except copy.
"""

from __future__ import annotations

import logging
import os

from shared.agent_service import call_foundry_llm, default_foundry_client, llm_disabled

from .scoring import ChipVerdict

logger = logging.getLogger("chips_agent.reasoning")

FOUNDRY_DEPLOYMENT = os.getenv("CHIPS_AGENT_LLM_MODEL", "gpt-5.4-nano")

SYSTEM_PROMPT = (
    "You are the Chips specialist on a Fantasy Premier League selection committee, "
    "arguing about WHEN to play a chip (Wildcard, Bench Boost, Triple Captain, Free Hit) "
    "rather than which players to pick. A deterministic rule has already decided whether "
    "this gameweek's fixture calendar shape (double gameweeks, blank gameweeks, fixture "
    "favorability) for OUR squad justifies playing a specific chip now, or holding. "
    "Write a SHORT argument - 2 to 4 sentences, under 90 words total - explaining that "
    "verdict. Ground every claim in the listed factors. Do NOT invent fixture details or "
    "recommend a chip the rule didn't pick. Most gameweeks the correct answer is to "
    "recommend NO chip - if so, explain why holding is right, don't manufacture urgency "
    "just to sound decisive. Plain prose, no bullet points."
)


def _fallback_reasoning(verdict: ChipVerdict) -> str:
    factor = "; ".join(verdict.reasoning_factors) or "no unusual fixture-calendar shape this gameweek"
    if verdict.chip is None:
        return f"No chip recommended this gameweek - {factor}."
    return f"Recommending {verdict.chip.value.replace('_', ' ')} this gameweek - {factor}."


def _build_user_payload(gameweek: int, verdict: ChipVerdict) -> str:
    chip_name = verdict.chip.value if verdict.chip is not None else "none"
    factors = "; ".join(verdict.reasoning_factors) or "no detail"
    return (
        f"Gameweek: {gameweek}\n"
        f"Rule's verdict: chip={chip_name}, confidence={verdict.confidence:.2f}\n"
        f"Factors: {factors}."
    )


def generate_reasoning(gameweek: int, verdict: ChipVerdict, *, client_factory=default_foundry_client) -> str:
    if llm_disabled("CHIPS_AGENT_DISABLE_LLM"):
        return _fallback_reasoning(verdict)
    try:
        return call_foundry_llm(
            model=FOUNDRY_DEPLOYMENT,
            system_prompt=SYSTEM_PROMPT,
            user_content=_build_user_payload(gameweek, verdict),
            max_completion_tokens=400,
            client_factory=client_factory,
        )
    except Exception as exc:  # noqa: BLE001 - genuinely want to catch everything here
        logger.warning("Foundry reasoning call failed (%s); using templated fallback", exc)
        return _fallback_reasoning(verdict)
