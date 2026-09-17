"""ARCHITECTURE.md 6b: "The LLM call's role is narration only: it explains
the already-computed decision... it does not make the decision." Every
field `narrate` receives (the summary text) is already final by the time
this is called - weights, vetoes, solver result, and captain are all fixed
before this module ever runs.
"""

from __future__ import annotations

from shared.agent_service import call_foundry_llm, llm_disabled

NARRATE_DISABLE_LLM_ENV = "MANAGER_NARRATE_DISABLE_LLM"


def narrate(gameweek: int, summary: str) -> str:
    """``summary`` is a deterministic plain-text recap of the already-final
    decision, built by the caller - used both as the LLM's input and as the
    fallback text if the LLM call fails or is disabled.
    """
    if llm_disabled(NARRATE_DISABLE_LLM_ENV):
        return summary
    try:
        return call_foundry_llm(
            model="gpt-4o-mini",
            system_prompt=(
                "You are the Manager agent for a Fantasy Premier League squad. Explain the "
                "squad decision already made below in 2-4 sentences for a human reviewing it - "
                "why this squad, why this captain, any notable trade-off. Do not suggest changes; "
                "the decision is final, you are only narrating it."
            ),
            user_content=f"Gameweek {gameweek}.\n\n{summary}",
            max_completion_tokens=200,
        )
    except Exception:
        return summary
