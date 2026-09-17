"""ARCHITECTURE.md 6b's bounded reaction round: text-only, decision-inert.
Deliberately kept in its own module with its own narrow return type
(``dict[str, str]`` - plain strings) so there is no code path by which a
reaction could touch `conviction`/`recommendations`/`vetoes` - the function
signature itself makes that mistake hard to make, not just discipline. See
test_reaction.py's test_reaction_round_cannot_change_the_final_squad for the
enforcement test CLAUDE.md calls for ("the reaction round is text-only and
decision-inert").
"""

from __future__ import annotations

from shared.agent_service import call_foundry_llm, llm_disabled
from shared.contracts import AgentArgument

REACT_DISABLE_LLM_ENV = "MANAGER_REACT_DISABLE_LLM"


def generate_reactions(gameweek: int, first_round: dict[str, AgentArgument]) -> dict[str, str]:
    """One short reaction per agent that produced a first-round output,
    given everyone else's first-round reasoning text. Best-effort: an LLM
    failure (or MANAGER_REACT_DISABLE_LLM=1, e.g. tests/backtest) falls back
    to a templated sentence rather than blocking the round - this is
    transcript flavor, never load-bearing for the decision.
    """
    return {
        agent_name: _react(gameweek, agent_name, own, first_round)
        for agent_name, own in first_round.items()
    }


def _react(gameweek: int, agent_name: str, own: AgentArgument, first_round: dict[str, AgentArgument]) -> str:
    if llm_disabled(REACT_DISABLE_LLM_ENV):
        return f"{agent_name} has nothing further to add."
    others = "\n".join(f"- {n}: {a.reasoning}" for n, a in first_round.items() if n != agent_name)
    try:
        return call_foundry_llm(
            model="gpt-4o-mini",
            system_prompt=(
                f"You are the {agent_name} agent in an FPL squad debate. You already made your "
                "case this round. Write ONE short (1-2 sentence) reaction to the other agents' "
                "arguments below - agree, push back, or note a conflict. This is commentary only: "
                "it cannot change your recommendation, conviction, or any veto."
            ),
            user_content=f"Gameweek {gameweek}.\n\nYour argument: {own.reasoning}\n\nOther agents:\n{others}",
            max_completion_tokens=120,
        )
    except Exception:
        return f"{agent_name} has nothing further to add."
