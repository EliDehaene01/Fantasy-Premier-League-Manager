"""ARCHITECTURE.md 6b: turns six agents' first-round outputs into one
adjusted_score per player, plus deterministic captain/vice-captain
selection. Pure functions, no DB/LLM/HTTP - fully deterministic given the
same AgentArgument inputs, satisfying CLAUDE.md's hard constraint that the
same inputs always produce the same squad.
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.contracts import AgentArgument, VetoStatus

from .config import AGENT_WEIGHTS, doubt_multiplier
from .schemas import ManagerPlayerFact

# The four agents whose `recommendations` are a normal, weighted vote - News
# here means its `recommendations` half (notable positive coverage), never
# its `vetoes` half, which is handled entirely separately below
# (CLAUDE.md: "Only the News agent's vetoes are a hard constraint... its
# recommendations are a normal vote").
ADJUSTMENT_AGENTS = ("fixtures", "contrarian", "template", "news")


@dataclass
class PlayerScore:
    player_id: int
    predicted_points: float
    adjusted_score: float
    veto: VetoStatus | None = None


def compute_adjusted_scores(
    stats: AgentArgument,
    adjustments: dict[str, AgentArgument],
    player_pool: list[ManagerPlayerFact] | None = None,
) -> dict[int, PlayerScore]:
    """``adjustments`` is keyed by agent name - any subset of
    ``ADJUSTMENT_AGENTS`` may be present; a missing agent contributes zero
    adjustment to every player, per ARCHITECTURE.md 6b ("Missing opinions
    default to zero adjustment... not a crash or a worst-case assumption").

    Stats' recommendations set every SCORED player's base number - it's the
    only agent with a real `predicted_points` (ARCHITECTURE.md 6b: "Stats'
    predicted_points is the base currency"). But Stats' own `top_k` (sized
    for the reaction-round transcript, which only wants a short list - see
    orchestrator/callers.py) can genuinely omit real current-squad players
    at live scale, and "the solver considered keeping the current squad" is
    not optional the way "Stats had an opinion on every bench player" is -
    a squad member Stats didn't score still gets a candidate-pool entry
    here, scored 0.0 (last, not first, in the objective - just no longer
    impossible to select). ``player_pool`` is optional and defaults to none
    so every pre-existing caller (unit tests with no squad-retention
    scenario in play) is unaffected.
    """
    scores: dict[int, PlayerScore] = {}
    for rec in stats.recommendations:
        if rec.predicted_points is None:
            continue  # Stats always sets this in practice; guard regardless
        scores[rec.player_id] = PlayerScore(
            player_id=rec.player_id, predicted_points=rec.predicted_points, adjusted_score=rec.predicted_points
        )

    for fact in player_pool or []:
        if fact.in_current_squad and fact.player_id not in scores:
            scores[fact.player_id] = PlayerScore(player_id=fact.player_id, predicted_points=0.0, adjusted_score=0.0)

    conviction_by_agent: dict[str, dict[int, float]] = {
        agent_name: (
            {r.player_id: r.conviction for r in adjustments[agent_name].recommendations}
            if agent_name in adjustments
            else {}
        )
        for agent_name in ADJUSTMENT_AGENTS
    }

    for player_id, score in scores.items():
        total_adjustment = sum(
            AGENT_WEIGHTS[agent_name] * conviction_by_agent[agent_name].get(player_id, 0.0)
            for agent_name in ADJUSTMENT_AGENTS
        )
        score.adjusted_score = score.predicted_points * (1.0 + total_adjustment)

    news = adjustments.get("news")
    if news is not None:
        for veto in news.vetoes:
            score = scores.get(veto.player_id)
            if score is None:
                continue
            score.veto = veto.status
            if veto.status == VetoStatus.DOUBT:
                score.adjusted_score *= doubt_multiplier(veto.confidence)
            # OUT is tagged here (so the caller can flag it to the solver as
            # `vetoed_out`) but NOT zeroed - the hard exclusion is the
            # solver's job (ARCHITECTURE.md 6a), not a score adjustment.

    return scores


def select_captain(scores: dict[int, PlayerScore], starting_ids: set[int]) -> tuple[int | None, int | None]:
    """Deterministic captain/vice-captain: highest/second-highest
    adjusted_score among the finalized starting 11 (ARCHITECTURE.md 6b) -
    never an LLM judgment call.
    """
    starters = sorted(
        (scores[pid] for pid in starting_ids if pid in scores),
        key=lambda s: s.adjusted_score,
        reverse=True,
    )
    captain_id = starters[0].player_id if starters else None
    vice_id = starters[1].player_id if len(starters) > 1 else None
    return captain_id, vice_id
