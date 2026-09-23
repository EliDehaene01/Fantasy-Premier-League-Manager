"""End-to-end tests for the Manager service (manager/service.py), calling
manage() directly (no HTTP/DB) - exercises the full
aggregate -> solve -> captain -> narrate pipeline together.
"""

from __future__ import annotations

import os

os.environ["MANAGER_NARRATE_DISABLE_LLM"] = "1"
os.environ["MANAGER_REACT_DISABLE_LLM"] = "1"

from manager.schemas import ManageRequest, ManagerPlayerFact
from manager.service import manage
from shared.contracts import AgentArgument, Recommendation, Veto, VetoStatus

CANDIDATE_SHAPE = [("GK", 3), ("DEF", 6), ("MID", 6), ("FWD", 4)]  # surplus over the required 2/5/5/3


def _pool():
    pool, pid, club = [], 1, 1
    for position, count in CANDIDATE_SHAPE:
        for _ in range(count):
            pool.append(ManagerPlayerFact(player_id=pid, club_id=club, position=position, price=4.5))
            pid += 1
            club += 1
    return pool


def _stats():
    recs, pid, score = [], 1, 4.0
    for position, count in CANDIDATE_SHAPE:
        for _ in range(count):
            recs.append(Recommendation(player_id=pid, conviction=1.0, predicted_points=score))
            pid += 1
            score += 1.0
    return AgentArgument(agent="stats", recommendations=recs, reasoning="model output")


def test_end_to_end_produces_a_feasible_squad_with_captain_and_narration():
    request = ManageRequest(gameweek=1, player_pool=_pool(), stats=_stats())
    result = manage(request)

    assert result.feasible
    assert len(result.squad) == 15
    assert result.captain_id is not None
    assert result.vice_captain_id is not None
    assert result.captain_id != result.vice_captain_id
    assert isinstance(result.narration, str) and result.narration


def test_news_out_veto_is_a_hard_constraint_end_to_end():
    """CLAUDE.md: "The Manager must never select a player the News agent
    has flagged as unavailable in vetoes, regardless of predicted points."
    Vetoing the single highest-scoring player OUT must remove them from the
    result even though nothing else beats their score.
    """
    stats = _stats()
    best_id = max(stats.recommendations, key=lambda r: r.predicted_points).player_id
    news = AgentArgument(agent="news", vetoes=[Veto(player_id=best_id, status=VetoStatus.OUT, confidence=1.0)], reasoning="ruled out")

    request = ManageRequest(gameweek=1, player_pool=_pool(), stats=stats, adjustments={"news": news})
    result = manage(request)

    assert result.feasible
    squad_ids = {s.player_id for s in result.squad}
    assert best_id not in squad_ids


def test_news_recommendations_are_a_normal_vote_not_a_veto():
    """CLAUDE.md: News's `recommendations` "get weighed like any other
    specialist's, not treated as authoritative." A recommendation alone
    (no veto) must never exclude a player.
    """
    stats = _stats()
    best_id = max(stats.recommendations, key=lambda r: r.predicted_points).player_id
    news = AgentArgument(
        agent="news",
        recommendations=[Recommendation(player_id=best_id, conviction=1.0, predicted_points=None)],
        reasoning="notable positive coverage",
    )
    request = ManageRequest(gameweek=1, player_pool=_pool(), stats=stats, adjustments={"news": news})
    result = manage(request)

    assert result.feasible
    squad_ids = {s.player_id for s in result.squad}
    assert best_id in squad_ids  # a recommendation is a vote, never grounds for exclusion


def test_reproducible_same_inputs_produce_the_same_squad():
    """CLAUDE.md: "The same inputs must always produce the same squad.\""""
    request = ManageRequest(gameweek=1, player_pool=_pool(), stats=_stats())
    first = manage(request)
    second = manage(request)
    assert first.model_dump(exclude={"narration"}) == second.model_dump(exclude={"narration"})


def test_current_squad_is_always_a_representable_candidate_even_if_stats_top_k_omits_it():
    """Real-world bug this reproduces: Stats' own top_k (sized for the
    reaction-round transcript, not the solver - orchestrator/callers.py)
    can genuinely omit real current-squad members at live scale, and
    "keep the current squad" must still be feasible regardless. Without
    manager/aggregation.py's fix, none of these 15 squad members would be
    solver candidates at all (Stats scored 5 entirely different players),
    so even the trivial zero-transfer solution would be unreachable and
    the whole gameweek would come back "no valid plan found" - found by
    running a real 659-player gameweek through the deployed cluster.
    """
    squad, pid, club = [], 1, 1
    for position, count in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        for _ in range(count):
            squad.append(ManagerPlayerFact(player_id=pid, club_id=club, position=position, price=4.5, in_current_squad=True))
            pid += 1
            club += 1

    other_recs = [Recommendation(player_id=100 + i, conviction=1.0, predicted_points=5.0) for i in range(5)]
    stats = AgentArgument(agent="stats", recommendations=other_recs, reasoning="model output")

    request = ManageRequest(gameweek=1, player_pool=squad, stats=stats)
    result = manage(request)

    assert result.feasible
    assert len(result.squad) == 15


def test_infeasible_plan_surfaces_a_message_not_a_degenerate_squad():
    request = ManageRequest(gameweek=1, player_pool=_pool(), stats=_stats(), budget=1.0)
    result = manage(request)

    assert result.feasible is False
    assert result.squad == []
    assert result.message == "no valid plan found"
