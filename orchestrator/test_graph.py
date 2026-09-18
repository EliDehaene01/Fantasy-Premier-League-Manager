"""End-to-end tests for the orchestrator's graph (orchestrator/graph.py) -
fan-out/fan-in wiring, the reaction round, Manager integration, and the
mode branch (backtest auto-accept / live interrupt-and-resume). Fake
callers stand in for the six live services and Manager's HTTP surface - the
"dry run" TODO.md's Phase 3 calls for, not a live multi-service integration
test (no Postgres, no running agent containers - Phase 4 hasn't happened
yet). Manager's own aggregation/solve logic runs for real (imported
directly), so the News-veto-propagates test is a genuine end-to-end check,
not a stub standing in for the decision itself.
"""

from __future__ import annotations

import os

os.environ["MANAGER_NARRATE_DISABLE_LLM"] = "1"
os.environ["MANAGER_REACT_DISABLE_LLM"] = "1"

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from manager.schemas import ManageRequest, ManageResult
from manager.service import manage
from orchestrator.graph import SPECIALIST_AGENTS, build_graph
from shared.contracts import AgentArgument, ChipRecommendation, ChipType, Recommendation, Veto, VetoStatus

CANDIDATE_SHAPE = [("GK", 3), ("DEF", 6), ("MID", 6), ("FWD", 4)]  # surplus over the required 2/5/5/3


def _pool():
    pool, pid, club = [], 1, 1
    for position, count in CANDIDATE_SHAPE:
        for _ in range(count):
            pool.append({"player_id": pid, "club_id": club, "position": position, "price": 4.5})
            pid += 1
            club += 1
    return pool


def _fake_specialist_factory(veto_player_id=None, chip_recommendation=None):
    def fake_specialist(agent_name, gameweek, player_pool):
        if agent_name == "stats":
            recs, score = [], 4.0
            for p in player_pool:
                recs.append(Recommendation(player_id=p["player_id"], conviction=1.0, predicted_points=score))
                score += 1.0
            return AgentArgument(agent="stats", recommendations=recs, reasoning="model output")
        if agent_name == "news" and veto_player_id is not None:
            return AgentArgument(agent="news", vetoes=[Veto(player_id=veto_player_id, status=VetoStatus.OUT, confidence=1.0)], reasoning="ruled out")
        if agent_name == "chips" and chip_recommendation is not None:
            return AgentArgument(
                agent="chips",
                chip_recommendation=ChipRecommendation(chip=chip_recommendation, confidence=1.0, reasoning="double gameweek ahead"),
                reasoning="double gameweek ahead",
            )
        return AgentArgument(agent=agent_name, reasoning=f"{agent_name} has no strong opinion this week")

    return fake_specialist


def _fake_manage(request: ManageRequest) -> ManageResult:
    return manage(request)  # the real aggregation/solve pipeline, no HTTP


def _fake_react(gameweek, first_round):
    return {name: f"{name} reacts" for name in first_round}


def _base_state(mode: str, gameweek: int = 1) -> dict:
    return {"mode": mode, "gameweek": gameweek, "player_pool": _pool(), "budget": 100.0, "free_transfers": 1, "hit_cost": 4.0}


def test_parallel_fan_out_reaches_all_six_specialists_then_manager():
    app = build_graph(agent_caller=_fake_specialist_factory(), manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver())
    result = app.invoke(_base_state("backtest"), config={"configurable": {"thread_id": "t1"}})

    assert set(result["first_round"]) == set(SPECIALIST_AGENTS)
    assert set(result["reactions"]) == set(SPECIALIST_AGENTS)
    assert result["manage_result"]["feasible"] is True
    assert result["approval_status"] == "auto_accepted"


def test_news_veto_propagates_end_to_end_through_the_graph():
    """The News agent's OUT veto (fake, but Manager's real hard-exclusion
    logic) must remove the vetoed player from the final squad even though
    they'd otherwise be the strongest pick - CLAUDE.md's hard constraint,
    checked all the way through the graph, not just at the Manager unit
    level.
    """
    best_id = _pool()[-1]["player_id"]  # fake stats assigns the highest score last, in pool order
    app = build_graph(agent_caller=_fake_specialist_factory(veto_player_id=best_id), manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver())
    result = app.invoke(_base_state("backtest"), config={"configurable": {"thread_id": "t2"}})

    assert result["manage_result"]["feasible"] is True
    squad_ids = {s["player_id"] for s in result["manage_result"]["squad"]}
    assert best_id not in squad_ids


def test_chips_agent_recommendation_drives_the_active_chip():
    """ARCHITECTURE.md 6a: chip-aware solving is "driven by the Chips
    agent's recommendation" - not a value the caller pre-sets independently
    of what Chips actually argued for this gameweek.
    """
    captured = {}

    def capturing_manage(request):
        captured["chip"] = request.chip
        return _fake_manage(request)

    app = build_graph(
        agent_caller=_fake_specialist_factory(chip_recommendation=ChipType.WILDCARD),
        manage_caller=capturing_manage, react_caller=_fake_react, checkpointer=InMemorySaver(),
    )
    # Deliberately do NOT set state["chip"] - it must come from the Chips
    # agent's first-round output, not a pre-set field.
    result = app.invoke(_base_state("backtest"), config={"configurable": {"thread_id": "t7"}})

    assert result["manage_result"]["feasible"] is True
    assert captured["chip"] == ChipType.WILDCARD


def _transfer_scenario_pool():
    """15 mediocre "current squad" players plus 15 clearly-better
    alternatives, each at a distinct club - enough surplus/contrast that
    the solver would want to replace most of the old squad if it could
    afford to, but adopting more than one of them costs a hit under a
    normal 1-free-transfer week.
    """
    pool, pid, club = [], 1, 1
    for position, count in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        for _ in range(count):
            pool.append({"player_id": pid, "club_id": club, "position": position, "price": 4.5, "in_current_squad": True})
            pid += 1
            club += 1
    for position, count in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        for _ in range(count):
            pool.append({"player_id": pid, "club_id": club, "position": position, "price": 4.5})
            pid += 1
            club += 1
    return pool


def _fake_specialist_with_dominant_new_squad(chip_recommendation):
    old_ids = {p["player_id"] for p in _transfer_scenario_pool() if p.get("in_current_squad")}

    def fake_specialist(agent_name, gameweek, player_pool):
        if agent_name == "stats":
            recs = [
                Recommendation(player_id=p["player_id"], conviction=1.0, predicted_points=(1.0 if p["player_id"] in old_ids else 20.0))
                for p in player_pool
            ]
            return AgentArgument(agent="stats", recommendations=recs, reasoning="model output")
        if agent_name == "chips" and chip_recommendation is not None:
            return AgentArgument(
                agent="chips",
                chip_recommendation=ChipRecommendation(chip=chip_recommendation, confidence=1.0, reasoning="fixture swing justifies it"),
                reasoning="fixture swing justifies it",
            )
        return AgentArgument(agent=agent_name, reasoning=f"{agent_name} has no strong opinion this week")

    return fake_specialist


def test_managers_final_squad_actually_reflects_the_chip_not_just_the_request_field():
    """Distinguishes "the chip value was passed to Manager" (the earlier
    bug - Manager wasn't even receiving it) from "the chip changed
    Manager's actual decision" (what the fix needs to be worth anything).
    Same dominant-alternative-squad scenario run twice, chip vs. no chip -
    only the Wildcard run may take zero-cost transfers beyond the free
    allowance; the no-chip control run, facing the identical incentive to
    replace the whole squad, must NOT get them for free.
    """
    state = {**_base_state("backtest"), "player_pool": _transfer_scenario_pool(), "free_transfers": 1}

    wildcard_app = build_graph(
        agent_caller=_fake_specialist_with_dominant_new_squad(ChipType.WILDCARD),
        manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver(),
    )
    wildcard_result = wildcard_app.invoke(state, config={"configurable": {"thread_id": "chip-final-a"}})

    no_chip_app = build_graph(
        agent_caller=_fake_specialist_with_dominant_new_squad(None),
        manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver(),
    )
    no_chip_result = no_chip_app.invoke(state, config={"configurable": {"thread_id": "chip-final-b"}})

    wildcard_manage = wildcard_result["manage_result"]
    no_chip_manage = no_chip_result["manage_result"]
    assert wildcard_manage["feasible"] and no_chip_manage["feasible"]

    # The actual, final decision - not a request field - proves the chip
    # took effect: many transfers, zero hit cost, only under Wildcard.
    assert wildcard_manage["transfers_made"] > 1
    assert wildcard_manage["hits_taken"] == 0
    assert wildcard_manage["hit_points_cost"] == 0.0

    # Same incentive, no chip: the solver may still take one free transfer,
    # but reaching for more of the dominant squad must cost a hit or simply
    # not happen - it cannot ALSO end up hits_taken == 0 with more than one
    # transfer, or the chip made no real difference to the outcome.
    assert not (no_chip_manage["transfers_made"] > 1 and no_chip_manage["hits_taken"] == 0)


def test_backtest_mode_auto_accepts_without_pausing():
    app = build_graph(agent_caller=_fake_specialist_factory(), manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver())
    result = app.invoke(_base_state("backtest"), config={"configurable": {"thread_id": "t3"}})

    assert "__interrupt__" not in result
    assert result["approval_status"] == "auto_accepted"


def test_live_mode_pauses_for_approval_then_resumes():
    app = build_graph(agent_caller=_fake_specialist_factory(), manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t4"}}

    paused = app.invoke(_base_state("live"), config=config)
    assert "__interrupt__" in paused
    assert app.get_state(config).next == ("mode_branch",)

    resumed = app.invoke(Command(resume="approve"), config=config)
    assert resumed["approval_status"] == "approved"


def test_live_mode_rejection_is_logged_not_retried():
    app = build_graph(agent_caller=_fake_specialist_factory(), manage_caller=_fake_manage, react_caller=_fake_react, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "t5"}}

    app.invoke(_base_state("live", gameweek=2), config=config)
    resumed = app.invoke(Command(resume="reject"), config=config)

    assert resumed["approval_status"] == "rejected"
    assert app.get_state(config).next == ()  # graph ended - no retry/negotiation loop


def test_infeasible_manager_result_declines_without_reaching_mode_branch():
    infeasible_manage = lambda request: ManageResult(feasible=False, message="no valid plan found", narration="")
    app = build_graph(agent_caller=_fake_specialist_factory(), manage_caller=infeasible_manage, react_caller=_fake_react, checkpointer=InMemorySaver())
    result = app.invoke(_base_state("live"), config={"configurable": {"thread_id": "t6"}})

    assert "__interrupt__" not in result  # never reached the human-approval pause
    assert result["approval_status"] == "declined"
    assert result["manage_result"]["message"] == "no valid plan found"
