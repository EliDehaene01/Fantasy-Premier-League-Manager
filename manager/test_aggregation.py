"""Tests for the Manager's aggregation math (manager/aggregation.py) -
ARCHITECTURE.md 6b's multiplicative formula, veto handling, and captain
selection. Pure functions, no DB/LLM/HTTP - fast and fully offline.
"""

from __future__ import annotations

import pytest

from manager import config
from manager.aggregation import compute_adjusted_scores, select_captain
from shared.contracts import AgentArgument, Recommendation, Veto, VetoStatus


def _agent(name, recs=None, vetoes=None, reasoning="because"):
    return AgentArgument(agent=name, recommendations=recs or [], vetoes=vetoes or [], reasoning=reasoning)


def test_multiplicative_formula_matches_hand_calculation():
    stats = _agent("stats", recs=[Recommendation(player_id=1, conviction=1.0, predicted_points=10.0)])
    adjustments = {
        "fixtures": _agent("fixtures", recs=[Recommendation(player_id=1, conviction=0.5, predicted_points=None)]),
        "contrarian": _agent("contrarian", recs=[Recommendation(player_id=1, conviction=0.2, predicted_points=8.0)]),
    }
    scores = compute_adjusted_scores(stats, adjustments)

    expected_adjustment = config.AGENT_WEIGHTS["fixtures"] * 0.5 + config.AGENT_WEIGHTS["contrarian"] * 0.2
    expected = 10.0 * (1.0 + expected_adjustment)
    assert scores[1].adjusted_score == pytest.approx(expected)
    assert scores[1].predicted_points == 10.0  # untouched, distinct from adjusted_score


def test_missing_opinions_default_to_zero_adjustment():
    stats = _agent("stats", recs=[Recommendation(player_id=1, conviction=1.0, predicted_points=10.0)])
    scores = compute_adjusted_scores(stats, {})  # no adjustment agents at all
    assert scores[1].adjusted_score == pytest.approx(10.0)


def test_player_not_in_stats_is_absent_even_if_another_agent_names_them():
    stats = _agent("stats", recs=[])
    adjustments = {"fixtures": _agent("fixtures", recs=[Recommendation(player_id=99, conviction=1.0)])}
    scores = compute_adjusted_scores(stats, adjustments)
    assert 99 not in scores


def test_news_doubt_applies_confidence_scaled_penalty_not_a_binary_exclusion():
    stats = _agent("stats", recs=[Recommendation(player_id=1, conviction=1.0, predicted_points=10.0)])
    news = _agent("news", vetoes=[Veto(player_id=1, status=VetoStatus.DOUBT, confidence=0.8)])
    scores = compute_adjusted_scores(stats, {"news": news})

    assert scores[1].veto == VetoStatus.DOUBT
    assert scores[1].adjusted_score == pytest.approx(10.0 * (1.0 - 0.8))
    assert scores[1].adjusted_score > 0.0  # never a hard zero - still squaddable


def test_news_out_is_tagged_but_score_left_for_the_solver_to_hard_exclude():
    stats = _agent("stats", recs=[Recommendation(player_id=1, conviction=1.0, predicted_points=10.0)])
    news = _agent("news", vetoes=[Veto(player_id=1, status=VetoStatus.OUT, confidence=1.0)])
    scores = compute_adjusted_scores(stats, {"news": news})

    assert scores[1].veto == VetoStatus.OUT
    assert scores[1].adjusted_score == pytest.approx(10.0)  # untouched - exclusion happens downstream


def test_captain_and_vice_are_highest_and_second_highest_among_starters():
    stats = _agent(
        "stats",
        recs=[
            Recommendation(player_id=1, conviction=1.0, predicted_points=5.0),
            Recommendation(player_id=2, conviction=1.0, predicted_points=9.0),
            Recommendation(player_id=3, conviction=1.0, predicted_points=7.0),
        ],
    )
    scores = compute_adjusted_scores(stats, {})
    captain, vice = select_captain(scores, starting_ids={1, 2, 3})
    assert captain == 2
    assert vice == 3


def test_captain_selection_ignores_bench_players():
    stats = _agent(
        "stats",
        recs=[
            Recommendation(player_id=1, conviction=1.0, predicted_points=100.0),  # benched, excluded below
            Recommendation(player_id=2, conviction=1.0, predicted_points=5.0),
            Recommendation(player_id=3, conviction=1.0, predicted_points=3.0),
        ],
    )
    scores = compute_adjusted_scores(stats, {})
    captain, vice = select_captain(scores, starting_ids={2, 3})
    assert captain == 2
    assert vice == 3


def test_weights_are_fixed_config_values_not_llm_decided():
    # Regression pin on ARCHITECTURE.md 6b's documented baseline - a change
    # here should be a deliberate retune, not an accident.
    assert config.AGENT_WEIGHTS == {"fixtures": 1.0, "contrarian": 1.0, "template": 1.0, "news": 1.3}
