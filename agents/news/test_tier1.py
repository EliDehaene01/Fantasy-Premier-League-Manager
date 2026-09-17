"""Tests for Tier 1's deterministic availability rules (see tier1.py's
module docstring for the exact thresholds). No LLM, no network, no DB -
these are pure function calls, asserted directly.
"""

from __future__ import annotations

from shared.contracts import VetoStatus
from agents.news.tier1 import check_tier1


def test_clearly_out_by_zero_percent_resolves_with_no_llm_call():
    veto = check_tier1(1, chance_of_playing=0, news="Hamstring injury")
    assert veto is not None
    assert veto.status == VetoStatus.OUT
    assert veto.confidence == 1.0


def test_clearly_out_by_explicit_suspension_text():
    veto = check_tier1(2, chance_of_playing=None, news="Suspended until 10 Oct")
    assert veto is not None
    assert veto.status == VetoStatus.OUT


def test_clearly_fit_with_no_news_and_no_chance_value():
    veto = check_tier1(3, chance_of_playing=None, news=None)
    assert veto is not None
    assert veto.status == VetoStatus.FIT
    assert veto.confidence == 1.0


def test_clearly_fit_at_100_percent():
    veto = check_tier1(4, chance_of_playing=100, news=None)
    assert veto is not None
    assert veto.status == VetoStatus.FIT


def test_ambiguous_middle_is_left_unresolved_for_tier_2():
    # 75% with no explicit ruled-out phrase - exactly the case Tier 1 must
    # NOT guess at.
    veto = check_tier1(5, chance_of_playing=75, news="Hamstring injury - 75% chance of playing")
    assert veto is None


def test_ambiguous_missing_chance_but_vague_news_is_unresolved():
    veto = check_tier1(6, chance_of_playing=None, news="Assessed ahead of the weekend")
    assert veto is None
