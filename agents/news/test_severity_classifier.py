"""Tests for the availability-severity labeling rule (see
severity_classifier.py's module docstring and docs/severity_classifier.md
section 1 for the full reasoning). Pure function, no LLM, no network, no DB.
"""

from __future__ import annotations

from agents.news.severity_classifier import _label_chunk


def test_low_percentage_labels_out():
    assert _label_chunk("Maatsen (Aston Villa, DEF): Ankle injury - 25% chance of playing") == "OUT"


def test_mid_percentage_labels_doubt():
    assert _label_chunk("Mendy (Hull City, DEF): Concussion - 50% chance of playing") == "DOUBT"


def test_high_percentage_labels_fit():
    assert _label_chunk("Someone (Team, MID): Knock - 75% chance of playing") == "FIT"


def test_suspension_labels_out():
    assert _label_chunk("Awoniyi (Coventry City, FWD): Suspended until 19 Oct") == "OUT"


def test_transfer_out_labels_out():
    assert _label_chunk("Moore (Spurs, MID): Has joined FC Koln on loan for the rest of the season") == "OUT"


def test_unknown_return_date_labels_out():
    assert _label_chunk("Saliba (Arsenal, DEF): Back injury - Unknown return date") == "OUT"


def test_expected_back_no_percentage_labels_out():
    assert _label_chunk("Emegha (Chelsea, FWD): Hamstring injury - Expected back 18 Sep") == "OUT"
