"""Integration test for backtest/benchmarks.py against the real seeded
archive (same rationale as test_engine.py - this module glues real
components together, so it isn't mocked).
"""

from __future__ import annotations

from backtest.benchmarks import compare


def test_never_transfer_baseline_is_computed_and_average_manager_is_honestly_unavailable():
    report = compare("2025-26", engine_total=100.0, start_gw=2, end_gw=3)

    assert report.never_transfer_total > 0.0
    assert report.average_manager_total is None
    assert "not available" in report.average_manager_unavailable_reason.lower() or "no per-gameweek" in report.average_manager_unavailable_reason.lower()
