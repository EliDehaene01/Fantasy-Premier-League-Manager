"""Tests for the monitoring job (agents/stats/monitor.py).

Runs against a real Postgres instance, isolated in its own schema
(``test_monitor``, truncated before each test) - the same pattern
``ingestion/test_ingestion.py`` already established, so this never touches
the real predictions_log/monitoring_runs data.

Covers the three things the task asks for:
  1. Rolling RMSE over a small synthetic predictions log matches a
     hand-calculated expected value.
  2. Threshold/trigger logic fires when it should and doesn't when it
     shouldn't (a clearly-degraded case and a clearly-fine case).
  3. A version tag correctly isolates pre- and post-retrain predictions -
     rolling_rmse for one model_version never picks up another's rows.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from agents.stats import monitor
from ingestion.db import get_connection

TEST_SCHEMA = "test_monitor"
_ALL_TABLES = ["predictions_log", "player_gameweek_stats", "players", "monitoring_runs", "retrain_log"]


def _truncate_all(conn) -> None:
    conn.execute(f"TRUNCATE TABLE {', '.join(_ALL_TABLES)} RESTART IDENTITY CASCADE")
    conn.commit()


@pytest.fixture
def conn():
    c = get_connection(schema=TEST_SCHEMA)
    _truncate_all(c)
    yield c
    c.close()


def _seed_actuals(conn, rows: list[tuple[int, int, int]]) -> None:
    """rows: (player_id, gw, total_points)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO player_gameweek_stats (player_id, gw, total_points, source, ingested_at)
        VALUES (%s, %s, %s, 'test', %s)
        ON CONFLICT (player_id, gw) DO UPDATE SET total_points = excluded.total_points
        """,
        [(pid, gw, pts, now) for pid, gw, pts in rows],
    )
    conn.commit()


def _seed_predictions(conn, rows: list[tuple[int, int, float, str]]) -> None:
    """rows: (player_id, gw, predicted_points, model_version)."""
    now = datetime.now(timezone.utc).isoformat()
    conn.executemany(
        """
        INSERT INTO predictions_log (player_id, gw, predicted_points, model_version, logged_at)
        VALUES (%s, %s, %s, %s, %s)
        """,
        [(pid, gw, pred, mv, now) for pid, gw, pred, mv in rows],
    )
    conn.commit()


def test_rolling_rmse_matches_hand_calculation(conn):
    # 2 players x 3 gameweeks, all one model_version.
    _seed_actuals(conn, [(1, 10, 6), (1, 11, 2), (1, 12, 10), (2, 10, 4), (2, 11, 8), (2, 12, 0)])
    _seed_predictions(conn, [
        (1, 10, 5.0, "v1"), (1, 11, 4.0, "v1"), (1, 12, 8.0, "v1"),
        (2, 10, 4.0, "v1"), (2, 11, 6.0, "v1"), (2, 12, 3.0, "v1"),
    ])
    # errors: (5-6)=-1, (4-2)=2, (8-10)=-2, (4-4)=0, (6-8)=-2, (3-0)=3
    # squared: 1, 4, 4, 0, 4, 9 -> mean=22/6 -> sqrt
    expected_rmse = (sum(e ** 2 for e in [-1, 2, -2, 0, -2, 3]) / 6) ** 0.5

    result = monitor.rolling_rmse(conn, "v1", window=3)
    assert result["rmse"] == pytest.approx(expected_rmse)
    assert result["n_gameweeks_used"] == 3
    assert result["gameweeks_used"] == [10, 11, 12]
    assert result["n_predictions"] == 6


def test_rolling_rmse_uses_latest_prediction_per_player_gw(conn):
    """A player re-scored twice for the same gameweek: only the LATEST
    logged prediction should count, not both (predictions_log is append-only
    and can carry more than one row per (player, gw))."""
    _seed_actuals(conn, [(1, 10, 5)])
    now1 = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
    now2 = datetime(2026, 1, 2, tzinfo=timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO predictions_log (player_id, gw, predicted_points, model_version, logged_at) "
        "VALUES (%s, %s, %s, %s, %s)",
        (1, 10, 1.0, "v1", now1),  # stale first guess
    )
    conn.execute(
        "INSERT INTO predictions_log (player_id, gw, predicted_points, model_version, logged_at) "
        "VALUES (%s, %s, %s, %s, %s)",
        (1, 10, 5.0, "v1", now2),  # corrected, later prediction - exactly right
    )
    conn.commit()

    result = monitor.rolling_rmse(conn, "v1", window=5)
    assert result["n_predictions"] == 1
    assert result["rmse"] == pytest.approx(0.0)


def test_rolling_rmse_isolates_model_versions(conn):
    """The version tag must keep a retrain from silently blending old-model
    and new-model predictions into one rolling number."""
    # Old model: consistently way off (big errors).
    _seed_actuals(conn, [(1, 20, 10), (1, 21, 10), (1, 22, 10)])
    _seed_predictions(conn, [(1, 20, 0.0, "old"), (1, 21, 0.0, "old"), (1, 22, 0.0, "old")])
    # New model: exactly right.
    _seed_predictions(conn, [(1, 20, 10.0, "new"), (1, 21, 10.0, "new"), (1, 22, 10.0, "new")])

    old_result = monitor.rolling_rmse(conn, "old", window=5)
    new_result = monitor.rolling_rmse(conn, "new", window=5)

    assert old_result["rmse"] == pytest.approx(10.0)
    assert new_result["rmse"] == pytest.approx(0.0)
    # Neither model's number is anywhere near an average of the two - proof
    # they were never blended together.
    assert new_result["rmse"] != pytest.approx(old_result["rmse"], abs=1.0)


def _baseline(model_version="v1", rmse=1.0):
    return {"model_version": model_version, "holdout_metrics": {"rmse": rmse}}


def test_threshold_fires_when_clearly_degraded(conn, monkeypatch):
    monkeypatch.setattr(monitor, "read_baseline", lambda: _baseline("v1", rmse=1.0))
    monkeypatch.setattr(monitor, "run_retrain", lambda reason, conn: {"stub": True, "reason": reason})

    # Rolling RMSE far above baseline (1.0 * 1.10 = 1.10 threshold at 10%).
    _seed_actuals(conn, [(1, 30, 0), (1, 31, 0), (1, 32, 0)])
    _seed_predictions(conn, [(1, 30, 5.0, "v1"), (1, 31, 5.0, "v1"), (1, 32, 5.0, "v1")])

    result_1 = monitor.evaluate_and_log(conn, window=3, degrade_threshold_pct=10.0, retrain_consecutive=2)
    assert result_1["is_degraded"] is True
    assert result_1["consecutive_degraded_gws"] == 1
    assert result_1["retrain_triggered"] is False  # only 1 consecutive so far

    # A second consecutive degraded run (same model_version) should trigger.
    result_2 = monitor.evaluate_and_log(conn, window=3, degrade_threshold_pct=10.0, retrain_consecutive=2)
    assert result_2["is_degraded"] is True
    assert result_2["consecutive_degraded_gws"] == 2
    assert result_2["retrain_triggered"] is True
    assert "retrain_result" in result_2


def test_threshold_does_not_fire_when_clearly_fine(conn, monkeypatch):
    monkeypatch.setattr(monitor, "read_baseline", lambda: _baseline("v1", rmse=5.0))
    monkeypatch.setattr(monitor, "run_retrain", lambda reason, conn: pytest.fail("retrain should not run"))

    # Predictions are exactly right - rolling RMSE = 0, nowhere near the
    # (generous) baseline of 5.0.
    _seed_actuals(conn, [(1, 30, 4), (1, 31, 6), (1, 32, 2)])
    _seed_predictions(conn, [(1, 30, 4.0, "v1"), (1, 31, 6.0, "v1"), (1, 32, 2.0, "v1")])

    result = monitor.evaluate_and_log(conn, window=3, degrade_threshold_pct=10.0, retrain_consecutive=2)
    assert result["is_degraded"] is False
    assert result["consecutive_degraded_gws"] == 0
    assert result["retrain_triggered"] is False
    assert "retrain_result" not in result


def test_consecutive_streak_resets_after_a_non_degraded_run(conn, monkeypatch):
    monkeypatch.setattr(monitor, "read_baseline", lambda: _baseline("v1", rmse=1.0))
    monkeypatch.setattr(monitor, "run_retrain", lambda reason, conn: {"stub": True})

    # Degraded run 1.
    _seed_actuals(conn, [(1, 30, 0)])
    _seed_predictions(conn, [(1, 30, 5.0, "v1")])
    r1 = monitor.evaluate_and_log(conn, window=1, degrade_threshold_pct=10.0, retrain_consecutive=2)
    assert r1["consecutive_degraded_gws"] == 1

    # Fine run 2 (new gameweek, prediction exactly right) - streak resets.
    _seed_actuals(conn, [(1, 31, 3)])
    _seed_predictions(conn, [(1, 31, 3.0, "v1")])
    r2 = monitor.evaluate_and_log(conn, window=1, degrade_threshold_pct=10.0, retrain_consecutive=2)
    assert r2["is_degraded"] is False
    assert r2["consecutive_degraded_gws"] == 0

    # Degraded again (run 3) - should be treated as consecutive-count 1, not 3.
    _seed_actuals(conn, [(1, 32, 0)])
    _seed_predictions(conn, [(1, 32, 5.0, "v1")])
    r3 = monitor.evaluate_and_log(conn, window=1, degrade_threshold_pct=10.0, retrain_consecutive=2)
    assert r3["consecutive_degraded_gws"] == 1
    assert r3["retrain_triggered"] is False
