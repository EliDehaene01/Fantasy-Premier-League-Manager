"""Unit tests for ingestion/deadline.py - pure functions over a bootstrap
dict, no Postgres needed (unlike most of test_ingestion.py), so this stays
its own fast file.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ingestion.deadline import hours_until_next_deadline, next_gameweek_id, should_trigger

NOW = datetime(2025, 9, 10, 12, 0, 0, tzinfo=timezone.utc)


def _bootstrap(deadline_times: list[str]) -> dict:
    return {
        "events": [
            {"id": i + 1, "deadline_time": dt, "finished": False, "is_current": False}
            for i, dt in enumerate(deadline_times)
        ]
    }


def test_no_ops_when_next_deadline_is_far_out():
    # Next deadline six days away - well outside the 24-36h trigger window.
    bootstrap = _bootstrap(["2025-09-16T17:30:00Z"])
    assert should_trigger(bootstrap, now=NOW) is False


def test_triggers_when_next_deadline_is_within_the_window():
    # ~30 hours away.
    bootstrap = _bootstrap(["2025-09-11T18:00:00Z"])
    assert should_trigger(bootstrap, now=NOW) is True


def test_triggers_at_the_window_boundary():
    bootstrap = _bootstrap(["2025-09-12T00:00:00Z"])  # exactly 36h from NOW
    assert should_trigger(bootstrap, window_hours=36.0, now=NOW) is True


def test_does_not_trigger_just_past_the_window_boundary():
    bootstrap = _bootstrap(["2025-09-12T00:00:01Z"])  # 36h + 1s from NOW
    assert should_trigger(bootstrap, window_hours=36.0, now=NOW) is False


def test_ignores_already_passed_deadlines_and_picks_the_next_upcoming_one():
    # GW1's deadline already passed; GW2 is the real "next deadline", ~30h out.
    bootstrap = _bootstrap(["2025-09-01T10:00:00Z", "2025-09-11T18:00:00Z"])
    assert should_trigger(bootstrap, now=NOW) is True
    hours = hours_until_next_deadline(bootstrap, now=NOW)
    assert 29.5 < hours < 30.5


def test_no_op_at_end_of_season_with_no_upcoming_deadlines():
    bootstrap = _bootstrap(["2025-01-01T10:00:00Z"])  # only a past deadline
    assert hours_until_next_deadline(bootstrap, now=NOW) is None
    assert should_trigger(bootstrap, now=NOW) is False


def test_next_gameweek_id_picks_the_soonest_upcoming_deadline_not_index_order():
    bootstrap = _bootstrap(["2025-09-01T10:00:00Z", "2025-09-20T18:00:00Z", "2025-09-11T18:00:00Z"])
    assert next_gameweek_id(bootstrap, now=NOW) == 3  # event id 3's deadline (index 2) is the soonest upcoming one


def test_next_gameweek_id_is_none_with_no_upcoming_deadlines():
    bootstrap = _bootstrap(["2025-01-01T10:00:00Z"])
    assert next_gameweek_id(bootstrap, now=NOW) is None
