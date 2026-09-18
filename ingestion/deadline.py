"""Deadline-aware trigger check for the weekly CronJob (ARCHITECTURE.md 8:
"runs daily, checks the FPL fixtures endpoint for the next deadline; if it's
within ~24-36 hours, triggers the pipeline Job"). Kept DB-free and
side-effect-free (a plain bootstrap-static GET, no landing/writes) so it's
cheap to run once a day even on the vast majority of days it's a no-op.
"""

from __future__ import annotations

from datetime import datetime, timezone

DEFAULT_WINDOW_HOURS = 36.0


def _parse_deadline(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def hours_until_next_deadline(bootstrap: dict, *, now: datetime | None = None) -> float | None:
    """Hours until the earliest not-yet-passed gameweek deadline, or None if
    every deadline in bootstrap-static has already passed (end of season).
    """
    now = now or datetime.now(timezone.utc)
    upcoming = [d for e in bootstrap["events"] if (d := _parse_deadline(e["deadline_time"])) > now]
    if not upcoming:
        return None
    return (min(upcoming) - now).total_seconds() / 3600.0


def should_trigger(bootstrap: dict, *, window_hours: float = DEFAULT_WINDOW_HOURS, now: datetime | None = None) -> bool:
    hours = hours_until_next_deadline(bootstrap, now=now)
    return hours is not None and hours <= window_hours
